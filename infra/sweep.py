#!/usr/bin/env python3
"""WandB Sweep runner for Scaling-CRL.

Creates a WandB sweep from a YAML config, then runs agents.
Each agent loads the base experiment, applies sweep overrides, and trains.

Disabled by default — activate for hyperparameter search on new papers.

Usage:
  # Create sweep and run locally
  python infra/sweep.py --config conf/sweep/example_sweep.yaml

  # Run with limited trials
  python infra/sweep.py --config conf/sweep/example_sweep.yaml --count 20

  # Create sweep only (run agents separately)
  python infra/sweep.py --config conf/sweep/example_sweep.yaml --create-only

  # Run agent for existing sweep
  python infra/sweep.py --sweep-id <entity>/<project>/<sweep_id>
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, read_write

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf")


def load_sweep_config(path):
    """Load sweep YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_sweep(sweep_cfg, entity, project):
    """Create a WandB sweep and return sweep ID."""
    sweep_config = {
        "name": sweep_cfg.get("name", sweep_cfg.get("base_experiment", "sweep")),
        "method": sweep_cfg.get("method", "bayes"),
        "metric": sweep_cfg.get("metric", {"name": "eval/episode_success_any", "goal": "maximize"}),
        "parameters": sweep_cfg["parameters"],
        "run_cap": sweep_cfg.get("run_cap", 100),
    }

    # Store base_experiment in sweep config so agents can find it
    sweep_config["base_experiment"] = sweep_cfg.get("base_experiment", "")

    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )
    return sweep_id


def run_agent(sweep_id, entity, project, count=None):
    """Run a WandB sweep agent.

    The agent picks param combos from the sweep, loads the base experiment,
    applies overrides, and runs training.
    """
    def train_fn():
        run = wandb.init()
        config = dict(wandb.config)

        # Get base experiment name
        base_experiment = config.pop("base_experiment", None)
        if not base_experiment:
            # Try to get from sweep config
            api = wandb.Api()
            sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
            base_experiment = sweep.config.get("base_experiment", "")

        if not base_experiment:
            print("ERROR: No base_experiment specified in sweep config")
            run.finish()
            return

        print(f"Sweep agent: base={base_experiment}, overrides={config}")

        # Load base experiment via Hydra
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=CONF_DIR, version_base=None)
        raw_cfg = compose(config_name=f"experiment/{base_experiment}")

        from conf.config import ExperimentConfig
        schema = OmegaConf.structured(ExperimentConfig)
        cfg = OmegaConf.merge(schema, raw_cfg)

        # Apply sweep overrides
        with read_write(cfg):
            for key, value in config.items():
                OmegaConf.update(cfg, key, value)

            # Update exp_name to include sweep info
            sweep_name = run.name or run.id
            cfg.exp_name = f"sweep_{sweep_name}"

        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        # Run training
        from train import apply_compile_check_overrides
        import jax
        import numpy as np
        import time
        import pickle
        import random
        import optax
        import flax.linen as nn
        from pathlib import Path
        from brax import envs
        from flax.training.train_state import TrainState
        from crl.networks import Actor, SA_encoder, G_encoder
        from crl.types import Transition, TrainingState
        from crl.buffer import TrajectoryUniformSamplingQueue
        from crl.algorithm import (
            make_actor_step, make_get_experience, make_update_actor,
            make_update_critic, make_sgd_step, make_training_step,
            make_training_epoch, setup_evaluator, make_prefill,
        )
        from utils.env_factory import make_env, wrap_env
        from utils.checkpoint import (
            CheckpointConfig, create_checkpoint_manager, save_checkpoint,
            restore_checkpoint,
        )

        # Compute derived values
        if cfg.eval_env_id is None:
            cfg.eval_env_id = cfg.env_id
        cfg.env_steps_per_actor_step = cfg.num_envs * cfg.unroll_length
        cfg.num_prefill_env_steps = cfg.min_replay_size * cfg.num_envs
        cfg.num_prefill_actor_steps = int(np.ceil(cfg.min_replay_size / cfg.unroll_length))
        cfg.num_training_steps_per_epoch = (cfg.total_env_steps - cfg.num_prefill_env_steps) // (cfg.num_epochs * cfg.env_steps_per_actor_step)

        # Env
        env, env_info = make_env(cfg.env_id)
        env = wrap_env(env, episode_length=cfg.episode_length)
        obs_size = env.observation_size
        action_size = env.action_size

        eval_env, _ = make_env(cfg.eval_env_id)
        eval_env = wrap_env(eval_env, episode_length=cfg.episode_length)

        cfg.obs_dim = env_info.obs_dim
        cfg.goal_start_idx = env_info.goal_start_idx
        cfg.goal_end_idx = env_info.goal_end_idx

        # Checkpoint
        save_path = Path(cfg.wandb_dir) / Path(f"runs/{cfg.exp_name}")
        os.makedirs(save_path, exist_ok=True)

        ckpt_config = CheckpointConfig(
            save_interval_epochs=cfg.checkpoint_save_interval_epochs,
            max_to_keep=cfg.checkpoint_max_to_keep,
            keep_period=cfg.checkpoint_keep_period,
        )
        ckpt_manager = create_checkpoint_manager(save_path, ckpt_config)

        # RNG
        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        key = jax.random.PRNGKey(seed)
        key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key, eval_actor_key = jax.random.split(key, 8)

        # Networks
        actor = Actor(action_size=action_size, network_width=cfg.actor_network_width,
                      network_depth=cfg.depth, use_relu=cfg.use_relu)
        sa_encoder = SA_encoder(network_width=cfg.critic_network_width, network_depth=cfg.depth,
                                use_relu=cfg.use_relu)
        g_encoder = G_encoder(network_width=cfg.critic_network_width, network_depth=cfg.depth,
                              use_relu=cfg.use_relu)

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=cfg.actor_lr))
        sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, env_info.obs_dim]), np.ones([1, action_size]))
        g_encoder_params = g_encoder.init(g_key, np.ones([1, env_info.goal_end_idx - env_info.goal_start_idx]))
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=cfg.critic_lr))
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
            tx=optax.adam(learning_rate=cfg.alpha_lr))

        training_state = TrainingState(
            env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()),
            actor_state=actor_state, critic_state=critic_state, alpha_state=alpha_state)

        # Replay buffer
        dummy_transition = Transition(
            observation=jnp.zeros((obs_size,)), action=jnp.zeros((action_size,)),
            reward=0.0, discount=0.0,
            extras={"state_extras": {"truncation": 0.0, "seed": 0.0}})
        replay_buffer = TrajectoryUniformSamplingQueue(
            max_replay_size=cfg.max_replay_size, dummy_data_sample=dummy_transition,
            sample_batch_size=cfg.batch_size, num_envs=cfg.num_envs,
            episode_length=cfg.episode_length)
        replay_buffer.insert_internal = jax.jit(replay_buffer.insert_internal)
        replay_buffer.sample_internal = jax.jit(replay_buffer.sample_internal)
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        # Build training functions
        if cfg.expl_actor == 0:
            actor_step_fn = make_actor_step(actor, env, mode="deterministic")
        elif cfg.expl_actor == 1:
            actor_step_fn = make_actor_step(actor, env, mode="stochastic")
        else:
            actor_step_fn = make_actor_step(actor, env, mode="multi_sample",
                obs_dim=env_info.obs_dim, goal_start_idx=env_info.goal_start_idx,
                goal_end_idx=env_info.goal_end_idx, sa_encoder=sa_encoder,
                g_encoder=g_encoder, K=cfg.expl_actor)

        target_entropy = -cfg.entropy_param * action_size
        get_experience = make_get_experience(actor_step_fn, replay_buffer, cfg.unroll_length)
        prefill = make_prefill(get_experience, cfg.env_steps_per_actor_step)
        update_actor_and_alpha = make_update_actor(actor, sa_encoder, g_encoder, target_entropy, cfg)
        update_critic = make_update_critic(sa_encoder, g_encoder, cfg)
        sgd_step = make_sgd_step(update_actor_and_alpha, update_critic)
        training_step = make_training_step(sgd_step, get_experience, replay_buffer, cfg)
        training_epoch = make_training_epoch(training_step, replay_buffer, cfg)

        # Prefill
        env_state = jax.jit(env.reset)(jax.random.split(env_key, cfg.num_envs))
        env.step = jax.jit(env.step)
        eval_env.step = jax.jit(eval_env.step)

        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _ = prefill(
            training_state, env_state, buffer_state, prefill_key, cfg.num_prefill_actor_steps)

        # Evaluator
        evaluator = setup_evaluator(actor, sa_encoder, g_encoder, eval_env, env_info.obs_dim,
                                    eval_actor_key, cfg, eval_env_key)

        # Training loop
        print(f'Sweep training: {cfg.exp_name}...', flush=True)
        start_time = time.time()

        for ne in range(cfg.num_epochs):
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key)

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_time = time.time() - t
            sps = (cfg.env_steps_per_actor_step * cfg.num_training_steps_per_epoch) / epoch_time
            metrics = {
                "training/sps": sps,
                "training/walltime": time.time() - start_time,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            metrics = evaluator.run_evaluation(training_state, metrics)

            if ne < 5 or ne >= cfg.num_epochs - 5 or ne % ckpt_config.save_interval_epochs == 0:
                save_checkpoint(ckpt_manager, training_state, ne)

            wandb.log(metrics, step=ne)

        # Final checkpoint
        final_step = int(training_state.env_steps)
        if final_step not in ckpt_manager.all_steps():
            save_checkpoint(ckpt_manager, training_state, cfg.num_epochs - 1)
        ckpt_manager.wait_until_finished()

        run.finish()

    # Run the agent
    wandb.agent(
        sweep_id,
        function=train_fn,
        project=project,
        entity=entity,
        count=count,
    )


def main():
    parser = argparse.ArgumentParser(description="WandB Sweep runner for Scaling-CRL")
    parser.add_argument("--config", help="Path to sweep YAML config")
    parser.add_argument("--sweep-id", help="Existing sweep ID to run agent for")
    parser.add_argument("--entity", default="sungwayne99999-national-cheng-kung-university-co-op")
    parser.add_argument("--project", default="scaling-crl-v2")
    parser.add_argument("--count", type=int, default=None, help="Max number of sweep trials")
    parser.add_argument("--create-only", action="store_true", help="Only create sweep, don't run agent")
    args = parser.parse_args()

    if args.sweep_id:
        # Run agent for existing sweep
        print(f"Running agent for sweep: {args.sweep_id}")
        run_agent(args.sweep_id, args.entity, args.project, count=args.count)

    elif args.config:
        # Create sweep from config
        sweep_cfg = load_sweep_config(args.config)
        print(f"Creating sweep: {sweep_cfg.get('name', sweep_cfg.get('base_experiment', 'sweep'))}")
        print(f"  Method: {sweep_cfg.get('method', 'bayes')}")
        print(f"  Parameters: {list(sweep_cfg['parameters'].keys())}")

        sweep_id = create_sweep(sweep_cfg, args.entity, args.project)
        full_id = f"{args.entity}/{args.project}/{sweep_id}"
        print(f"Sweep created: {full_id}")

        if args.create_only:
            print(f"Run agents with: python infra/sweep.py --sweep-id {full_id}")
        else:
            print(f"Running agent (count={args.count or 'unlimited'})...")
            run_agent(full_id, args.entity, args.project, count=args.count)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
