"""CRL Training Script.

All experiment config comes from YAML. No fallbacks, no defaults in code.

Usage:
  python train.py --yaml all_experiments.yaml --exp_name ant_d8_s1000
  python train.py --yaml all_experiments.yaml --exp_name ant_d8_s1000 --compile_check
"""

import os
import time
import pickle
import random
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax
import flax.linen as nn
import tyro
import wandb

from dataclasses import dataclass
from pathlib import Path

from brax import envs


class AttrDict(dict):
    """Dict that supports attribute access (cfg.key = cfg["key"])."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config has no key '{key}'")
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"config has no key '{key}'")
from flax.training.train_state import TrainState

from crl.networks import Actor, SA_encoder, G_encoder
from crl.types import Transition
from crl.buffer import TrajectoryUniformSamplingQueue
from crl.algorithm import (
    make_actor_step, make_get_experience, make_update_actor,
    make_update_critic, make_sgd_step, make_training_step,
    make_training_epoch, setup_evaluator, make_prefill,
)
from utils.env_factory import make_env, wrap_env
from utils.checkpoint import (
    CheckpointConfig, create_checkpoint_manager, save_checkpoint,
    restore_checkpoint, load_legacy_checkpoint, find_legacy_checkpoint,
)
from utils.logging import setup_wandb


# All fields that must exist in YAML. Missing any = error.
_REQUIRED_FIELDS = [
    "exp_name", "env_id", "seed", "depth",
    "num_epochs", "total_env_steps",
    "actor_network_width", "critic_network_width",
    "batch_size", "num_envs",
    "actor_lr", "critic_lr", "alpha_lr",
    "unroll_length", "max_replay_size", "min_replay_size",
    "entropy_param", "gamma", "logsumexp_penalty_coeff",
    "disable_entropy", "use_relu",
    "num_episodes_per_env", "training_steps_multiplier",
    "use_all_batches", "num_sgd_batches_per_training_step",
    "eval_actor", "expl_actor", "num_eval_envs", "episode_length",
    "resume_from",
    "checkpoint", "checkpoint_save_interval_epochs",
    "checkpoint_max_to_keep", "checkpoint_keep_period",
    "save_buffer", "capture_vis", "num_render",
    "track", "wandb_mode", "wandb_group", "wandb_project_name", "wandb_entity", "wandb_dir",
]


@dataclass
class Args:
    yaml: str               # Path to experiment YAML
    exp_name: str           # Which experiment to run
    compile_check: bool = False  # If True: 1 epoch / 1M steps, no checkpoint/track


@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


def load_exp_config(args):
    """Load experiment config from YAML. Returns dict. Raises on missing fields."""
    with open(args.yaml) as f:
        all_exps = yaml.safe_load(f)

    exp = None
    for e in all_exps:
        if e.get("exp_name") == args.exp_name:
            exp = e
            break
    if exp is None:
        names = [e.get("exp_name", "?") for e in all_exps]
        raise ValueError(f"exp_name '{args.exp_name}' not found in {args.yaml}. Available: {names}")

    # Validate required fields
    missing = [f for f in _REQUIRED_FIELDS if f not in exp]
    if missing:
        raise ValueError(f"YAML entry '{args.exp_name}' missing required fields: {missing}")

    # depth → actor_depth / critic_depth
    exp["actor_depth"] = exp.pop("depth")
    exp["critic_depth"] = exp["actor_depth"]

    # Compile check overrides
    if args.compile_check:
        exp["exp_name"] = "compile_" + exp["exp_name"]
        exp["num_epochs"] = 1
        exp["total_env_steps"] = 1_000_000
        exp["checkpoint"] = False
        exp["track"] = False
        exp["wandb_mode"] = "offline"

    return AttrDict(exp)


def main():
    cli_args = tyro.cli(Args)
    cfg = load_exp_config(cli_args)

    # --- Compute derived values ---
    cfg["eval_env_id"] = cfg.get("eval_env_id", cfg["env_id"])
    cfg["env_steps_per_actor_step"] = cfg["num_envs"] * cfg["unroll_length"]
    cfg["num_prefill_env_steps"] = cfg["min_replay_size"] * cfg["num_envs"]
    cfg["num_prefill_actor_steps"] = int(np.ceil(cfg["min_replay_size"] / cfg["unroll_length"]))
    cfg["num_training_steps_per_epoch"] = (cfg["total_env_steps"] - cfg["num_prefill_env_steps"]) // (cfg["num_epochs"] * cfg["env_steps_per_actor_step"])

    # --- Env ---
    env, env_info = make_env(cfg["env_id"])
    env = wrap_env(env, episode_length=cfg["episode_length"])
    obs_size = env.observation_size
    action_size = env.action_size

    eval_env, _ = make_env(cfg["eval_env_id"])
    eval_env = wrap_env(eval_env, episode_length=cfg["episode_length"])

    cfg["obs_dim"] = env_info.obs_dim
    cfg["goal_start_idx"] = env_info.goal_start_idx
    cfg["goal_end_idx"] = env_info.goal_end_idx

    # --- Checkpoint/Wandb paths ---
    save_path = None
    if cfg["checkpoint"]:
        save_path = Path(cfg.get("wandb_dir", ".")) / Path(f"runs/{cfg['exp_name']}")
        os.makedirs(save_path, exist_ok=True)

    # --- Checkpoint setup ---
    ckpt_manager = None
    ckpt_config = CheckpointConfig(
        save_interval_epochs=cfg["checkpoint_save_interval_epochs"],
        max_to_keep=cfg["checkpoint_max_to_keep"],
        keep_period=cfg["checkpoint_keep_period"],
    )
    if cfg["checkpoint"]:
        ckpt_manager = create_checkpoint_manager(save_path, ckpt_config)
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(cfg, f)

    # --- RNG ---
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key, eval_actor_key = jax.random.split(key, 8)

    # --- Networks ---
    actor = Actor(action_size=action_size, network_width=cfg["actor_network_width"],
                  network_depth=cfg["actor_depth"], use_relu=cfg["use_relu"])
    sa_encoder = SA_encoder(network_width=cfg["critic_network_width"], network_depth=cfg["critic_depth"],
                            use_relu=cfg["use_relu"])
    g_encoder = G_encoder(network_width=cfg["critic_network_width"], network_depth=cfg["critic_depth"],
                          use_relu=cfg["use_relu"])

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=cfg["actor_lr"]))
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, env_info.obs_dim]), np.ones([1, action_size]))
    g_encoder_params = g_encoder.init(g_key, np.ones([1, env_info.goal_end_idx - env_info.goal_start_idx]))
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=cfg["critic_lr"]))
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
        tx=optax.adam(learning_rate=cfg["alpha_lr"]))

    training_state = TrainingState(
        env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()),
        actor_state=actor_state, critic_state=critic_state, alpha_state=alpha_state)

    # --- Resume ---
    start_epoch = 0
    wandb_resume_id = None
    resume_from = cfg["resume_from"]
    if resume_from and not cfg["checkpoint"]:
        print("WARNING: resume_from set but checkpoint=False, ignoring", flush=True)
        resume_from = ""

    if resume_from and ckpt_manager is not None:
        args_path = os.path.join(str(save_path), "args.pkl")
        if os.path.exists(args_path):
            try:
                with open(args_path, 'rb') as _f:
                    ckpt_args = pickle.load(_f)
                for ck_key in ['env_id', 'actor_depth', 'critic_depth', 'seed']:
                    if ck_key in cfg and ck_key in ckpt_args:
                        if cfg[ck_key] != ckpt_args[ck_key]:
                            raise ValueError(f"Checkpoint config mismatch: {ck_key}={ckpt_args[ck_key]} vs current={cfg[ck_key]}")
                print(f"Config validated OK (env={cfg['env_id']}, d={cfg['actor_depth']}, seed={seed})", flush=True)
            except ValueError:
                raise
            except Exception as e:
                print(f"WARNING: Could not validate checkpoint config: {e}", flush=True)

        ckpt_data, ckpt_step = restore_checkpoint(ckpt_manager)
        if ckpt_data is not None:
            print(f"Restored Orbax checkpoint at step {ckpt_step}", flush=True)
        else:
            legacy_path = find_legacy_checkpoint(str(save_path))
            if legacy_path:
                ckpt_data = load_legacy_checkpoint(legacy_path)
            if ckpt_data is None:
                print("No valid checkpoint found, starting from scratch", flush=True)
                resume_from = ""

        if ckpt_data is not None:
            if isinstance(ckpt_data, dict) and 'actor_params' in ckpt_data:
                training_state = training_state.replace(
                    actor_state=training_state.actor_state.replace(params=ckpt_data['actor_params']),
                    critic_state=training_state.critic_state.replace(params=ckpt_data['critic_params']),
                    alpha_state=training_state.alpha_state.replace(params=ckpt_data['alpha_params']))
                start_epoch = ckpt_data.get('epoch', 0) + 1
                # Save wandb resume ID
                wandb_id_path = os.path.join(str(save_path), "wandb_id.txt")
                if os.path.exists(wandb_id_path):
                    with open(wandb_id_path) as f:
                        wandb_resume_id = f.read().strip()
                ckpt_env_steps = ckpt_data.get('env_steps')
                ckpt_gradient_steps = ckpt_data.get('gradient_steps')
                if ckpt_env_steps is not None:
                    training_state = training_state.replace(
                        env_steps=jnp.array(ckpt_env_steps),
                        gradient_steps=jnp.array(ckpt_gradient_steps))
            else:
                alpha_params, actor_params, critic_params = ckpt_data
                training_state = training_state.replace(
                    actor_state=training_state.actor_state.replace(params=actor_params),
                    critic_state=training_state.critic_state.replace(params=critic_params),
                    alpha_state=training_state.alpha_state.replace(params=alpha_params))

    # --- Wandb (after resume so start_epoch is known) ---
    trigger_sync = None
    if cfg["track"]:
        trigger_sync, run = setup_wandb(cfg, resume_id=wandb_resume_id)
        # Save wandb run ID for future resume
        if run and save_path:
            wandb_id_path = os.path.join(str(save_path), "wandb_id.txt")
            with open(wandb_id_path, 'w') as f:
                f.write(run.id)

    # --- Replay Buffer ---
    dummy_transition = Transition(
        observation=jnp.zeros((obs_size,)), action=jnp.zeros((action_size,)),
        reward=0.0, discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}})
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=cfg["max_replay_size"], dummy_data_sample=dummy_transition,
        sample_batch_size=cfg["batch_size"], num_envs=cfg["num_envs"],
        episode_length=cfg["episode_length"])
    replay_buffer.insert_internal = jax.jit(replay_buffer.insert_internal)
    replay_buffer.sample_internal = jax.jit(replay_buffer.sample_internal)
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # --- Build training functions ---
    if cfg["expl_actor"] == 0:
        actor_step_fn = make_actor_step(actor, env, mode="deterministic")
    elif cfg["expl_actor"] == 1:
        actor_step_fn = make_actor_step(actor, env, mode="stochastic")
    else:
        actor_step_fn = make_actor_step(actor, env, mode="multi_sample",
            obs_dim=env_info.obs_dim, goal_start_idx=env_info.goal_start_idx,
            goal_end_idx=env_info.goal_end_idx, sa_encoder=sa_encoder,
            g_encoder=g_encoder, K=cfg["expl_actor"])

    target_entropy = -cfg["entropy_param"] * action_size
    get_experience = make_get_experience(actor_step_fn, replay_buffer, cfg["unroll_length"])
    prefill = make_prefill(get_experience, cfg["env_steps_per_actor_step"])
    update_actor_and_alpha = make_update_actor(actor, sa_encoder, g_encoder, target_entropy, cfg)
    update_critic = make_update_critic(sa_encoder, g_encoder, cfg)
    sgd_step = make_sgd_step(update_actor_and_alpha, update_critic)
    training_step = make_training_step(sgd_step, get_experience, replay_buffer, cfg)
    training_epoch = make_training_epoch(training_step, replay_buffer, cfg)

    # --- Prefill ---
    env_state = jax.jit(env.reset)(jax.random.split(env_key, cfg["num_envs"]))
    env.step = jax.jit(env.step)
    eval_env.step = jax.jit(eval_env.step)

    key, prefill_key = jax.random.split(key)
    training_state, env_state, buffer_state, _ = prefill(
        training_state, env_state, buffer_state, prefill_key, cfg["num_prefill_actor_steps"])

    # --- Evaluator ---
    evaluator = setup_evaluator(actor, sa_encoder, g_encoder, eval_env, env_info.obs_dim,
                                eval_actor_key, cfg, eval_env_key)

    # --- Training loop ---
    print('starting training....', flush=True)
    start_time = time.time()

    for ne in range(start_epoch, cfg["num_epochs"]):
        t = time.time()
        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_time = time.time() - t
        sps = (cfg["env_steps_per_actor_step"] * cfg["num_training_steps_per_epoch"]) / epoch_time
        metrics = {
            "training/sps": sps,
            "training/walltime": time.time() - start_time,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        metrics = evaluator.run_evaluation(training_state, metrics)
        print(f"epoch {ne} out of {cfg['num_epochs']} complete. metrics: {metrics}", flush=True)

        if ckpt_manager:
            if ne < 5 or ne >= cfg["num_epochs"] - 5 or ne % ckpt_config.save_interval_epochs == 0:
                save_checkpoint(ckpt_manager, training_state, ne)

        if cfg["track"]:
            wandb.log(metrics, step=ne)
            if cfg["wandb_mode"] == 'offline' and trigger_sync:
                trigger_sync()

        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    # --- Final checkpoint ---
    if ckpt_manager:
        final_step = int(training_state.env_steps)
        if final_step not in ckpt_manager.all_steps():
            save_checkpoint(ckpt_manager, training_state, cfg["num_epochs"] - 1)
        ckpt_manager.wait_until_finished()

    # --- Save replay buffer ---
    if ckpt_manager and cfg["save_buffer"]:
        print("Saving final buffer_state and buffer data...", flush=True)
        try:
            buffer_path = f"{save_path}/final_buffer.pkl"
            buffer_data = {
                'buffer_state': buffer_state,
                'max_replay_size': cfg["max_replay_size"],
                'batch_size': cfg["batch_size"],
                'num_envs': cfg["num_envs"],
                'episode_length': cfg["episode_length"],
            }
            with open(buffer_path, 'wb') as f:
                pickle.dump(buffer_data, f)
            print(f"Saved replay_buffer to {buffer_path}", flush=True)
        except Exception as e:
            print(f"Error saving final replay buffer: {e}", flush=True)

    # --- Render ---
    if cfg["capture_vis"]:
        render_env, _ = make_env(cfg["eval_env_id"])
        @jax.jit
        def policy_step(env_state, actor_params):
            means, _ = actor.apply(actor_params, env_state.obs)
            return render_env.step(env_state, nn.tanh(means))

        rollout_states = []
        for i in range(cfg["num_render"]):
            rng = jax.random.PRNGKey(seed * 1000 + i)
            es = jax.jit(render_env.reset)(rng)
            for _ in range(cfg["episode_length"]):
                es = policy_step(es, training_state.actor_state.params)
                rollout_states.append(es.pipeline_state)
                if es.done:
                    break

        from brax.io import html
        html_string = html.render(render_env.sys, rollout_states)
        with open(f"{save_path}/vis.html", "w") as f:
            f.write(html_string)
        if cfg["track"]:
            wandb.log({"vis": wandb.Html(html_string)})


if __name__ == "__main__":
    main()
