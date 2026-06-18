"""CRL Training Script — thin orchestrator.

All logic lives in crl/ and utils/. This file only:
1. Parses CLI args
2. Creates env, networks, optimizer, buffer
3. Resumes from checkpoint if needed
4. Runs the training loop
5. Saves checkpoints and logs to wandb
"""

import os
import time
import pickle
import random
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import tyro
import wandb

from dataclasses import dataclass
from typing import NamedTuple, Any
from pathlib import Path

from brax import envs
from flax.training.train_state import TrainState

from crl.networks import Actor, SA_encoder, G_encoder
from crl.buffer import TrajectoryUniformSamplingQueue
from crl.algorithm import (
    make_actor_step, make_get_experience, make_update_actor,
    make_update_critic, make_sgd_step, make_training_step,
    make_training_epoch, setup_evaluator,
)
from utils.env_factory import make_env, wrap_env
from utils.checkpoint import (
    CheckpointConfig, create_checkpoint_manager, save_checkpoint,
    restore_checkpoint, load_legacy_checkpoint, find_legacy_checkpoint,
)
from utils.logging import setup_wandb


@dataclass
class Args:
    # === Identity ===
    exp_name: str = ""
    seed: int = 1000

    # === Wandb ===
    track: bool = True
    wandb_project_name: str = "scaling-crl-nano4"
    wandb_entity: str = "sungwayne99999-national-cheng-kung-university-co-op"
    wandb_mode: str = "online"
    wandb_dir: str = "."
    wandb_group: str = ""

    # === Checkpoint (all required — no defaults) ===
    checkpoint: bool = True
    checkpoint_save_interval_epochs: int = 10
    checkpoint_max_to_keep: int = 3
    checkpoint_keep_period: int = 50

    # === Environment ===
    env_id: str = "ant"
    eval_env_id: str = ""
    episode_length: int = 1000
    num_envs: int = 512
    num_eval_envs: int = 128

    # === Training budget ===
    total_env_steps: int = 100_000_000
    num_epochs: int = 100

    # === Network architecture ===
    actor_network_width: int = 256
    critic_network_width: int = 256
    actor_depth: int = 4
    critic_depth: int = 4
    actor_skip_connections: int = 4
    critic_skip_connections: int = 4
    use_relu: int = 0

    # === Optimizer ===
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # === CRL specific ===
    batch_size: int = 512
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    entropy_param: float = 0.5
    disable_entropy: int = 0
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    num_episodes_per_env: int = 1
    training_steps_multiplier: int = 1
    use_all_batches: int = 0
    num_sgd_batches_per_training_step: int = 800

    # === Eval/Render ===
    eval_actor: int = 0
    expl_actor: int = 1
    capture_vis: bool = False
    num_render: int = 10
    vis_length: int = 1000
    save_buffer: int = 0
    resume_from: str = ""


class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def main():
    args = tyro.cli(Args)

    # --- Compute derived values ---
    args.eval_env_id = args.eval_env_id or args.env_id
    env_steps_per_actor_step = args.num_envs * args.unroll_length
    num_prefill_env_steps = args.min_replay_size * args.num_envs
    num_prefill_actor_steps = int(np.ceil(args.min_replay_size / args.unroll_length))
    num_training_steps_per_epoch = (args.total_env_steps - num_prefill_env_steps) // (args.num_epochs * env_steps_per_actor_step)

    # --- Env ---
    env, env_info = make_env(args.env_id)
    env = wrap_env(env, episode_length=args.episode_length)
    obs_size = env.observation_size
    action_size = env.action_size

    eval_env, _ = make_env(args.eval_env_id)
    eval_env = wrap_env(eval_env, episode_length=args.episode_length)

    # Update args with runtime values
    args.obs_dim = env_info.obs_dim
    args.goal_start_idx = env_info.goal_start_idx
    args.goal_end_idx = env_info.goal_end_idx

    # --- Wandb ---
    trigger_sync = None
    if args.track:
        trigger_sync = setup_wandb(args)

    # --- Checkpoint setup ---
    save_path = None
    ckpt_manager = None
    ckpt_config = CheckpointConfig(
        save_interval_epochs=args.checkpoint_save_interval_epochs,
        max_to_keep=args.checkpoint_max_to_keep,
        keep_period=args.checkpoint_keep_period,
    )
    if args.checkpoint:
        save_path = Path(args.wandb_dir) / Path(f"runs/{args.exp_name}")
        os.makedirs(save_path, exist_ok=True)
        ckpt_manager = create_checkpoint_manager(save_path, ckpt_config)
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(args, f)

    # --- RNG ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    # --- Networks ---
    actor = Actor(action_size=action_size, network_width=args.actor_network_width,
                  network_depth=args.actor_depth, skip_connections=args.actor_skip_connections,
                  use_relu=args.use_relu)
    sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth,
                            skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth,
                          skip_connections=args.critic_skip_connections, use_relu=args.use_relu)

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr))
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, env_info.obs_dim]), np.ones([1, action_size]))
    g_encoder_params = g_encoder.init(g_key, np.ones([1, env_info.goal_end_idx - env_info.goal_start_idx]))
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=args.critic_lr))
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
        tx=optax.adam(learning_rate=args.alpha_lr))

    training_state = TrainingState(
        env_steps=jnp.zeros(()), gradient_steps=jnp.zeros(()),
        actor_state=actor_state, critic_state=critic_state, alpha_state=alpha_state)

    # --- Resume ---
    start_epoch = 0
    if args.resume_from and ckpt_manager is not None:
        ckpt_data, ckpt_step = restore_checkpoint(ckpt_manager)
        if ckpt_data is not None:
            print(f"Restored Orbax checkpoint at step {ckpt_step}", flush=True)
        else:
            legacy_path = find_legacy_checkpoint(str(save_path))
            if legacy_path:
                ckpt_data = load_legacy_checkpoint(legacy_path)
            if ckpt_data is None:
                print("No valid checkpoint found, starting from scratch", flush=True)
                args.resume_from = ""

        if ckpt_data is not None:
            if isinstance(ckpt_data, dict) and 'actor_params' in ckpt_data:
                training_state = training_state.replace(
                    actor_state=training_state.actor_state.replace(params=ckpt_data['actor_params']),
                    critic_state=training_state.critic_state.replace(params=ckpt_data['critic_params']),
                    alpha_state=training_state.alpha_state.replace(params=ckpt_data['alpha_params']))
                start_epoch = ckpt_data.get('epoch', 0) + 1
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

    # --- Replay Buffer ---
    dummy_transition = Transition(
        observation=jnp.zeros((obs_size,)), action=jnp.zeros((action_size,)),
        reward=0.0, discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}})
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=args.max_replay_size, dummy_data_sample=dummy_transition,
        sample_batch_size=args.batch_size, num_envs=args.num_envs,
        episode_length=args.episode_length)
    replay_buffer.insert_internal = jax.jit(replay_buffer.insert_internal)
    replay_buffer.sample_internal = jax.jit(replay_buffer.sample_internal)
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # --- Build training functions ---
    actor_step_fn = make_actor_step(actor, mode="stochastic")
    get_experience = make_get_experience(actor_step_fn, replay_buffer, args)
    update_actor_and_alpha = make_update_actor(actor, sa_encoder, g_encoder, args)
    update_critic = make_update_critic(sa_encoder, g_encoder, args)
    sgd_step = make_sgd_step(update_actor_and_alpha, update_critic)
    training_step = make_training_step(sgd_step, get_experience, replay_buffer, args)
    training_epoch = make_training_epoch(training_step, replay_buffer, args)

    # --- Prefill ---
    env_state = jax.jit(env.reset)(jax.random.split(env_key, args.num_envs))
    env.step = jax.jit(env.step)
    eval_env_state = jax.jit(eval_env.reset)(jax.random.split(eval_env_key, args.num_envs))
    eval_env.step = jax.jit(eval_env.step)

    key, prefill_key = jax.random.split(key)
    for _ in range(num_prefill_actor_steps):
        env_state, buffer_state = get_experience(training_state, env_state, buffer_state, prefill_key, env)
        training_state = training_state.replace(env_steps=training_state.env_steps + env_steps_per_actor_step)
        prefill_key, _ = jax.random.split(prefill_key)

    # --- Evaluator ---
    evaluator = setup_evaluator(actor, sa_encoder, g_encoder, eval_env, args, eval_env_key)

    # --- Training loop ---
    print('starting training....', flush=True)
    start_time = time.time()

    for ne in range(start_epoch, args.num_epochs):
        t = time.time()
        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key, env)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_time = time.time() - t
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_time
        metrics = {
            "training/sps": sps,
            "training/walltime": time.time() - start_time,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        metrics = evaluator.run_evaluation(training_state, metrics)
        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}", flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % ckpt_config.save_interval_epochs == 0:
                save_checkpoint(ckpt_manager, training_state, ne)

        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == 'offline' and trigger_sync:
                trigger_sync()

        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    # --- Final checkpoint ---
    if args.checkpoint:
        final_step = int(training_state.env_steps)
        if final_step not in ckpt_manager.all_steps():
            save_checkpoint(ckpt_manager, training_state, args.num_epochs - 1)
        ckpt_manager.wait_until_finished()

    # --- Render ---
    if args.capture_vis:
        render_env, _ = make_env(args.eval_env_id)
        @jax.jit
        def policy_step(env_state, actor_params):
            means, _ = actor.apply(actor_params, env_state.obs)
            return render_env.step(env_state, nn.tanh(means))

        rollout_states = []
        for i in range(args.num_render):
            rng = jax.random.PRNGKey(args.seed * 1000 + i)
            es = jax.jit(render_env.reset)(rng)
            for _ in range(args.episode_length):
                es = policy_step(es, training_state.actor_state.params)
                rollout_states.append(es.pipeline_state)
                if es.done:
                    break

        from brax.io import html
        html_string = html.render(render_env.sys, rollout_states)
        with open(f"{save_path}/vis.html", "w") as f:
            f.write(html_string)
        if args.track:
            wandb.log({"vis": wandb.Html(html_string)})


if __name__ == "__main__":
    main()
