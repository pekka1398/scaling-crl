#!/usr/bin/env python3
"""Render trained policy as an interactive HTML visualization.

Usage:
  python render.py --exp_name ant_d8_s1000                 # render final.pkl
  python render.py --exp_name ant_d8_s1000 --num_episodes 5
  python render.py --all                                    # render all with final.pkl
"""

import os
import sys
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from brax import envs
from brax.io import html

from train import Actor, load_params, Args
import __main__
__main__.Args = Args


def make_env_raw(env_id):
    """Create an unwrapped env (for rendering sys + single-env reset)."""
    if env_id == "ant":
        from envs.ant import Ant
        return Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
    elif "ant" in env_id and "maze" in env_id:
        from envs.ant_maze import AntMaze
        return AntMaze(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True, maze_layout_name=env_id[4:])
    elif env_id == "ant_ball":
        from envs.ant_ball import AntBall
        return AntBall(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
    elif env_id == "ant_push":
        from envs.ant_push import AntPush
        return AntPush(backend="mjx")
    elif env_id == "humanoid":
        from envs.humanoid import Humanoid
        return Humanoid(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
    elif "humanoid" in env_id and "maze" in env_id:
        from envs.humanoid_maze import HumanoidMaze
        return HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])
    elif env_id == "arm_reach":
        from envs.manipulation.arm_reach import ArmReach
        return ArmReach(backend="mjx")
    elif env_id == "arm_grasp":
        from envs.manipulation.arm_grasp import ArmGrasp
        return ArmGrasp(cube_noise_scale=0.3, backend="mjx")
    elif env_id == "arm_push_easy":
        from envs.manipulation.arm_push_easy import ArmPushEasy
        return ArmPushEasy(backend="mjx")
    elif env_id == "arm_push_hard":
        from envs.manipulation.arm_push_hard import ArmPushHard
        return ArmPushHard(backend="mjx")
    elif env_id == "arm_binpick_easy":
        from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
        return ArmBinpickEasy(backend="mjx")
    elif env_id == "arm_binpick_hard":
        from envs.manipulation.arm_binpick_hard import ArmBinpickHard
        return ArmBinpickHard(backend="mjx")
    else:
        raise NotImplementedError(f"Unknown env_id: {env_id}")


def load_actor(exp_name, checkpoint=None):
    """Load checkpoint and return (actor, actor_params, env_id, seed)."""
    if not checkpoint:
        save_path = f"runs/{exp_name}"
        checkpoint = os.path.join(save_path, "final.pkl")
        if not os.path.exists(checkpoint):
            import glob
            ckpts = sorted(glob.glob(f"{save_path}/step_*.pkl"),
                          key=lambda f: int(f.rsplit('_', 1)[-1].rsplit('.', 1)[0]))
            if not ckpts:
                print(f"ERROR: No checkpoint found in {save_path}")
                return None
            checkpoint = ckpts[-1]
        print(f"Using checkpoint: {checkpoint}")

    args_path = os.path.join(os.path.dirname(checkpoint), "args.pkl")
    if os.path.exists(args_path):
        with open(args_path, 'rb') as f:
            ckpt_args = pickle.load(f)
        env_id = ckpt_args.env_id
        depth = ckpt_args.actor_depth
        seed = ckpt_args.seed
        actor_skip = ckpt_args.actor_skip_connections
        actor_width = ckpt_args.actor_network_width
        use_relu = ckpt_args.use_relu
    else:
        raise RuntimeError("args.pkl required to infer model architecture")

    ckpt_data = load_params(checkpoint)
    if ckpt_data is None:
        raise RuntimeError(f"Corrupt checkpoint: {checkpoint}")

    if isinstance(ckpt_data, dict) and 'actor_params' in ckpt_data:
        actor_params = ckpt_data['actor_params']
    elif isinstance(ckpt_data, tuple):
        _, actor_params, _ = ckpt_data
    else:
        raise RuntimeError(f"Unknown checkpoint format: {type(ckpt_data)}")

    # Create env to get action_size
    raw_env = make_env_raw(env_id)
    action_size = raw_env.action_size

    actor = Actor(action_size=action_size, network_width=actor_width,
                  network_depth=depth, skip_connections=actor_skip, use_relu=use_relu)

    return actor, actor_params, env_id, seed, raw_env


def render_exp(exp_name, num_episodes=10, episode_length=1000, force=False):
    """Render policy rollout and save vis.html."""
    save_path = f"runs/{exp_name}"
    vis_path = os.path.join(save_path, "vis.html")
    if os.path.exists(vis_path) and not force:
        print(f"  vis.html already exists, skipping (use --force to overwrite)")
        return False

    result = load_actor(exp_name)
    if result is None:
        return False
    actor, actor_params, env_id, seed, raw_env = result

    @jax.jit
    def policy_step(env_state):
        means, _ = actor.apply(actor_params, env_state.obs)
        actions = nn.tanh(means)
        return raw_env.step(env_state, actions)

    rollout_states = []
    for i in range(num_episodes):
        rng = jax.random.PRNGKey(seed * 1000 + i)
        env_state = jax.jit(raw_env.reset)(rng)

        for step in range(episode_length):
            env_state = policy_step(env_state)
            rollout_states.append(env_state.pipeline_state)
            if env_state.done:
                break

        print(f"  Episode {i}: {step+1} steps, done={bool(env_state.done)}")

    html_string = html.render(raw_env.sys, rollout_states)
    with open(vis_path, "w") as f:
        f.write(html_string)
    print(f"  Saved {vis_path} ({len(rollout_states)} frames, {num_episodes} episodes)")
    return True


def render_all(num_episodes=10, force=False):
    """Render all experiments with final.pkl."""
    ok = 0
    skipped = 0
    failed = 0
    for d in sorted(os.listdir("runs")):
        if d == "_old" or not os.path.isdir(f"runs/{d}"):
            continue
        if not os.path.exists(f"runs/{d}/final.pkl"):
            continue
        print(f"\n{'='*60}")
        print(f"Rendering: {d}")
        print(f"{'='*60}")
        try:
            if render_exp(d, num_episodes=num_episodes, force=force):
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Rendered: {ok}, skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render trained policy as HTML")
    parser.add_argument("--exp_name", help="Experiment name")
    parser.add_argument("--all", action="store_true", help="Render all with final.pkl")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to render")
    parser.add_argument("--episode_length", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--force", action="store_true", help="Overwrite existing vis.html")
    args = parser.parse_args()

    if args.all:
        render_all(num_episodes=args.num_episodes, force=args.force)
    elif args.exp_name:
        render_exp(args.exp_name, num_episodes=args.num_episodes,
                   episode_length=args.episode_length, force=args.force)
    else:
        parser.print_help()
