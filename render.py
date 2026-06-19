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

from crl.networks import Actor
from utils.env_factory import make_env, wrap_env
from utils.checkpoint import (
    load_legacy_checkpoint, create_checkpoint_manager, restore_checkpoint,
    find_legacy_checkpoint, CheckpointConfig,
)
from train import Args, AttrDict
import __main__
__main__.Args = Args
__main__.AttrDict = AttrDict


def load_actor(exp_name, checkpoint=None):
    """Load checkpoint and return (actor, actor_params, env_id, seed)."""
    save_path = f"runs/{exp_name}" if exp_name else os.path.dirname(checkpoint or "")

    args_path = os.path.join(save_path, "args.pkl")
    if os.path.exists(args_path):
        with open(args_path, 'rb') as f:
            ckpt_args = pickle.load(f)
        # Support both dict and dataclass formats
        def _get(key, default=None):
            if isinstance(ckpt_args, dict):
                return ckpt_args.get(key, default)
            return getattr(ckpt_args, key, default)
        env_id = _get("env_id")
        depth = _get("actor_depth")
        seed = _get("seed")
        actor_width = _get("actor_network_width", 256)
        use_relu = _get("use_relu", 0)
    else:
        raise RuntimeError("args.pkl required to infer model architecture")

    # Load checkpoint: try Orbax first, then legacy pickle
    ckpt_data = None
    if os.path.isdir(os.path.join(save_path, "checkpoints")):
        render_ckpt_config = CheckpointConfig(save_interval_epochs=10, max_to_keep=3, keep_period=50)
        manager = create_checkpoint_manager(save_path, render_ckpt_config)
        ckpt_data, step = restore_checkpoint(manager)
        if ckpt_data is not None:
            print(f"Loaded Orbax checkpoint at step {step}")

    if ckpt_data is None:
        if checkpoint:
            ckpt_data = load_legacy_checkpoint(checkpoint)
        else:
            legacy_path = find_legacy_checkpoint(save_path)
            if legacy_path:
                ckpt_data = load_legacy_checkpoint(legacy_path)
                print(f"Loaded legacy checkpoint: {legacy_path}")

    if ckpt_data is None:
        raise RuntimeError(f"No valid checkpoint found in {save_path}")

    if isinstance(ckpt_data, dict) and 'actor_params' in ckpt_data:
        actor_params = ckpt_data['actor_params']
    elif isinstance(ckpt_data, tuple):
        _, actor_params, _ = ckpt_data
    else:
        raise RuntimeError(f"Unknown checkpoint format: {type(ckpt_data)}")

    # Create env to get action_size
    raw_env, _ = make_env(env_id)
    action_size = raw_env.action_size

    actor = Actor(action_size=action_size, network_width=actor_width,
                  network_depth=depth, use_relu=use_relu)

    return actor, actor_params, env_id, seed, raw_env


def render_exp(exp_name, num_episodes=10, episode_length=1000, force=False, mp4=False):
    """Render policy rollout and save vis.html (and optionally vis.mp4)."""
    save_path = f"runs/{exp_name}"
    vis_path = os.path.join(save_path, "vis.html")
    mp4_path = os.path.join(save_path, "vis.mp4")
    if os.path.exists(vis_path) and not force and not mp4:
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

    if mp4:
        try:
            import imageio
            from brax.io import image as brax_image
            print(f"  Rendering MP4 frames...")
            imgs = brax_image.render(raw_env.sys, rollout_states, width=480, height=360)
            imageio.mimsave(mp4_path, imgs, fps=30)
            print(f"  Saved {mp4_path} ({len(imgs)} frames)")
        except Exception as e:
            print(f"  MP4 render failed: {e}")
            print(f"  Try: export MUJOCO_GL=egl")

    return True


def render_all(num_episodes=10, force=False, mp4=False):
    """Render all experiments with checkpoints."""
    ok = 0
    skipped = 0
    failed = 0
    for d in sorted(os.listdir("runs")):
        if d.startswith("_") or d.startswith(".") or not os.path.isdir(f"runs/{d}"):
            continue
        has_ckpt = (os.path.isdir(f"runs/{d}/checkpoints") or
                    os.path.exists(f"runs/{d}/final.pkl"))
        if not has_ckpt:
            continue
        print(f"\n{'='*60}")
        print(f"Rendering: {d}")
        print(f"{'='*60}")
        try:
            if render_exp(d, num_episodes=num_episodes, force=force, mp4=mp4):
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
    parser.add_argument("--mp4", action="store_true", help="Also render vis.mp4")
    args = parser.parse_args()

    if args.all:
        render_all(num_episodes=args.num_episodes, force=args.force, mp4=args.mp4)
    elif args.exp_name:
        render_exp(args.exp_name, num_episodes=args.num_episodes,
                   episode_length=args.episode_length, force=args.force, mp4=args.mp4)
    else:
        parser.print_help()
