#!/usr/bin/env python3
"""Standalone evaluation script — load checkpoint, run eval, save metrics.

Usage:
  python eval.py --exp_name ant_d8_s1000                    # eval final.pkl
  python eval.py --exp_name ant_d8_s1000 --num_eval_envs 256
  python eval.py --checkpoint runs/ant_d8_s1000/final.pkl --env_id ant --depth 8
  python eval.py --all                                      # eval all completed
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import flax.linen as nn
from brax import envs

from evaluator import CrlEvaluator
from crl.networks import Actor
from utils.env_factory import make_env, wrap_env
from utils.checkpoint import (
    load_legacy_checkpoint, create_checkpoint_manager, restore_checkpoint,
    find_legacy_checkpoint, CheckpointConfig,
)
from train import Args, TrainingState, Transition
from flax.training.train_state import TrainState as FlaxTrainState
import __main__
__main__.Args = Args



def run_eval(exp_name=None, checkpoint=None, env_id=None, depth=None, seed=None,
             num_eval_envs=128, episode_length=1000):
    """Run evaluation on a checkpoint, return metrics dict."""

    # Resolve exp_name → save_path
    if exp_name and not checkpoint:
        save_path = f"runs/{exp_name}"
    elif checkpoint:
        save_path = os.path.dirname(checkpoint)
    else:
        print("ERROR: Must specify --exp_name or --checkpoint")
        return None

    # Load args.pkl if available (supports both dict and dataclass formats)
    args_path = os.path.join(save_path, "args.pkl")
    if os.path.exists(args_path):
        with open(args_path, 'rb') as f:
            ckpt_args = pickle.load(f)
        # Support both dict and dataclass formats
        def _get(key, default=None):
            if isinstance(ckpt_args, dict):
                return ckpt_args.get(key, default)
            return getattr(ckpt_args, key, default)
        if env_id is None:
            env_id = _get("env_id")
        if depth is None:
            depth = _get("actor_depth")
        if seed is None:
            seed = _get("seed")
        actor_width = _get("actor_network_width", 256)
        use_relu = _get("use_relu", 0)
        print(f"Loaded args: env_id={env_id}, depth={depth}, seed={seed}")
    else:
        if env_id is None or depth is None:
            print("ERROR: Must specify --env_id and --depth when no args.pkl found")
            return None
        actor_width = 256
        use_relu = 0
        print(f"WARNING: No args.pkl, using defaults")

    # Load checkpoint: try Orbax first, then legacy pickle
    ckpt_data = None
    if os.path.isdir(os.path.join(save_path, "checkpoints")):
        # Use default config for eval — only need to read, not write
        eval_ckpt_config = CheckpointConfig(save_interval_epochs=10, max_to_keep=3, keep_period=50)
        manager = create_checkpoint_manager(save_path, eval_ckpt_config)
        ckpt_data, step = restore_checkpoint(manager)
        if ckpt_data is not None:
            print(f"Loaded Orbax checkpoint at step {step}")

    if ckpt_data is None:
        # Fall back to legacy pickle
        if checkpoint:
            ckpt_data = load_legacy_checkpoint(checkpoint)
        else:
            legacy_path = find_legacy_checkpoint(save_path)
            if legacy_path:
                ckpt_data = load_legacy_checkpoint(legacy_path)
                print(f"Loaded legacy checkpoint: {legacy_path}")

    if ckpt_data is None:
        print(f"ERROR: No valid checkpoint found in {save_path}")
        return None

    if isinstance(ckpt_data, dict) and 'actor_params' in ckpt_data:
        actor_params = ckpt_data['actor_params']
        print(f"Checkpoint: epoch={ckpt_data.get('epoch','?')}, env_steps={ckpt_data.get('env_steps','?')}")
    elif isinstance(ckpt_data, tuple):
        _, actor_params, _ = ckpt_data
        print("Checkpoint: OLD format (bare tuple)")
    else:
        print(f"ERROR: Unknown checkpoint format: {type(ckpt_data)}")
        return None

    # Create env
    env_raw, env_info = make_env(env_id)
    env = wrap_env(env_raw, episode_length=episode_length)
    action_size = env.action_size

    # Create actor
    actor = Actor(action_size=action_size, network_width=actor_width,
                  network_depth=depth, use_relu=use_relu)

    # Build minimal TrainingState — only actor_state.params is used by eval
    import optax
    fake_actor_state = FlaxTrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optax.sgd(0.0),  # dummy optimizer, never used
    )
    fake_ts = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=fake_actor_state,
        critic_state=None,
        alpha_state=None,
    )

    # Deterministic actor step
    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    # Eval
    key = jax.random.PRNGKey(seed + 99999)
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        key=key,
    )

    print(f"Evaluating: env_id={env_id}, depth={depth}, num_eval_envs={num_eval_envs}")
    print("Compiling eval (JIT)...", flush=True)
    t0 = time.time()

    metrics = evaluator.run_evaluation(fake_ts, {})

    # Clean up JAX arrays → float
    clean = {}
    for k, v in metrics.items():
        if hasattr(v, 'item'):
            clean[k] = float(v)
        elif isinstance(v, (int, float)):
            clean[k] = float(v)
        else:
            clean[k] = str(v)

    print(f"Eval done in {time.time() - t0:.1f}s")
    print(json.dumps(clean, indent=2))
    return clean


def eval_all(force=False):
    """Eval all experiments with checkpoints in runs/."""
    results = []
    skipped = 0
    for d in sorted(os.listdir("runs")):
        if d.startswith("_") or d.startswith("."):
            continue
        # Check for Orbax checkpoints dir or legacy final.pkl
        has_ckpt = (os.path.isdir(f"runs/{d}/checkpoints") or
                    os.path.exists(f"runs/{d}/final.pkl"))
        if not has_ckpt:
            continue
        if not force and os.path.exists(f"runs/{d}/eval_metrics.json"):
            skipped += 1
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {d}")
        print(f"{'='*60}")
        try:
            metrics = run_eval(exp_name=d)
            if metrics:
                metrics['exp_name'] = d
                results.append(metrics)
                with open(f"runs/{d}/eval_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"Saved to runs/{d}/eval_metrics.json")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({'exp_name': d, 'error': str(e)})

    summary_path = "runs/eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Summary saved to {summary_path}")
    print(f"Evaluated {len(results)} experiments, skipped {skipped} (already had eval)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints")
    parser.add_argument("--exp_name", help="Experiment name (looks for runs/{exp_name}/final.pkl)")
    parser.add_argument("--checkpoint", help="Direct path to checkpoint .pkl")
    parser.add_argument("--env_id", help="Environment ID (auto from args.pkl)")
    parser.add_argument("--depth", type=int, help="Network depth (auto from args.pkl)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--all", action="store_true", help="Eval all completed experiments")
    parser.add_argument("--force", action="store_true", help="Re-eval even if eval_metrics.json exists")
    args = parser.parse_args()

    if args.all:
        eval_all(force=args.force)
    elif args.exp_name or args.checkpoint:
        metrics = run_eval(
            exp_name=args.exp_name,
            checkpoint=args.checkpoint,
            env_id=args.env_id,
            depth=args.depth,
            seed=args.seed,
            num_eval_envs=args.num_eval_envs,
        )
        if metrics and args.exp_name:
            with open(f"runs/{args.exp_name}/eval_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved to runs/{args.exp_name}/eval_metrics.json")
    else:
        parser.print_help()
