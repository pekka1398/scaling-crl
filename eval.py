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
from train import Actor, Transition, load_params, Args, TrainingState
from flax.training.train_state import TrainState as FlaxTrainState
# Make Args available as __main__.Args so pickle can find it
import __main__
__main__.Args = Args


# ─── Environment creation (mirrors train.py make_env) ───
def make_env(env_id, episode_length=1000):
    """Create environment by env_id. Returns (env, action_size)."""
    if env_id == "ant":
        from envs.ant import Ant
        env = Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
    elif "ant" in env_id and "maze" in env_id:
        from envs.ant_maze import AntMaze
        env = AntMaze(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True, maze_layout_name=env_id[4:])
    elif env_id == "ant_ball":
        from envs.ant_ball import AntBall
        env = AntBall(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
    elif env_id == "ant_push":
        from envs.ant_push import AntPush
        env = AntPush(backend="mjx")
    elif env_id == "humanoid":
        from envs.humanoid import Humanoid
        env = Humanoid(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
    elif "humanoid" in env_id and "maze" in env_id:
        from envs.humanoid_maze import HumanoidMaze
        env = HumanoidMaze(backend="spring", maze_layout_name=env_id[9:])
    elif env_id == "arm_reach":
        from envs.manipulation.arm_reach import ArmReach
        env = ArmReach(backend="mjx")
    elif env_id == "arm_grasp":
        from envs.manipulation.arm_grasp import ArmGrasp
        env = ArmGrasp(cube_noise_scale=0.3, backend="mjx")
    elif env_id == "arm_push_easy":
        from envs.manipulation.arm_push_easy import ArmPushEasy
        env = ArmPushEasy(backend="mjx")
    elif env_id == "arm_push_hard":
        from envs.manipulation.arm_push_hard import ArmPushHard
        env = ArmPushHard(backend="mjx")
    elif env_id == "arm_binpick_easy":
        from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
        env = ArmBinpickEasy(backend="mjx")
    elif env_id == "arm_binpick_hard":
        from envs.manipulation.arm_binpick_hard import ArmBinpickHard
        env = ArmBinpickHard(backend="mjx")
    else:
        raise NotImplementedError(f"Unknown env_id: {env_id}")

    env = envs.training.wrap(env, episode_length=episode_length)
    action_size = env.action_size
    return env, action_size


def run_eval(exp_name=None, checkpoint=None, env_id=None, depth=None, seed=None,
             num_eval_envs=128, episode_length=1000):
    """Run evaluation on a checkpoint, return metrics dict."""
    
    # Resolve exp_name → checkpoint path
    if exp_name and not checkpoint:
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

    # Load args.pkl if available
    args_path = os.path.join(os.path.dirname(checkpoint), "args.pkl")
    if os.path.exists(args_path):
        with open(args_path, 'rb') as f:
            ckpt_args = pickle.load(f)
        if env_id is None:
            env_id = ckpt_args.env_id
        if depth is None:
            depth = ckpt_args.actor_depth
        if seed is None:
            seed = ckpt_args.seed
        actor_skip = ckpt_args.actor_skip_connections
        actor_width = ckpt_args.actor_network_width
        use_relu = ckpt_args.use_relu
        print(f"Loaded args: env_id={env_id}, depth={depth}, seed={seed}")
    else:
        if env_id is None or depth is None:
            print("ERROR: Must specify --env_id and --depth when no args.pkl found")
            return None
        actor_skip = 4
        actor_width = 256
        use_relu = 0
        print(f"WARNING: No args.pkl, using defaults")

    # Load checkpoint
    ckpt_data = load_params(checkpoint)
    if ckpt_data is None:
        print(f"ERROR: Corrupt checkpoint: {checkpoint}")
        return None

    if isinstance(ckpt_data, dict) and 'actor_params' in ckpt_data:
        actor_params = ckpt_data['actor_params']
        ckpt_epoch = ckpt_data.get('epoch', '?')
        ckpt_steps = ckpt_data.get('env_steps', '?')
        print(f"Checkpoint format: NEW (epoch={ckpt_epoch}, env_steps={ckpt_steps})")
    elif isinstance(ckpt_data, tuple):
        _, actor_params, _ = ckpt_data
        print("Checkpoint format: OLD (bare tuple)")
    else:
        print(f"ERROR: Unknown checkpoint format: {type(ckpt_data)}")
        return None

    # Create env
    env, action_size = make_env(env_id, episode_length)

    # Create actor
    actor = Actor(action_size=action_size, network_width=actor_width,
                  network_depth=depth, skip_connections=actor_skip, use_relu=use_relu)

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
    """Eval all experiments with final.pkl in runs/."""
    results = []
    skipped = 0
    for d in sorted(os.listdir("runs")):
        if d == "_old":
            continue
        if not os.path.exists(f"runs/{d}/final.pkl"):
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
