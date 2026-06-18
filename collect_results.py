#!/usr/bin/env python3
"""Scan all runs/*/ and generate a summary table of experiment results.

Usage:
  python collect_results.py              # print table to stdout
  python collect_results.py --csv results.csv  # also save CSV
  python collect_results.py --json results.json
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"
TARGET_STEPS = 100_000_000


def parse_exp_name(name):
    """Parse exp_name like 'ant_d8_s1000' → (env, depth, seed)."""
    parts = name.rsplit("_s", 1)
    if len(parts) == 2:
        prefix, seed = parts
        seed = int(seed)
    else:
        prefix, seed = name, "?"

    if "_d" in prefix:
        env, depth_str = prefix.rsplit("_d", 1)
        try:
            depth = int(depth_str)
        except ValueError:
            depth = "?"
    else:
        env, depth = prefix, "?"

    return env, depth, seed


def get_max_step(exp_dir):
    """Get max env steps from step_*.pkl filenames."""
    ckpts = sorted(exp_dir.glob("step_*.pkl"),
                   key=lambda f: int(f.name.rsplit('_', 1)[-1].rsplit('.', 1)[0]))
    if ckpts:
        return int(ckpts[-1].name.rsplit('_', 1)[-1].rsplit('.', 1)[0])
    return 0


def scan_experiment(exp_dir):
    """Extract metrics from one experiment directory."""
    name = exp_dir.name
    env, depth, seed = parse_exp_name(name)

    has_final = (exp_dir / "final.pkl").exists()
    max_step = get_max_step(exp_dir)
    train_pct = max_step / TARGET_STEPS * 100 if max_step else 0
    # Flag experiments that crashed early (< 50% of target steps)
    if not has_final:
        status = "INCOMPLETE"
    elif train_pct < 50:
        status = "CRASHED"
    else:
        status = "DONE"

    # Check for eval_metrics.json (from eval.py)
    eval_path = exp_dir / "eval_metrics.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_metrics = json.load(f)
    else:
        eval_metrics = {}

    return {
        "exp_name": name,
        "env": env,
        "depth": depth,
        "seed": seed,
        "status": status,
        "max_step": max_step,
        "train_pct": round(train_pct, 1),
        # eval/episode_success = total steps within 0.5 of goal (NOT a rate)
        "success_steps": eval_metrics.get("eval/episode_success", ""),
        # eval/episode_success_any = fraction of envs that reached goal at least once
        "success_rate": eval_metrics.get("eval/episode_success_any", ""),
        "success_easy_steps": eval_metrics.get("eval/episode_success_easy", ""),
        "episode_reward": eval_metrics.get("eval/episode_reward", ""),
        # eval/episode_dist = sum of per-step distances (NOT final distance)
        "episode_total_dist": eval_metrics.get("eval/episode_dist", ""),
        "avg_episode_length": eval_metrics.get("eval/avg_episode_length", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("--csv", help="Save CSV to this path")
    parser.add_argument("--json", help="Save JSON to this path")
    args = parser.parse_args()

    results = []
    if not RUNS_DIR.exists():
        print("No runs/ directory found")
        return

    for d in sorted(RUNS_DIR.iterdir()):
        if d.name == "_old" or not d.is_dir():
            continue
        results.append(scan_experiment(d))

    if not results:
        print("No experiments found in runs/")
        return

    # Print table
    print(f"{'exp_name':<35} {'d':>3} {'s':>5} {'status':<10} {'step':>12} {'%':>5} {'succ%':>6} {'ep_len':>7} {'reward':>9}")
    print("-" * 100)
    for r in results:
        succ = f"{r['success_rate']*100:.1f}" if isinstance(r['success_rate'], (int, float)) else "?"
        ep_len = f"{r['avg_episode_length']:.0f}" if isinstance(r['avg_episode_length'], (int, float)) else "?"
        reward = f"{r['episode_reward']:.1f}" if isinstance(r['episode_reward'], (int, float)) else "?"
        print(f"{r['exp_name']:<35} {str(r['depth']):>3} {str(r['seed']):>5} {r['status']:<10} {r['max_step']:>12,} {r['train_pct']:>4.0f}% {succ:>5}% {ep_len:>7} {reward:>9}")

    done = sum(1 for r in results if r['status'] == 'DONE')
    crashed = sum(1 for r in results if r['status'] == 'CRASHED')
    incomplete = sum(1 for r in results if r['status'] == 'INCOMPLETE')
    print(f"\nTotal: {len(results)} ({done} done, {crashed} crashed, {incomplete} incomplete)")

    # Save CSV
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV saved to {args.csv}")

    # Save JSON
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON saved to {args.json}")


if __name__ == "__main__":
    main()
