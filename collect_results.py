#!/usr/bin/env python3
"""Aggregate experiment results from WandB.

Replaces the old filesystem-scanning collect_results.py.
All metrics are already logged to WandB during training.

Usage:
  python collect_results.py                          # print table from WandB
  python collect_results.py --csv results.csv        # also save CSV
  python collect_results.py --json results.json      # also save JSON
  python collect_results.py --env ant                # filter by env
  python collect_results.py --state crashed          # filter: finished/crashed/running
  python collect_results.py --tag paper_v1           # filter by wandb tag
  python collect_results.py --sort success_rate      # sort column
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import wandb
from utils.wandb_defaults import DEFAULT_ENTITY, DEFAULT_PROJECT

# Metrics to pull from each run's summary
METRIC_KEYS = [
    "eval/episode_success",
    "eval/episode_success_any",
    "eval/episode_success_easy",
    "eval/episode_reward",
    "eval/episode_dist",
    "eval/avg_episode_length",
    "training/envsteps",
    "training/walltime",
]


def fetch_runs(entity, project, filters=None):
    """Fetch all runs from WandB project."""
    api = wandb.Api()
    path = f"{entity}/{project}"
    runs = api.runs(path, filters=filters or {})
    return list(runs)


def run_to_row(run):
    """Convert a WandB run to a result row dict."""
    cfg = run.config or {}
    summary = run.summary or {}

    envsteps = summary.get("training/envsteps", 0)
    target = cfg.get("total_env_steps", 100_000_000)
    train_pct = (envsteps / target * 100) if target and envsteps else 0

    # Determine status from run state + training progress
    state = run.state  # "finished", "crashed", "running", "failed"
    if state == "finished" and train_pct >= 50:
        status = "DONE"
    elif state in ("crashed", "failed"):
        status = "CRASHED"
    elif state == "running":
        status = "RUNNING"
    else:
        status = "INCOMPLETE"

    row = {
        "exp_name": run.name or cfg.get("exp_name", "?"),
        "env": cfg.get("env_id", "?"),
        "depth": cfg.get("depth", "?"),
        "seed": cfg.get("seed", "?"),
        "status": status,
        "state": state,
        "envsteps": int(envsteps) if envsteps else 0,
        "train_pct": round(train_pct, 1),
        "wandb_url": run.url,
    }

    # Pull eval metrics from summary
    for key in METRIC_KEYS:
        short = key.replace("eval/", "").replace("training/", "train_")
        val = summary.get(key)
        if val is not None and not isinstance(val, str):
            row[short] = float(val)
        else:
            row[short] = ""

    return row


def print_table(rows, sort_by=None):
    """Print a formatted table of results."""
    if sort_by and sort_by in rows[0]:
        rows = sorted(rows, key=lambda r: r.get(sort_by, 0) if isinstance(r.get(sort_by, 0), (int, float)) else 0,
                      reverse=True)

    header = (f"{'exp_name':<35} {'d':>3} {'s':>5} {'status':<10} "
              f"{'envsteps':>12} {'%':>5} {'succ%':>6} {'ep_len':>7} {'reward':>9}")
    print(header)
    print("-" * len(header))

    for r in rows:
        succ = f"{r['episode_success_any']*100:.1f}" if isinstance(r.get('episode_success_any'), (int, float)) else "?"
        ep_len = f"{r['avg_episode_length']:.0f}" if isinstance(r.get('avg_episode_length'), (int, float)) else "?"
        reward = f"{r['episode_reward']:.1f}" if isinstance(r.get('episode_reward'), (int, float)) else "?"
        print(f"{r['exp_name']:<35} {str(r['depth']):>3} {str(r['seed']):>5} "
              f"{r['status']:<10} {r['envsteps']:>12,} {r['train_pct']:>4.0f}% "
              f"{succ:>5}% {ep_len:>7} {reward:>9}")

    done = sum(1 for r in rows if r['status'] == 'DONE')
    crashed = sum(1 for r in rows if r['status'] == 'CRASHED')
    running = sum(1 for r in rows if r['status'] == 'RUNNING')
    incomplete = sum(1 for r in rows if r['status'] == 'INCOMPLETE')
    print(f"\nTotal: {len(rows)} ({done} done, {crashed} crashed, {running} running, {incomplete} incomplete)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results from WandB")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="WandB entity")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="WandB project")
    parser.add_argument("--env", help="Filter by env_id")
    parser.add_argument("--state", help="Filter by status: done/crashed/running/incomplete")
    parser.add_argument("--tag", help="Filter by wandb tag")
    parser.add_argument("--sort", default=None, help="Sort by column (e.g. success_rate, episode_reward)")
    parser.add_argument("--csv", help="Save CSV to this path")
    parser.add_argument("--json", help="Save JSON to this path")
    args = parser.parse_args()

    # Build WandB filters
    filters = {}
    if args.env:
        filters["config.env_id"] = args.env
    if args.tag:
        filters["tags"] = {"$in": [args.tag]}

    print(f"Fetching runs from {args.entity}/{args.project}...", flush=True)
    runs = fetch_runs(args.entity, args.project, filters=filters)
    print(f"Found {len(runs)} runs")

    if not runs:
        print("No runs found.")
        return

    rows = [run_to_row(r) for r in runs]

    # Status filter (post-fetch, since WandB API doesn't filter on derived status)
    if args.state:
        state_map = {
            "done": "DONE", "finished": "DONE",
            "crashed": "CRASHED", "failed": "CRASHED",
            "running": "RUNNING",
            "incomplete": "INCOMPLETE",
        }
        target_status = state_map.get(args.state.lower(), args.state.upper())
        rows = [r for r in rows if r['status'] == target_status]
        print(f"Filtered to {len(rows)} runs with status={target_status}")

    if not rows:
        print("No runs match filter.")
        return

    print_table(rows, sort_by=args.sort)

    # Save CSV
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {args.csv}")

    # Save JSON
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f"\nJSON saved to {args.json}")


if __name__ == "__main__":
    main()
