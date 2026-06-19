#!/usr/bin/env python3
"""Push SLURM cluster status to WandB dashboard.

Run on login node — opens a persistent WandB run that logs SLURM state as time-series.
View from browser: wandb.ai → project "scaling-crl-v2" → run "cluster-monitor"

Usage:
  python infra/monitor_to_wandb.py              # run once
  python infra/monitor_to_wandb.py --loop 60    # refresh every 60s
  python infra/monitor_to_wandb.py --loop 60 --project scaling-crl-v2
"""

import argparse
import subprocess
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


def get_my_jobs():
    """Get my SLURM job counts by state."""
    raw = run("squeue -u $USER -o '%T|%b|%M|%R' --noheader")
    pending = 0
    running = 0
    pending_gpus = 0
    running_gpus = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        state = parts[0] if len(parts) > 0 else ""
        gres = parts[1] if len(parts) > 1 else ""
        runtime = parts[2] if len(parts) > 2 else ""
        reason = parts[3] if len(parts) > 3 else ""

        gpu = 0
        if "gpu" in gres.lower():
            for tok in gres.replace("gres/gpu:", "").split(":"):
                if tok.isdigit():
                    gpu = int(tok)
                    break

        if state == "PENDING":
            pending += 1
            pending_gpus += gpu
        elif state == "RUNNING":
            running += 1
            running_gpus += gpu

    return pending, running, pending_gpus, running_gpus


def get_cluster_status():
    """Get cluster-wide SLURM queue depth."""
    raw = run("squeue -o '%T' --noheader")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    total = len(lines)
    running = sum(1 for l in lines if l == "RUNNING")
    pending = sum(1 for l in lines if l == "PENDING")
    return total, running, pending


def get_gpu_utilization():
    """Get cluster GPU allocation from sinfo."""
    raw = run('sinfo -h -N -o "%G %C" 2>/dev/null')
    total_gpus = 0
    alloc_gpus = 0
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        gres = parts[0]  # e.g. gpu:H200:8
        cpus = parts[1:]  # e.g. ['48', '0', '48', '48']
        # Extract GPU count
        gpu = 0
        if "gpu" in gres.lower():
            for tok in gres.split(":"):
                if tok.isdigit():
                    gpu = int(tok)
                    break
        total_gpus += gpu
        # alloc/total from CPU fields isn't GPU-specific, skip for now
    return total_gpus, alloc_gpus


def collect_and_log():
    """Collect SLURM state and return as dict for wandb.log()."""
    my_pending, my_running, my_pending_gpus, my_running_gpus = get_my_jobs()
    cluster_total, cluster_running, cluster_pending = get_cluster_status()

    metrics = {
        # My jobs
        "slurm/my_pending_jobs": my_pending,
        "slurm/my_running_jobs": my_running,
        "slurm/my_pending_gpus": my_pending_gpus,
        "slurm/my_running_gpus": my_running_gpus,
        # Cluster-wide
        "slurm/cluster_total_jobs": cluster_total,
        "slurm/cluster_running_jobs": cluster_running,
        "slurm/cluster_pending_jobs": cluster_pending,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Push SLURM status to WandB")
    parser.add_argument("--loop", type=int, default=0, help="Refresh every N seconds (0=one-shot)")
    parser.add_argument("--project", default="scaling-crl-v2", help="WandB project")
    parser.add_argument("--entity", default="sungwayne99999-national-cheng-kung-university-co-op")
    parser.add_argument("--interval", type=int, default=60, help="Log interval in seconds (with --loop)")
    args = parser.parse_args()

    import wandb

    run_obj = wandb.init(
        project=args.project,
        entity=args.entity,
        name="cluster-monitor",
        job_type="monitor",
        resume="allow",
        id="cluster-monitor",
    )

    if args.loop > 0:
        print(f"Monitoring SLURM → WandB (every {args.interval}s). Ctrl+C to stop.")
        step = 0
        while True:
            try:
                metrics = collect_and_log()
                wandb.log(metrics, step=step)
                step += 1
                status = (f"  step {step}: my={metrics['slurm/my_running_jobs']}R/"
                          f"{metrics['slurm/my_pending_jobs']}P, "
                          f"cluster={metrics['slurm/cluster_running_jobs']}R/"
                          f"{metrics['slurm/cluster_pending_jobs']}P")
                print(f"[{time.strftime('%H:%M:%S')}]{status}")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopping monitor.")
                break
            except Exception as e:
                print(f"Error: {e}, retrying in {args.interval}s...")
                time.sleep(args.interval)
    else:
        metrics = collect_and_log()
        wandb.log(metrics)
        print("Logged once:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    wandb.finish()


if __name__ == "__main__":
    main()
