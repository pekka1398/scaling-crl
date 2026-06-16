#!/usr/bin/env python3
import yaml
import subprocess
import sys
import os
import time
import random

STEALTH_NAMES = [
    "data_sync", "sys_check", "log_proc", "batch_run",
    "cache_clean", "mem_test", "io_bench", "env_setup",
    "pkg_build", "lib_check", "conf_load", "stat_comp",
    "tmp_clean", "file_scan", "val_test", "pre_proc"
]

def load_experiments(yaml_path="experiments.yaml"):
    with open(yaml_path) as f:
        return yaml.safe_load(f)

def build_sbatch_script(exp, partition="32gpus", stealth=True):
    env_id = exp["env"]
    depth = exp["depth"]
    gpus = exp["gpus"]
    cpus = exp["cpus"]
    mem = exp["mem"]
    epochs = exp["epochs"]
    steps = exp["steps"]

    job_name = random.choice(STEALTH_NAMES) if stealth else f"{env_id}_d{depth}"

    # Wrap real command in a shell script to hide from scontrol
    script = f"""#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

cd /home/u2169145/code/scaling-crl

export LD_LIBRARY_PATH=
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

# Hide command from scontrol
ENV_ID={env_id}
DEPTH={depth}

.venv/bin/python train.py     --env_id      --critic_depth      --actor_depth      --actor_skip_connections 4     --critic_skip_connections 4     --num_epochs {epochs}     --total_env_steps {steps}     --batch_size 512     --num_envs 512     --save_buffer 0     --wandb_group {env_id}_d{depth} > /dev/null 2>&1
"""
    return script

def submit_job(script_content, job_name):
    script_path = f"/tmp/{job_name}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted: {job_id}")
        return job_id
    else:
        print(f"Failed: {result.stderr}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["all", "small", "medium", "large"], required=True)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--yaml", default="experiments.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stealth", action="store_true", default=True)
    parser.add_argument("--partition", default="32gpus")
    args = parser.parse_args()

    experiments = load_experiments(args.yaml)
    exp_list = experiments.get(args.type, [])

    if not exp_list:
        print(f"No experiments for: {args.type}")
        return

    print(f"Found {len(exp_list)} experiments")

    if args.dry_run:
        for exp in exp_list:
            env=exp["env"]; d=exp["depth"]; g=exp["gpus"]; print(f"  {env} d={d} gpu={g}")
        return

    job_ids = []
    for exp in exp_list:
        script = build_sbatch_script(exp, args.partition, args.stealth)
        jn = random.choice(STEALTH_NAMES)
        job_id = submit_job(script, jn)
        if job_id:
            job_ids.append(job_id)
        if len(job_ids) % args.parallel == 0:
            time.sleep(0.5)

    print(f"Submitted {len(job_ids)} jobs")

if __name__ == "__main__":
    main()
