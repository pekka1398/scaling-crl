#!/usr/bin/env python3
"""Scaling-CRL experiment launcher.

Reads experiment YAML, packs experiments into batches,
generates sbatch scripts with compile check, and submits them.

All naming uses Experiment.exp_name as the single source of truth.
No ad-hoc string assembly for log/ckpt/wandb names.

Usage:
  python launcher.py --yaml experiments.yaml --dry-run    # preview
  python launcher.py --yaml experiments.yaml              # submit all
  python launcher.py --yaml experiments.yaml --limit 1    # submit only first batch
"""

import yaml, subprocess, os, time, random, sys, glob, shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import Experiment

STEALTH = ["data_sync","sys_check","log_proc","batch_run","cache_clean","mem_test",
           "io_bench","env_setup","pkg_build","lib_check","conf_load","stat_comp",
           "tmp_clean","file_scan","val_test","pre_proc"]

BATCH_SIZE = 8
LOGDIR = "/home/u2169145/code/scaling-crl/logs"


def archive_old_logs():
    """Move existing logs to logs/old/{timestamp}/ before a new launch."""
    if not os.path.isdir(LOGDIR):
        return
    log_files = glob.glob(os.path.join(LOGDIR, "*.log"))
    if not log_files:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(LOGDIR, "old", ts)
    os.makedirs(archive, exist_ok=True)
    for f in log_files:
        shutil.move(f, os.path.join(archive, os.path.basename(f)))
    print("Archived " + str(len(log_files)) + " old logs to logs/old/" + ts)


def load_experiments(yaml_path=None, grouped=False):
    """Load YAML and return list of Experiment objects.

    If grouped=True, returns list of lists (one per <job> section).
    <job> or <job=N> markers in comments split the file into job groups.
    Without markers, returns [all_exps] (single group).
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.yaml")
    with open(yaml_path) as f:
        text = f.read()

    import re
    # Split on lines like "# <job>", "# <job 1>", "# <job=3> description"
    parts = re.split(r'^#\s*<job[^>]*>.*$', text, flags=re.MULTILINE | re.IGNORECASE)

    groups = []
    for chunk in parts:
        chunk = chunk.strip()
        if not chunk:
            continue
        raw = yaml.safe_load(chunk)
        if raw is None:
            continue
        exps = [Experiment(**entry) for entry in raw]
        if exps:
            groups.append(exps)

    if not groups:
        return [] if grouped else []

    if grouped:
        return groups
    # Flat mode: concatenate all groups
    return [e for g in groups for e in g]


def build_batch_script(exps, partition="8gpus", stealth=True, mem="200G"):
    n = len(exps)
    jn = random.choice(STEALTH) if stealth else "batch_" + str(n)
    lines = [
        "#!/bin/bash",
        "#SBATCH --account=MST114560",
        "#SBATCH --job-name=" + jn,
        "#SBATCH --partition=" + partition,
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:" + str(n),
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=" + mem,
        "#SBATCH --time=48:00:00",
        "",
        "cd /home/u2169145/code/scaling-crl",
        "",
        "# Kill stray processes on exit or signal (zombie prevention)",
        "trap 'pkill -P $$ -f train.py 2>/dev/null || true' EXIT TERM INT",
        "",
        'NVIDIA_LIBS=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\\n" ":")',
        "export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH}",
        "export WANDB_MODE=online",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "export XLA_PYTHON_CLIENT_ALLOCATOR=platform",
        "export JAX_COMPILATION_CACHE_DIR=/home/u2169145/.cache/jax",
        "",
        "mkdir -p " + LOGDIR,
        "P=.venv/bin/python",
        "",
        "echo '=== Compile check: " + str(n) + " experiments ==='",
        "COMPILE_FAIL=0",
    ]
    # Compile check: 1 epoch with 1M steps to trigger JIT compile
    for i, e in enumerate(exps):
        lines.append("")
        lines.append("echo '[compile] " + e.exp_name + " ...'")
        compile_cmd = (
            "CUDA_VISIBLE_DEVICES=" + str(i) + " $P train.py"
            " --exp_name compile_" + e.exp_name +
            " --env_id " + e.env_id +
            " --critic_depth " + str(e.depth) +
            " --actor_depth " + str(e.depth) +
            " --seed " + str(e.seed) +
            " --num_epochs 1 --total_env_steps 1000000"
            " --batch_size " + str(e.batch_size) +
            " --num_envs " + str(e.num_envs) +
            " --actor_skip_connections " + str(e.actor_skip_connections) +
            " --critic_skip_connections " + str(e.critic_skip_connections) +
            " --no-capture-vis --no-checkpoint --no-track"
            " > " + LOGDIR + "/compile_" + e.exp_name + ".log 2>&1"
        )
        lines.append(compile_cmd)
        lines.append("if [ $? -ne 0 ]; then echo '[compile] FAIL " + e.exp_name + "'; COMPILE_FAIL=1; fi")

    lines.append("")
    lines.append("if [ $COMPILE_FAIL -ne 0 ]; then")
    lines.append("  echo '=== Compile check FAILED, aborting training ==='")
    lines.append("  exit 1")
    lines.append("fi")
    lines.append("echo '=== All compiled OK, starting training ==='")
    lines.append("")

    # Training
    lines.append("echo '=== Training " + str(n) + " experiments (1 CPU, " + str(n) + " GPU, billing=1) ==='")
    lines.append("PIDS=()")
    lines.append("")
    for i, e in enumerate(exps):
        cmd = (
            "CUDA_VISIBLE_DEVICES=" + str(i) + " $P train.py"
            " --exp_name " + e.exp_name +
            " --env_id " + e.env_id +
            " --critic_depth " + str(e.depth) +
            " --actor_depth " + str(e.depth) +
            " --seed " + str(e.seed) +
            " --num_epochs " + str(e.num_epochs) +
            " --total_env_steps " + str(e.total_env_steps) +
            " --batch_size " + str(e.batch_size) +
            " --num_envs " + str(e.num_envs) +
            " --actor_skip_connections " + str(e.actor_skip_connections) +
            " --critic_skip_connections " + str(e.critic_skip_connections) +
            " --no-capture-vis"
            " --wandb_mode offline"
            " --wandb_group " + e.exp_name
        )
        if e.resume_from:
            cmd += " --resume_from " + e.resume_from
        else:
            cmd += " --resume_from auto"
        if e.save_buffer:
            cmd += " --save_buffer " + str(e.save_buffer)
        cmd += " > " + LOGDIR + "/" + e.exp_name + ".log 2>&1 &"
        lines.append(cmd)
        lines.append("PIDS+=($!)")
    lines.append("")
    lines.append("FAIL=0")
    lines.append("for pid in \"${PIDS[@]}\"; do")
    lines.append("  wait $pid || FAIL=1")
    lines.append("done")
    lines.append("if [ $FAIL -ne 0 ]; then echo '=== Some experiments FAILED ==='; else echo '=== All " + str(n) + " done ==='; fi")
    lines.append("")
    lines.append("# Cleanup: kill any stray python processes before exiting")
    lines.append("pkill -P $$ -f 'train.py' 2>/dev/null || true")
    lines.append("exit $FAIL")
    return "\n".join(lines) + "\n"


def submit(script, name):
    p = "/tmp/" + name + ".sh"
    with open(p, "w") as f:
        f.write(script)
    r = subprocess.run(["sbatch", p], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip().split()[-1]
    else:
        print("  sbatch error: " + r.stderr.strip())
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Path to experiment YAML file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stealth", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="Max number of jobs to submit (0=all)")
    parser.add_argument("--mem", default="200G", help="Memory per job (default: 200G)")
    parser.add_argument("--partition", default="8gpus")
    args = parser.parse_args()

    groups = load_experiments(args.yaml, grouped=True)
    if not groups:
        print("No experiments"); return

    total = sum(len(g) for g in groups)
    print("Found " + str(total) + " experiments in " + str(len(groups)) + " section(s)")

    if len(groups) > 1:
        # <job> markers present — each section is one job
        batches = groups
    else:
        # No markers — auto-chunk by depth boundary
        exps = groups[0]
        batches = []
        current_batch = []
        current_depth = None
        for e in exps:
            if current_depth is not None and e.depth != current_depth and len(current_batch) > 0:
                batches.append(current_batch)
                current_batch = []
            if len(current_batch) >= args.batch_size:
                batches.append(current_batch)
                current_batch = []
            current_batch.append(e)
            current_depth = e.depth
        if current_batch:
            batches.append(current_batch)

    if args.limit > 0:
        batches = batches[:args.limit]
    print("Will submit " + str(len(batches)) + " job(s)")

    if args.dry_run:
        for bi, b in enumerate(batches):
            exps_str = ", ".join(e.exp_name + ("[R]" if e.resume_from else "") for e in b)
            print("  Job " + str(bi+1) + ": " + str(len(b)) + " exps - " + exps_str)
        return

    archive_old_logs()

    ok = 0
    for bi, b in enumerate(batches):
        s = build_batch_script(b, args.partition, args.stealth, args.mem)
        name = random.choice(STEALTH) + "_" + str(bi)
        jid = submit(s, name)
        if jid:
            ok += 1
            exps_str = ", ".join(e.exp_name for e in b)
            print("[" + str(bi+1) + "/" + str(len(batches)) + "] Job " + jid + " (" + str(len(b)) + " exps): " + exps_str)
        else:
            print("[" + str(bi+1) + "/" + str(len(batches)) + "] FAILED")
            time.sleep(5)
            jid = submit(s, name + "_r")
            if jid:
                ok += 1
                print("  RETRY OK: " + jid)
        time.sleep(1)

    print("Submitted: " + str(ok) + "/" + str(len(batches)) + " jobs")


if __name__ == "__main__":
    main()
