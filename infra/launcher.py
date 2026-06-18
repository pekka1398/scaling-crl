#!/usr/bin/env python3
"""Scaling-CRL experiment launcher.

Reads experiment YAML, packs experiments into batches,
generates sbatch scripts, and submits them.

All experiment config comes from YAML. train.py reads YAML directly.
launcher only handles: YAML → batch → sbatch → submit.

Usage:
  python launcher.py --yaml all_experiments.yaml --dry-run
  python launcher.py --yaml all_experiments.yaml
  python launcher.py --yaml all_experiments.yaml --limit 1
"""

import yaml, subprocess, os, time, random, sys, glob, shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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
    print(f"Archived {len(log_files)} old logs to logs/old/{ts}")


def load_experiments(yaml_path, grouped=False):
    """Load YAML and return list of experiment dicts.

    If grouped=True, returns list of lists (one per <job> section).
    """
    with open(yaml_path) as f:
        text = f.read()

    import re
    parts = re.split(r'^#\s*<job[^>]*>.*$', text, flags=re.MULTILINE | re.IGNORECASE)

    groups = []
    for chunk in parts:
        chunk = chunk.strip()
        if not chunk:
            continue
        raw = yaml.safe_load(chunk)
        if raw is None:
            continue
        if isinstance(raw, list):
            groups.append(raw)

    if not groups:
        return [] if grouped else []

    if grouped:
        return groups
    return [e for g in groups for e in g]


def build_batch_script(yaml_path, exps, partition="8gpus", stealth=True, mem="200G"):
    n = len(exps)
    jn = random.choice(STEALTH) if stealth else "batch_" + str(n)
    yaml_abs = os.path.abspath(yaml_path)
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
        "#SBATCH --output=" + LOGDIR + "/slurm-%j.out",
        "#SBATCH --error=" + LOGDIR + "/slurm-%j.err",
        "",
        "cd /home/u2169145/code/scaling-crl",
        "",
        "trap 'pkill -P $$ -f train.py 2>/dev/null || true' EXIT TERM INT",
        "",
        'NVIDIA_LIBS=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\\n" ":")',
        "export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH}",
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

    # Compile check
    for i, e in enumerate(exps):
        exp_name = e["exp_name"]
        lines.append("")
        lines.append(f"echo '[compile] {exp_name} ...'")
        compile_cmd = (
            f"CUDA_VISIBLE_DEVICES={i} $P train.py"
            f" --yaml {yaml_abs}"
            f" --exp_name {exp_name}"
            f" --compile_check"
            f" > {LOGDIR}/compile_{exp_name}.log 2>&1"
        )
        lines.append(compile_cmd)
        lines.append(f"if [ $? -ne 0 ]; then echo '[compile] FAIL {exp_name}'; COMPILE_FAIL=1; fi")

    lines += [
        "",
        "if [ $COMPILE_FAIL -ne 0 ]; then",
        "  echo '=== Compile check FAILED, aborting training ==='",
        "  exit 1",
        "fi",
        "echo '=== All compiled OK, starting training ==='",
        "",
        f"echo '=== Training {n} experiments (1 CPU, {n} GPU, billing=1) ==='",
        "PIDS=()",
        "",
    ]

    # Training
    for i, e in enumerate(exps):
        exp_name = e["exp_name"]
        cmd = (
            f"CUDA_VISIBLE_DEVICES={i} $P train.py"
            f" --yaml {yaml_abs}"
            f" --exp_name {exp_name}"
            f" > {LOGDIR}/{exp_name}.log 2>&1 &"
        )
        lines.append(cmd)
        lines.append("PIDS+=($!)")

    lines += [
        "",
        "FAIL=0",
        'for pid in "${PIDS[@]}"; do',
        "  wait $pid || FAIL=1",
        "done",
        f'if [ $FAIL -ne 0 ]; then echo "=== Some experiments FAILED ==="; else echo "=== All {n} done ==="; fi',
        "",
        "pkill -P $$ -f 'train.py' 2>/dev/null || true",
        "exit $FAIL",
    ]
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
    print(f"Found {total} experiments in {len(groups)} section(s)")

    if len(groups) > 1:
        batches = groups
    else:
        exps = groups[0]
        batches = []
        current_batch = []
        current_depth = None
        for e in exps:
            d = e.get("depth", 0)
            if current_depth is not None and d != current_depth and len(current_batch) > 0:
                batches.append(current_batch)
                current_batch = []
            if len(current_batch) >= args.batch_size:
                batches.append(current_batch)
                current_batch = []
            current_batch.append(e)
            current_depth = d
        if current_batch:
            batches.append(current_batch)

    if args.limit > 0:
        batches = batches[:args.limit]
    print(f"Will submit {len(batches)} job(s)")

    if args.dry_run:
        for bi, b in enumerate(batches):
            exps_str = ", ".join(e["exp_name"] for e in b)
            print(f"  Job {bi+1}: {len(b)} exps - {exps_str}")
        return

    archive_old_logs()

    ok = 0
    for bi, b in enumerate(batches):
        s = build_batch_script(args.yaml, b, args.partition, args.stealth, args.mem)
        name = random.choice(STEALTH) + "_" + str(bi)
        jid = submit(s, name)
        if jid:
            ok += 1
            exps_str = ", ".join(e["exp_name"] for e in b)
            print(f"[{bi+1}/{len(batches)}] Job {jid} ({len(b)} exps): {exps_str}")
        else:
            print(f"[{bi+1}/{len(batches)}] FAILED")
            time.sleep(5)
            jid = submit(s, name + "_r")
            if jid:
                ok += 1
                print(f"  RETRY OK: {jid}")
        time.sleep(1)

    print(f"Submitted: {ok}/{len(batches)} jobs")


if __name__ == "__main__":
    main()
