#!/usr/bin/env python3
"""Scaling-CRL experiment launcher.

Scans conf/experiment/ for experiment configs, packs them into batches,
generates sbatch scripts, and submits them.

Usage:
  python infra/launcher.py --dry-run
  python infra/launcher.py
  python infra/launcher.py --limit 1
  python infra/launcher.py --experiments ant_d8_s2000 ant_d16_s2000
"""

import subprocess, os, time, random, sys, glob, shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

STEALTH = ["data_sync","sys_check","log_proc","batch_run","cache_clean","mem_test",
           "io_bench","env_setup","pkg_build","lib_check","conf_load","stat_comp",
           "tmp_clean","file_scan","val_test","pre_proc"]

LOGDIR = "/home/u2169145/code/scaling-crl/logs"
CONF_EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conf", "experiment")


def list_experiments():
    """List all experiment names from conf/experiment/."""
    exps = []
    for f in sorted(os.listdir(CONF_EXPERIMENT_DIR)):
        if f.endswith(".yaml"):
            exps.append(f[:-5])
    return exps


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


def pack_jobs(experiments, gpus_per_node=8):
    """Pack experiments into jobs, one experiment per GPU.

    Returns list of lists, each inner list is one SLURM job.
    """
    jobs = []
    for i in range(0, len(experiments), gpus_per_node):
        jobs.append(experiments[i:i + gpus_per_node])
    return jobs


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
        "# Parse SLURM GPU allocation (SLURM sets CUDA_VISIBLE_DEVICES)",
        'IFS="," read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"',
        "",
        "echo 'CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'",
        "echo 'GPUS=${GPUS[*]}'",
        "nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -5",
        "",
        "echo '=== Compile check: " + str(n) + " experiments ==='",
        "COMPILE_FAIL=0",
    ]

    # Compile check
    for i, exp_name in enumerate(exps):
        lines.append("")
        lines.append(f"echo '[compile] {exp_name} ...'")
        compile_cmd = (
            f"CUDA_VISIBLE_DEVICES=${{GPUS[{i}]}} $P train.py"
            f" --experiment {exp_name}"
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
    for i, exp_name in enumerate(exps):
        cmd = (
            f"CUDA_VISIBLE_DEVICES=${{GPUS[{i}]}} $P train.py"
            f" --experiment {exp_name}"
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
    parser = argparse.ArgumentParser(description="Scaling-CRL experiment launcher")
    parser.add_argument("--experiments", nargs="*", help="Specific experiments to run (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stealth", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0, help="Max number of jobs to submit (0=all)")
    parser.add_argument("--mem", default="200G", help="Memory per job (default: 200G)")
    parser.add_argument("--partition", default="8gpus")
    parser.add_argument("--gpus-per-node", type=int, default=8, help="GPUs per SLURM node")
    args = parser.parse_args()

    # Get experiment list
    if args.experiments:
        all_exps = args.experiments
    else:
        all_exps = list_experiments()

    if not all_exps:
        print("No experiments found"); return

    print(f"Found {len(all_exps)} experiment(s)")

    # Pack into jobs
    jobs = pack_jobs(all_exps, gpus_per_node=args.gpus_per_node)

    if args.limit > 0:
        jobs = jobs[:args.limit]
    print(f"Will submit {len(jobs)} job(s)")

    if args.dry_run:
        for ji, job in enumerate(jobs):
            print(f"  Job {ji+1}: {len(job)} exps - {', '.join(job)}")
        return

    archive_old_logs()

    ok = 0
    for ji, job in enumerate(jobs):
        s = build_batch_script(job, args.partition, args.stealth, args.mem)
        name = random.choice(STEALTH) + "_" + str(ji)
        jid = submit(s, name)
        if jid:
            ok += 1
            print(f"[{ji+1}/{len(jobs)}] Job {jid} ({len(job)} exps): {', '.join(job)}")
        else:
            print(f"[{ji+1}/{len(jobs)}] FAILED")
            time.sleep(5)
            jid = submit(s, name + "_r")
            if jid:
                ok += 1
                print(f"  RETRY OK: {jid}")
        time.sleep(1)

    print(f"Submitted: {ok}/{len(jobs)} jobs")


if __name__ == "__main__":
    main()
