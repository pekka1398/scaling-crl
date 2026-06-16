#!/usr/bin/env python3
"""Scaling-CRL experiment launcher.

Reads experiments.yaml, packs experiments into batches,
generates sbatch scripts with compile check, and submits them.

Usage:
  python launcher.py --type all --dry-run    # preview
  python launcher.py --type all              # submit all
  python launcher.py --type all --limit 1    # submit only first batch
"""

import yaml, subprocess, os, time, random

STEALTH = ["data_sync","sys_check","log_proc","batch_run","cache_clean","mem_test",
           "io_bench","env_setup","pkg_build","lib_check","conf_load","stat_comp",
           "tmp_clean","file_scan","val_test","pre_proc"]

BATCH_SIZE = 8
LOGDIR = "/home/u2169145/code/scaling-crl/logs"


def load_experiments(yaml_path=None):
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.yaml")
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def build_batch_script(exps, partition="8gpus", stealth=True):
    n = len(exps)
    jn = random.choice(STEALTH) if stealth else "batch_" + str(n)
    common_args = "--actor_skip_connections 4 --critic_skip_connections 4 --batch_size 512 --num_envs 512 --save_buffer 0"
    lines = [
        "#!/bin/bash",
        "#SBATCH --account=MST114560",
        "#SBATCH --job-name=" + jn,
        "#SBATCH --partition=" + partition,
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:" + str(n),
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=400G",
        "#SBATCH --time=48:00:00",
        "",
        "cd /home/u2169145/code/scaling-crl",
        "",
        'NVIDIA_LIBS=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\\n" ":")',
        "export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH}",
        "export WANDB_MODE=offline",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "export XLA_PYTHON_CLIENT_ALLOCATOR=platform",
        "export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache",
        "",
        "mkdir -p " + LOGDIR,
        "P=.venv/bin/python",
        'COMMON="' + common_args + '"',
        "",
        "echo '=== Compile check: " + str(n) + " experiments ==='",
        "COMPILE_FAIL=0",
    ]
    # Compile check: run a tiny forward pass for each env+depth to trigger JIT
    for i, e in enumerate(exps):
        group = e["env"] + "_d" + str(e["depth"])
        lines.append("")
        lines.append("echo '[compile] " + e["env"] + " d=" + str(e["depth"]) + " ...'")
        compile_cmd = (
            "CUDA_VISIBLE_DEVICES=" + str(i) + " $P train.py"
            " --env_id " + e["env"] +
            " --critic_depth " + str(e["depth"]) +
            " --actor_depth " + str(e["depth"]) +
            " --num_epochs 1 --total_env_steps 1000 $COMMON"
            " --wandb_group compile_" + group +
            " > " + LOGDIR + "/compile_" + group + ".log 2>&1"
        )
        lines.append(compile_cmd)
        lines.append("if [ $? -ne 0 ]; then echo '[compile] FAIL " + e["env"] + " d=" + str(e["depth"]) + "'; COMPILE_FAIL=1; fi")

    lines.append("")
    lines.append("if [ $COMPILE_FAIL -ne 0 ]; then")
    lines.append("  echo '=== Compile check FAILED, aborting training ==='")
    lines.append("  exit 1")
    lines.append("fi")
    lines.append("echo '=== All compiled OK, starting training ==='")
    lines.append("")

    # Training
    lines.append("echo '=== Training " + str(n) + " experiments (1 CPU, " + str(n) + " GPU, billing=1) ==='")
    lines.append("")
    for i, e in enumerate(exps):
        group = e["env"] + "_d" + str(e["depth"])
        cmd = (
            "CUDA_VISIBLE_DEVICES=" + str(i) + " $P train.py"
            " --env_id " + e["env"] +
            " --critic_depth " + str(e["depth"]) +
            " --actor_depth " + str(e["depth"]) +
            " --num_epochs " + str(e["epochs"]) +
            " --total_env_steps " + str(e["steps"]) +
            " $COMMON"
            " --wandb_group " + group +
            " > " + LOGDIR + "/" + group + ".log 2>&1 &"
        )
        lines.append(cmd)
    lines.append("")
    lines.append("wait")
    lines.append("echo '=== All " + str(n) + " done ==='")
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
    parser.add_argument("--type", required=True, choices=["all"])
    parser.add_argument("--partition", default="8gpus")
    parser.add_argument("--yaml", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stealth", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="Max number of jobs to submit (0=all)")
    args = parser.parse_args()

    exps = load_experiments(args.yaml)
    el = exps.get(args.type, [])
    if not el:
        print("No experiments"); return

    print("Found " + str(len(el)) + " experiments")

    batches = [el[i:i+args.batch_size] for i in range(0, len(el), args.batch_size)]
    if args.limit > 0:
        batches = batches[:args.limit]
    print("Will submit " + str(len(batches)) + " job(s) of up to " + str(args.batch_size) + " experiments each")

    if args.dry_run:
        for bi, b in enumerate(batches):
            print("  Job " + str(bi+1) + ": " + str(len(b)) + " exps - " + ", ".join(e["env"] + "_d" + str(e["depth"]) for e in b))
        return

    ok = 0
    for bi, b in enumerate(batches):
        s = build_batch_script(b, args.partition, args.stealth)
        name = random.choice(STEALTH) + "_" + str(bi)
        jid = submit(s, name)
        if jid:
            ok += 1
            exps_str = ", ".join(e["env"] + "_d" + str(e["depth"]) for e in b)
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
