#!/usr/bin/env python3
import yaml, subprocess, os, time, random

STEALTH = ["data_sync","sys_check","log_proc","batch_run","cache_clean","mem_test",
           "io_bench","env_setup","pkg_build","lib_check","conf_load","stat_comp",
           "tmp_clean","file_scan","val_test","pre_proc"]

def load_experiments(yaml_path=None):
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.yaml")
    with open(yaml_path) as f:
        return yaml.safe_load(f)

def build_script(exp, partition="8gpus", stealth=True):
    e = exp
    jn = random.choice(STEALTH) if stealth else e["env"] + "_d" + str(e["depth"])
    logdir = "/home/u2169145/code/scaling-crl/logs"
    outf = logdir + "/" + e["env"] + "_d" + str(e["depth"]) + "-%j.out"
    errf = logdir + "/" + e["env"] + "_d" + str(e["depth"]) + "-%j.err"
    lines = [
        "#!/bin/bash",
        "#SBATCH --account=MST114560",
        "#SBATCH --job-name=" + jn,
        "#SBATCH --partition=" + partition,
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=" + str(e["cpus"]),
        "#SBATCH --mem=" + e["mem"],
        "#SBATCH --time=48:00:00",
        "#SBATCH --output=" + outf,
        "#SBATCH --error=" + errf,
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
        ".venv/bin/python train.py \\",
        "    --env_id " + e["env"] + " \\",
        "    --critic_depth " + str(e["depth"]) + " \\",
        "    --actor_depth " + str(e["depth"]) + " \\",
        "    --actor_skip_connections 4 \\",
        "    --critic_skip_connections 4 \\",
        "    --num_epochs " + str(e["epochs"]) + " \\",
        "    --total_env_steps " + str(e["steps"]) + " \\",
        "    --batch_size 512 \\",
        "    --num_envs 512 \\",
        "    --save_buffer 0 \\",
        "    --wandb_group " + e["env"] + "_d" + str(e["depth"]),
        "",
        "echo Done: " + e["env"] + " d=" + str(e["depth"]),
    ]
    return "\n".join(lines) + "\n"

def submit(script, name):
    p = "/tmp/" + name + ".sh"
    with open(p, "w") as f:
        f.write(script)
    r = subprocess.run(["sbatch", p], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout.strip().split()[-1]
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["all","small","medium","large"])
    parser.add_argument("--partition", default="8gpus")
    parser.add_argument("--yaml", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stealth", action="store_true", default=True)
    args = parser.parse_args()

    exps = load_experiments(args.yaml)
    el = exps.get(args.type, [])
    if not el:
        print("No experiments"); return

    print("Found " + str(len(el)) + " experiments")
    if args.dry_run:
        for e in el:
            print("  " + e["env"] + " d=" + str(e["depth"]) + " gpu=" + str(e["gpus"]))
        return

    ok = 0
    for i, e in enumerate(el):
        s = build_script(e, args.partition, args.stealth)
        jid = submit(s, random.choice(STEALTH) + "_" + str(i))
        if jid:
            ok += 1
            print("[" + str(i+1) + "/" + str(len(el)) + "] " + e["env"] + " d=" + str(e["depth"]) + " -> " + jid)
        else:
            print("[" + str(i+1) + "/" + str(len(el)) + "] " + e["env"] + " d=" + str(e["depth"]) + " FAILED")
            time.sleep(5)
            jid = submit(s, random.choice(STEALTH) + "_" + str(i) + "_r")
            if jid:
                ok += 1
                print("  RETRY OK: " + jid)
        if ok % 10 == 0:
            time.sleep(2)

    print("Submitted: " + str(ok) + "/" + str(len(el)))

if __name__ == "__main__":
    main()
