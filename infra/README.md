# Scaling-CRL Infrastructure

## Architecture

Single entry point: `launcher.py` + `experiments.yaml`

No manual shell scripts. All job submission goes through the launcher.

## Consensus

- 1 CPU, 8 GPU per job → billing = 1
- 8 experiments per job (CUDA_VISIBLE_DEVICES 0-7)
- Job finishes after its batch, no auto-continuation
- Compile check before training (JIT warmup, 1 epoch / 1000 steps)
- All logs go to `logs/` (one .log per experiment, compile logs separate)
- No SBATCH output/error files
- Stealth job names (random generic names)
- WANDB_MODE=offline, sync later from login node

## Usage

    # Preview what will be submitted
    python launcher.py --type all --dry-run

    # Submit all experiments (3 jobs x 8 exps)
    python launcher.py --type all

    # Custom batch size (e.g. 4 GPUs)
    python launcher.py --type all --batch-size 4

## experiments.yaml

Defines all experiments under `all:`. Each entry:

    {env: ant, depth: 8, gpus: 1, cpus: 1, mem: 50G, epochs: 100, steps: 100000000}

`cpus` and `mem` are per-experiment metadata; actual job always uses 1 CPU / 400G mem.

## Log files

    logs/compile_ant_d8.log     # compile check output
    logs/ant_d8.log             # training output
    logs/ant_big_maze_d8.log
