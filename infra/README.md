# Scaling-CRL Infrastructure

## Architecture

Hydra structured configs + per-experiment YAML files in `conf/experiment/`.

All experiment naming uses `exp_name` as the single source of truth.
No ad-hoc string assembly — every log, checkpoint, and wandb identifier comes from `exp_name`.

## Experiment Definition

Each experiment is a self-contained YAML file in `conf/experiment/`.
Shared settings (network, optimizer, replay, etc.) are config groups in `conf/`.

```
conf/
├── config.py              # Structured config (all fields MISSING = required)
├── network.yaml           # Actor/critic width
├── optimizer.yaml         # Learning rates
├── replay.yaml            # Buffer, batch size, num_envs
├── rl.yaml                # RL hyperparameters
├── eval.yaml              # Eval settings
├── wandb.yaml             # Wandb project/entity
├── checkpoint.yaml        # Checkpoint settings
└── experiment/
    ├── ant_d8_s2000.yaml  # One file per experiment
    ├── ant_d16_s2000.yaml
    └── ...
```

Each experiment file includes config groups and overrides only what varies:

```yaml
# @package _global_
defaults:
  - /network
  - /optimizer
  - /replay
  - /rl
  - /eval
  - /wandb
  - /checkpoint

exp_name: ant_d8_s2000
env_id: ant
depth: 8
seed: 2000
num_epochs: 100
total_env_steps: 100000000
```

## Usage

    # Run single experiment
    python train.py --experiment ant_d8_s1000

    # Run with CLI overrides
    python train.py --experiment ant_d8_s1000 actor_lr=0.001

    # Compile check (1 epoch, no checkpoint)
    python train.py --experiment ant_d8_s1000 --compile_check

## Launcher

### Option A: Custom launcher (recommended, handles compile checks + GPU packing)

    # Preview all jobs
    python infra/launcher.py --dry-run

    # Submit all
    python infra/launcher.py

    # Submit specific experiments
    python infra/launcher.py --experiments ant_d8_s2000 ant_d16_s2000

    # Limit to first job
    python infra/launcher.py --limit 1

### Option B: Hydra submitit launcher (pure Python, no shell scripts)

    # Submit single experiment to SLURM
    python train.py -m experiment=ant_d8_s2000 hydra/launcher=submitit_slurm

    # Submit multiple experiments (each becomes a SLURM job)
    python train.py -m experiment=ant_d8_s2000,ant_d32_s2000 hydra/launcher=submitit_slurm

    # Override partition
    python train.py -m experiment=ant_d8_s2000 hydra/launcher=submitit_slurm hydra.launcher.partition=dev

## Naming Convention

| Artifact | Path | Example |
|---|---|---|
| Compile log | `logs/compile_{exp_name}.log` | `logs/compile_ant_d8_s1000.log` |
| Training log | `logs/{exp_name}.log` | `logs/ant_d8_s1000.log` |
| Checkpoint dir | `runs/{exp_name}/` | `runs/ant_d8_s1000/` |
| Checkpoint dir | `runs/{exp_name}/checkpoints/` | `runs/ant_d8_s1000/checkpoints/` |
| Args dump | `runs/{exp_name}/args.pkl` | `runs/ant_d8_s1000/args.pkl` |
| Wandb group | `{exp_name}` | `ant_d8_s1000` |

## Job Layout

- 1 CPU, N GPU per job (one per experiment in the batch)
- Compile check before training (JIT warmup, 1 epoch, no checkpoint)
- Wandb online mode (compute nodes have network access)
- Default memory: 200G per job
- Checkpoints via Orbax (atomic writes, auto-cleanup keep=3, auto-resume)
- Legacy pickle checkpoints still loadable for old experiments

## Monitoring

SLURM cluster status (terminal):

    bash infra/monitor.sh              # all
    bash infra/monitor.sh nodes        # per-node GPU
    bash infra/monitor.sh running      # running jobs
    bash infra/monitor.sh pending      # pending jobs

SLURM → WandB dashboard (browser):

    # One-shot
    python infra/monitor_to_wandb.py

    # Continuous (refresh every 60s)
    python infra/monitor_to_wandb.py --loop 60

    # Background
    nohup python infra/monitor_to_wandb.py --loop 60 &

## Sweeps (disabled by default)

WandB Sweeps for hyperparameter search. Not used for current paper (depth is the independent variable, other hyperparams are fixed).

For future papers with hyperparameter search:

    # Create sweep config
    vi conf/sweep/my_sweep.yaml

    # Create and run sweep
    python infra/sweep.py --config conf/sweep/my_sweep.yaml

    # Create sweep only (run agents on SLURM later)
    python infra/sweep.py --config conf/sweep/my_sweep.yaml --create-only

    # Run agent for existing sweep
    python infra/sweep.py --sweep-id <entity>/<project>/<sweep_id>

## Evaluation

    # Eval a single experiment
    .venv/bin/python eval.py --exp_name ant_d8_s1000

    # Eval all experiments (skips those with existing eval_metrics.json)
    .venv/bin/python eval.py --all

    # Force re-eval everything
    .venv/bin/python eval.py --all --force

## Visualization

    # Render a single experiment's policy as vis.html
    .venv/bin/python render.py --exp_name ant_d8_s1000

    # Render all experiments (skips those with existing vis.html)
    .venv/bin/python render.py --all

    # Force re-render everything
    .venv/bin/python render.py --all --force

## Checkpoints (WandB Artifacts)

Checkpoints are automatically uploaded to WandB as Artifacts after training.
On the WandB dashboard: Run page → Artifacts tab → download any checkpoint.

No manual backup needed — checkpoints are linked to their training run.

## Results

Primary: Use WandB web UI (Workspace → Add Panel → Weave Table).

For scripting/CSV export:

    python collect_results.py
    python collect_results.py --csv results.csv
    python collect_results.py --env ant --sort episode_reward
