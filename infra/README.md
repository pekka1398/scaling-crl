# Scaling-CRL Infrastructure

## Architecture

Single entry point: `launcher.py` + YAML experiment definitions.

All experiment naming uses `config.Experiment.exp_name` as the single source of truth.
No ad-hoc string assembly — every log, checkpoint, and wandb identifier comes from `exp_name`.

## Experiment Definition

`config.py` defines the `Experiment` dataclass with all hyperparameters.
YAML files list experiments; each entry maps 1:1 to `Experiment`.

```yaml
- exp_name: ant_d8_s1000    # identity — all naming uses this
  env_id: ant
  depth: 8
  seed: 1000
  num_epochs: 100
  total_env_steps: 100000000
```

Fields with defaults (can omit from YAML):
`actor_skip_connections: 4`, `critic_skip_connections: 4`,
`batch_size: 512`, `num_envs: 512`, `actor_lr: 3e-4`, etc.

## Usage

    # Preview
    python launcher.py --yaml all_experiments.yaml --dry-run

    # Submit all
    python launcher.py --yaml all_experiments.yaml

    # Limit to first job
    python launcher.py --yaml all_experiments.yaml --limit 1

    # Override memory
    python launcher.py --yaml all_experiments.yaml --mem 100G

## Naming Convention

| Artifact | Path | Example |
|---|---|---|
| Compile log | `logs/compile_{exp_name}.log` | `logs/compile_ant_d8_s1000.log` |
| Training log | `logs/{exp_name}.log` | `logs/ant_d8_s1000.log` |
| Checkpoint dir | `runs/{exp_name}/` | `runs/ant_d8_s1000/` |
| Checkpoint file | `runs/{exp_name}/step_{N}.pkl` | `runs/ant_d8_s1000/step_5000000.pkl` |
| Args dump | `runs/{exp_name}/args.pkl` | `runs/ant_d8_s1000/args.pkl` |
| Wandb group | `{exp_name}` | `ant_d8_s1000` |

## Job Layout

- 1 CPU, N GPU per job (one per experiment in the batch)
- Compile check before training (JIT warmup, 1 epoch, no checkpoint)
- WANDB_MODE=offline, sync via wandb-osh daemon on login node
- Default memory: 200G per job

## Monitoring

    # Snapshot daemon (run on login node)
    bash infra/start_daemons.sh

    # View cluster status
    bash infra/monitor.sh              # all
    bash infra/monitor.sh nodes        # per-node GPU
    bash infra/monitor.sh running      # running jobs
    bash infra/monitor.sh pending      # pending jobs

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

    # Customize number of episodes
    .venv/bin/python render.py --exp_name ant_d8_s1000 --num_episodes 5

## Results

    # Print summary table
    python collect_results.py

    # Save CSV/JSON
    python collect_results.py --csv results.csv --json results.json
