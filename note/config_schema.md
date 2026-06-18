# Experiment Configuration Schema

The single source of truth is the YAML file (e.g. `infra/all_experiments.yaml`).
`train.py` reads YAML directly. No Experiment dataclass needed.

## YAML Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `exp_name` | str | **required** | Unique experiment name |
| `env_id` | str | **required** | Environment id |
| `depth` | int | **required** | Network depth (actor + critic) |
| `seed` | int | **required** | Random seed |
| `num_epochs` | int | 100 | Training epochs |
| `total_env_steps` | int | 100000000 | Total environment steps |
| `actor_network_width` | int | 256 | Actor network width |
| `critic_network_width` | int | 256 | Critic network width |
| `batch_size` | int | 512 | Training batch size |
| `num_envs` | int | 512 | Number of parallel environments |
| `actor_lr` | float | 0.0003 | Actor learning rate |
| `critic_lr` | float | 0.0003 | Critic learning rate |
| `alpha_lr` | float | 0.0003 | Alpha (entropy) learning rate |
| `unroll_length` | int | 62 | Steps per trajectory unroll |
| `max_replay_size` | int | 10000 | Max replay buffer size |
| `min_replay_size` | int | 1000 | Min replay buffer size |
| `entropy_param` | float | 0.5 | Entropy coefficient |
| `save_buffer` | int | 0 | Save replay buffer at end |
| `resume_from` | str | "auto" | Resume checkpoint path |
| `wandb_mode` | str | "online" | Wandb mode |
| `wandb_group` | str | "" | Wandb group (default: exp_name) |
| `wandb_project_name` | str | "scaling-crl-nano4" | Wandb project |
| `capture_vis` | bool | False | Render policy at end |
| `checkpoint` | bool | True | Enable checkpointing |
| `checkpoint_save_interval_epochs` | int | 10 | Save interval |
| `checkpoint_max_to_keep` | int | 3 | Max checkpoints to keep |
| `checkpoint_keep_period` | int | 50 | Force-keep interval |

## CLI-only Fields (not in YAML)

These are set via CLI args or use hardcoded defaults:

| Field | Default | Description |
|-------|---------|-------------|
| `track` | True | Enable wandb tracking |
| `wandb_entity` | "sungwayne99999-..." | Wandb entity |
| `wandb_dir` | "." | Wandb directory |
| `episode_length` | 1000 | Episode length |
| `num_eval_envs` | 128 | Eval environments |
| `gamma` | 0.99 | Discount factor |
| `logsumexp_penalty_coeff` | 0.1 | Logsumexp penalty |
| `disable_entropy` | 0 | Disable entropy |
| `use_relu` | 0 | Use ReLU instead of Swish |
| `num_episodes_per_env` | 1 | Episodes per env sample |
| `training_steps_multiplier` | 1 | Training steps multiplier |
| `use_all_batches` | 0 | Use all SGD batches |
| `num_sgd_batches_per_training_step` | 800 | SGD batches per step |
| `eval_actor` | 0 | Eval actor mode |
| `expl_actor` | 1 | Exploration actor mode |
| `num_render` | 10 | Render episodes |
| `vis_length` | 1000 | Render length |

## Usage

```bash
# Production (from YAML)
python train.py --yaml all_experiments.yaml --exp_name ant_d8_s1000

# Debug (from CLI)
python train.py --exp_name test --env_id ant --depth 8 --seed 42
```
