# Experiment Configuration Schema

All experiment config comes from YAML, validated via Hydra structured config (`conf/config.py`).
No defaults in code. Missing field = error at config load time.

## Structured Config

`conf/config.py` defines `ExperimentConfig` as a dataclass where every field is `MISSING`.
Hydra raises `MissingMandatoryValue` if any required field is not provided.

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `exp_name` | str | Unique experiment name |
| `env_id` | str | Environment id |
| `depth` | int | Network depth (actor + critic) |
| `seed` | int | Random seed |
| `num_epochs` | int | Training epochs |
| `total_env_steps` | int | Total environment steps |
| `actor_network_width` | int | Actor network width |
| `critic_network_width` | int | Critic network width |
| `batch_size` | int | Training batch size |
| `num_envs` | int | Number of parallel environments |
| `actor_lr` | float | Actor learning rate |
| `critic_lr` | float | Critic learning rate |
| `alpha_lr` | float | Alpha (entropy) learning rate |
| `unroll_length` | int | Steps per trajectory unroll |
| `max_replay_size` | int | Max replay buffer size |
| `min_replay_size` | int | Min replay buffer size |
| `entropy_param` | float | Entropy coefficient |
| `gamma` | float | Discount factor |
| `logsumexp_penalty_coeff` | float | Logsumexp penalty |
| `disable_entropy` | int | Disable entropy (0/1) |
| `use_relu` | int | Use ReLU instead of Swish (0/1) |
| `num_episodes_per_env` | int | Episodes per env sample |
| `training_steps_multiplier` | int | Training steps multiplier |
| `use_all_batches` | int | Use all SGD batches (0/1) |
| `num_sgd_batches_per_training_step` | int | SGD batches per step |
| `eval_actor` | int | Eval actor mode (0=deterministic, 1=stochastic, K=multi-sample) |
| `expl_actor` | int | Exploration actor mode |
| `num_eval_envs` | int | Eval environments |
| `episode_length` | int | Episode length |
| `resume_from` | str | Resume checkpoint path ("auto" or path) |
| `checkpoint` | bool | Enable checkpointing |
| `checkpoint_save_interval_epochs` | int | Save interval |
| `checkpoint_max_to_keep` | int | Max checkpoints to keep |
| `checkpoint_keep_period` | int | Force-keep interval |
| `save_buffer` | int | Save replay buffer at end (0/1) |
| `capture_vis` | bool | Render policy at end |
| `num_render` | int | Render episodes |
| `track` | bool | Enable wandb tracking |
| `wandb_mode` | str | Wandb mode ("online"/"offline") |
| `wandb_group` | str | Wandb group ("" = use exp_name) |
| `wandb_project_name` | str | Wandb project |
| `wandb_entity` | str | Wandb entity |
| `wandb_dir` | str | Wandb directory |

## Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_env_id` | str | None | Eval environment (defaults to env_id at runtime) |

## Usage

```bash
# Run single experiment
python train.py --experiment ant_d8_s1000

# Run with CLI overrides (Hydra-style dotlist)
python train.py --experiment ant_d8_s1000 actor_lr=0.001

# Compile check (1 epoch, no checkpoint)
python train.py --experiment ant_d8_s1000 --compile_check
```
