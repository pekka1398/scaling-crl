"""Single source of truth for experiment configuration.

Every place that names/logs/checkpoints an experiment MUST use
Experiment.exp_name — no ad-hoc string assembly allowed.
"""

from dataclasses import dataclass


@dataclass
class Experiment:
    # === Identity ===
    exp_name: str                  # Unique name — sole source for all log/ckpt/wandb naming
    env_id: str                    # Environment id (passed to train.py --env_id)
    depth: int                     # Network depth (both actor and critic)
    seed: int                      # Random seed

    # === Training budget ===
    num_epochs: int = 100
    total_env_steps: int = 100_000_000

    # === Network architecture ===
    actor_skip_connections: int = 4
    critic_skip_connections: int = 4
    actor_network_width: int = 256
    critic_network_width: int = 256
    batch_size: int = 512
    num_envs: int = 512

    # === Occasionally tuned ===
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    unroll_length: int = 62
    max_replay_size: int = 10000
    entropy_param: float = 0.5
    save_buffer: int = 0
    resume_from: str = ""
