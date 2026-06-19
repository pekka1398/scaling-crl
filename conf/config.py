"""Structured config for Scaling-CRL experiments.

Every field is required (MISSING). No defaults, no fallbacks.
Hydra raises MissingMandatoryValue if any field is not provided.
"""

from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    # --- Identity ---
    exp_name: str = MISSING
    env_id: str = MISSING
    seed: int = MISSING

    # --- Training ---
    num_epochs: int = MISSING
    total_env_steps: int = MISSING

    # --- Network ---
    depth: int = MISSING
    actor_network_width: int = MISSING
    critic_network_width: int = MISSING

    # --- Optimization ---
    batch_size: int = MISSING
    num_envs: int = MISSING
    actor_lr: float = MISSING
    critic_lr: float = MISSING
    alpha_lr: float = MISSING

    # --- RL ---
    unroll_length: int = MISSING
    max_replay_size: int = MISSING
    min_replay_size: int = MISSING
    entropy_param: float = MISSING
    gamma: float = MISSING
    logsumexp_penalty_coeff: float = MISSING
    disable_entropy: int = MISSING
    use_relu: int = MISSING
    num_episodes_per_env: int = MISSING
    training_steps_multiplier: int = MISSING
    use_all_batches: int = MISSING
    num_sgd_batches_per_training_step: int = MISSING

    # --- Evaluation ---
    eval_actor: int = MISSING
    expl_actor: int = MISSING
    num_eval_envs: int = MISSING
    episode_length: int = MISSING
    eval_env_id: Optional[str] = None  # defaults to env_id at runtime

    # --- Checkpointing ---
    resume_from: bool = MISSING
    checkpoint: bool = MISSING
    checkpoint_save_interval_epochs: int = MISSING
    checkpoint_max_to_keep: int = MISSING
    checkpoint_keep_period: int = MISSING
    save_buffer: int = MISSING

    # --- Rendering ---
    capture_vis: bool = MISSING
    num_render: int = MISSING

    # --- Artifacts ---
    artifact_upload_interval_epochs: int = MISSING

    # --- Wandb ---
    track: bool = MISSING
    wandb_mode: str = MISSING
    wandb_group: str = MISSING
    wandb_project_name: str = MISSING
    wandb_entity: str = MISSING
    wandb_dir: str = MISSING
