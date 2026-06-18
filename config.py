"""Single source of truth for experiment configuration.

Every place that names/logs/checkpoints an experiment MUST use
Experiment.exp_name — no ad-hoc string assembly allowed.
"""

from dataclasses import dataclass, field


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
    resume_from: str = "auto"

    # === Wandb / logging ===
    wandb_mode: str = "online"
    wandb_group: str = ""          # Empty = use exp_name
    wandb_project_name: str = "scaling-crl-nano4"
    capture_vis: bool = False

    def get_wandb_group(self):
        return self.wandb_group if self.wandb_group else self.exp_name

    def to_train_args(self, compile_check=False):
        """Generate train.py CLI args from this experiment.

        If compile_check=True, override training budget to 1 epoch / 1M steps
        and disable checkpoint/track/vis.
        """
        args = [
            "--exp_name", self.exp_name if not compile_check else "compile_" + self.exp_name,
            "--env_id", self.env_id,
            "--seed", str(self.seed),
            "--actor_depth", str(self.depth),
            "--critic_depth", str(self.depth),
            "--actor_network_width", str(self.actor_network_width),
            "--critic_network_width", str(self.critic_network_width),
            "--batch_size", str(self.batch_size),
            "--num_envs", str(self.num_envs),
            "--actor_lr", str(self.actor_lr),
            "--critic_lr", str(self.critic_lr),
            "--alpha_lr", str(self.alpha_lr),
            "--unroll_length", str(self.unroll_length),
            "--max_replay_size", str(self.max_replay_size),
            "--entropy_param", str(self.entropy_param),
            "--save_buffer", str(self.save_buffer),
            "--no-capture-vis" if not self.capture_vis else "--capture-vis",
            "--wandb_mode", self.wandb_mode if not compile_check else "offline",
            "--wandb_group", self.get_wandb_group(),
            "--wandb_project_name", self.wandb_project_name,
        ]

        if compile_check:
            args += [
                "--num_epochs", "1",
                "--total_env_steps", "1000000",
                "--no-checkpoint",
                "--no-track",
            ]
        else:
            args += [
                "--num_epochs", str(self.num_epochs),
                "--total_env_steps", str(self.total_env_steps),
                "--resume_from", self.resume_from,
            ]

        return args
