"""Logging setup — Wandb initialization.

All parameters come from config. No hardcoded defaults.
"""

import wandb
import wandb_osh
from wandb_osh.hooks import TriggerWandbSyncHook


def setup_wandb(args):
    """Initialize wandb. Returns trigger_sync function (or None if offline)."""
    if not args.wandb_group:
        args.wandb_group = None

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        group=args.wandb_group,
        dir=args.wandb_dir,
        config=vars(args),
        name=args.exp_name or args.env_id,
        monitor_gym=True,
        save_code=True,
        resume="allow",
        id=args.exp_name or args.env_id,
    )

    trigger_sync = None
    if args.wandb_mode == 'offline':
        wandb_osh.set_log_level("ERROR")
        trigger_sync = TriggerWandbSyncHook()

    return trigger_sync
