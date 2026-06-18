"""Logging setup — Wandb initialization.

All parameters come from config. No hardcoded defaults.
"""

import wandb
import wandb_osh
from wandb_osh.hooks import TriggerWandbSyncHook


def setup_wandb(cfg):
    """Initialize wandb. cfg can be dict or SimpleNamespace. Returns trigger_sync or None."""
    # Convert to dict for wandb config
    if isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        cfg_dict = vars(cfg)

    wandb_group = cfg_dict.get("wandb_group", "") or None
    exp_name = cfg_dict.get("exp_name", "")

    wandb.init(
        project=cfg_dict.get("wandb_project_name", "scaling-crl-nano4"),
        entity=cfg_dict.get("wandb_entity", "sungwayne99999-national-cheng-kung-university-co-op"),
        mode=cfg_dict.get("wandb_mode", "online"),
        group=wandb_group,
        dir=cfg_dict.get("wandb_dir", "."),
        config=cfg_dict,
        name=exp_name,
        monitor_gym=True,
        save_code=True,
        resume="allow",
        id=exp_name,
    )

    trigger_sync = None
    if cfg_dict.get("wandb_mode", "online") == 'offline':
        wandb_osh.set_log_level("ERROR")
        trigger_sync = TriggerWandbSyncHook()

    return trigger_sync
