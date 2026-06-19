"""Logging setup — Wandb initialization with resume support.
"""

import wandb
import wandb_osh
from wandb_osh.hooks import TriggerWandbSyncHook


def setup_wandb(cfg, resume_id=None):
    """Initialize wandb. Returns (trigger_sync, run).

    Args:
        cfg: config dict or SimpleNamespace
        resume_id: wandb run ID to resume (from wandb_id.txt)
    """
    if isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        cfg_dict = vars(cfg)

    wandb_group = cfg_dict.get("wandb_group", "") or None
    exp_name = cfg_dict.get("exp_name", "")

    init_kwargs = dict(
        project=cfg_dict.get("wandb_project_name", "scaling-crl-nano4"),
        entity=cfg_dict.get("wandb_entity", "sungwayne99999-national-cheng-kung-university-co-op"),
        mode=cfg_dict.get("wandb_mode", "online"),
        group=wandb_group,
        dir=cfg_dict.get("wandb_dir", "."),
        config=cfg_dict,
        name=exp_name,
        monitor_gym=True,
        save_code=True,
    )

    if resume_id:
        init_kwargs.update(id=resume_id, resume="allow")

    run = wandb.init(**init_kwargs)

    trigger_sync = None
    if cfg_dict.get("wandb_mode", "online") == 'offline':
        wandb_osh.set_log_level("ERROR")
        trigger_sync = TriggerWandbSyncHook()

    return trigger_sync, run
