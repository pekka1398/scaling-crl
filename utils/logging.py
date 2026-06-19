"""Logging setup — Wandb initialization with resume support.
"""

import wandb
from omegaconf import DictConfig, OmegaConf


def setup_wandb(cfg, resume_id=None):
    """Initialize wandb. Returns (trigger_sync, run).

    Args:
        cfg: OmegaConf DictConfig, dict, or SimpleNamespace
        resume_id: wandb run ID to resume (from wandb_id.txt)
    """
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        cfg_dict = vars(cfg)

    wandb_group = cfg_dict.get("wandb_group", "") or None
    exp_name = cfg_dict.get("exp_name", "")

    init_kwargs = dict(
        project=cfg_dict.get("wandb_project_name", "scaling-crl-v2"),
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

    return None, run
