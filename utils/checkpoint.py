"""Checkpoint management — Orbax-based.

All parameters come from config. No hardcoded defaults.
"""

import os
import glob
import pickle
from dataclasses import dataclass

import orbax.checkpoint as ocp
from etils import epath


@dataclass
class CheckpointConfig:
    """Checkpoint configuration — must be provided, no defaults."""
    save_interval_epochs: int
    max_to_keep: int
    keep_period: int


def create_checkpoint_manager(save_path: str, config: CheckpointConfig):
    """Create an Orbax CheckpointManager."""
    ckpt_dir = os.path.abspath(os.path.join(str(save_path), "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.max_to_keep,
        keep_period=config.keep_period,
    )
    return ocp.CheckpointManager(ckpt_dir, checkpointer, options)


def save_checkpoint(manager, training_state, epoch: int):
    """Save checkpoint via Orbax. Step = env_steps for resume tracking."""
    step = int(training_state.env_steps)
    ckpt = {
        'alpha_params': training_state.alpha_state.params,
        'actor_params': training_state.actor_state.params,
        'critic_params': training_state.critic_state.params,
        'epoch': epoch,
        'env_steps': step,
        'gradient_steps': int(training_state.gradient_steps),
    }
    manager.save(step, ckpt, force=True)
    manager.wait_until_finished()


def restore_checkpoint(manager):
    """Restore latest Orbax checkpoint. Returns (ckpt_dict, step) or (None, 0)."""
    step = manager.latest_step()
    if step is None:
        return None, 0
    return manager.restore(step), step


def load_legacy_checkpoint(path: str):
    """Load old pickle-format checkpoint. Returns None if corrupt."""
    try:
        with epath.Path(path).open('rb') as fin:
            buf = fin.read()
        return pickle.loads(buf)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"WARNING: Corrupt checkpoint {path}: {e}", flush=True)
        return None


def find_legacy_checkpoint(save_path: str) -> str:
    """Find latest step_*.pkl or final.pkl (old format). Returns path or ''."""
    ckpts = sorted(glob.glob(f"{save_path}/step_*.pkl"),
                   key=lambda f: int(f.rsplit('_', 1)[-1].rsplit('.', 1)[0]),
                   reverse=True)
    for ckpt in ckpts:
        if os.path.getsize(ckpt) > 100:
            return ckpt
    final = os.path.join(str(save_path), "final.pkl")
    if os.path.exists(final) and os.path.getsize(final) > 100:
        return final
    return ""
