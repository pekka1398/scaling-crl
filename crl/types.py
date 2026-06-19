"""Shared types for CRL — no circular dependencies."""

from typing import NamedTuple
import flax
import jax.numpy as jnp
from flax.training.train_state import TrainState


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
