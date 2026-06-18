"""Shared types for CRL — no circular dependencies."""

from typing import NamedTuple
import jax.numpy as jnp


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()
