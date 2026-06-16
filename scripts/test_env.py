import jax, flax, brax, mujoco
import flax.linen as nn
import jax.numpy as jnp

print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
print(f"Flax: {flax.__version__}, Brax: {brax.__version__}, MuJoCo: {mujoco.__version__}")

from brax import envs
from envs.ant import Ant
env = Ant(backend="spring", exclude_current_positions_from_observation=False, terminate_when_unhealthy=True)
print(f"Ant env OK, action_size: {env.action_size}")

# Quick GPU matmul
x = jnp.ones((500, 500))
y = jnp.dot(x, x)
print(f"GPU matmul OK: {y.shape}")

print("=== All passed ===")
