import os
import jax
import jax.numpy as jnp
import time

proc_id = os.getpid()
print(f"[PID {proc_id}] JAX devices: {jax.devices()}")

# 佔一些 GPU memory
x = jnp.ones((2000, 2000))
y = jnp.dot(x, x)
y.block_until_ready()
print(f"[PID {proc_id}] GPU matmul done, holding memory...")

# 持久佔用 30 秒
time.sleep(30)
print(f"[PID {proc_id}] Done")
