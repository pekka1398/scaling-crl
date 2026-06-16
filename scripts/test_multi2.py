import os
import jax
import jax.numpy as jnp
import time

proc_id = os.getpid()
print(f"[PID {proc_id}] Starting, devices: {jax.devices()}", flush=True)

# 佔 GPU memory
x = jnp.ones((3000, 3000))
y = jnp.dot(x, x)
y.block_until_ready()
print(f"[PID {proc_id}] Allocated memory, sleeping 15s...", flush=True)

time.sleep(15)
print(f"[PID {proc_id}] Done", flush=True)
