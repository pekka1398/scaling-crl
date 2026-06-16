#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=test_infra
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_infra-%j.out
#SBATCH --error=logs/test_infra-%j.err

cd /home/u2169145/code/scaling-crl

export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\n" ":")$LD_LIBRARY_PATH
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

echo "=== Test 1: JAX cache ==="
ls -la /tmp/jax_cache/ 2>/dev/null || echo "cache dir empty (first run)"

echo "=== Test 2: GPU ==="
nvidia-smi | head -15

echo "=== Test 3: Training (5 epochs) ==="
.venv/bin/python train.py     --env_id ant     --critic_depth 4     --actor_depth 4     --num_epochs 5     --total_env_steps 1000000     --batch_size 256     --num_envs 64     --save_buffer 0     --wandb_group test_infra

echo "=== Test 4: Checkpoint ==="
ls -la runs/ | tail -5
find runs/ -name "*.pkl" | tail -5

echo "=== Test 5: Wandb ==="
ls -la wandb/ | tail -5

echo "=== Test 6: JAX cache after ==="
ls -la /tmp/jax_cache/ | head -10

echo "=== All tests done ==="
