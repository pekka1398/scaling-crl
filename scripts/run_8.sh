#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=data_proc
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/home/u2169145/code/scaling-crl/logs/batch-%j.out
#SBATCH --error=/home/u2169145/code/scaling-crl/logs/batch-%j.err

cd /home/u2169145/code/scaling-crl

NVIDIA_LIBS=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\n" ":")
export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH}
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

P=.venv/bin/python
LOGDIR=logs/exp
mkdir -p $LOGDIR
COMMON="--actor_skip_connections 4 --critic_skip_connections 4 --batch_size 512 --num_envs 512 --save_buffer 0"

echo "=== Starting 8 experiments (1 CPU, 8 GPU, billing=1) ==="

CUDA_VISIBLE_DEVICES=0 $P train.py --env_id ant --critic_depth 8 --actor_depth 8 --num_epochs 100 --total_env_steps 100000000 $COMMON --wandb_group ant_d8 > $LOGDIR/ant_d8.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $P train.py --env_id ant_big_maze --critic_depth 8 --actor_depth 8 --num_epochs 100 --total_env_steps 100000000 $COMMON --wandb_group ant_big_maze_d8 > $LOGDIR/ant_big_maze_d8.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $P train.py --env_id ant_u_maze --critic_depth 8 --actor_depth 8 --num_epochs 100 --total_env_steps 100000000 $COMMON --wandb_group ant_u_maze_d8 > $LOGDIR/ant_u_maze_d8.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $P train.py --env_id ant_hardest_maze --critic_depth 8 --actor_depth 8 --num_epochs 200 --total_env_steps 200000000 $COMMON --wandb_group ant_hardest_maze_d8 > $LOGDIR/ant_hardest_maze_d8.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $P train.py --env_id arm_push_easy --critic_depth 8 --actor_depth 8 --num_epochs 100 --total_env_steps 100000000 $COMMON --wandb_group arm_push_easy_d8 > $LOGDIR/arm_push_easy_d8.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $P train.py --env_id arm_push_hard --critic_depth 8 --actor_depth 8 --num_epochs 100 --total_env_steps 100000000 $COMMON --wandb_group arm_push_hard_d8 > $LOGDIR/arm_push_hard_d8.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $P train.py --env_id arm_binpick_easy --critic_depth 8 --actor_depth 8 --num_epochs 100 --total_env_steps 100000000 $COMMON --wandb_group arm_binpick_easy_d8 > $LOGDIR/arm_binpick_easy_d8.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $P train.py --env_id arm_binpick_hard --critic_depth 8 --actor_depth 8 --num_epochs 100 --total_env_steps 100000000 $COMMON --wandb_group arm_binpick_hard_d8 > $LOGDIR/arm_binpick_hard_d8.log 2>&1 &

wait
echo "=== All 8 done ==="
