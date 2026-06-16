#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=run_8gpu
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/run_8gpu-%j.out
#SBATCH --error=logs/run_8gpu-%j.err

cd /home/u2169145/code/scaling-crl

export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\n" ":")$LD_LIBRARY_PATH
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

P=.venv/bin/python
COMMON="--num_epochs 100 --total_env_steps 100000000 --batch_size 512 --num_envs 512 --save_buffer 0 --actor_skip_connections 4 --critic_skip_connections 4"

# 8 tasks on 8 GPUs
$P train.py --env_id ant --critic_depth 8 --actor_depth 8 $COMMON --wandb_group ant &
$P train.py --env_id ant_big_maze --critic_depth 8 --actor_depth 8 $COMMON --wandb_group ant_big_maze &
$P train.py --env_id ant_u_maze --critic_depth 8 --actor_depth 8 $COMMON --wandb_group ant_u_maze &
$P train.py --env_id ant_hardest_maze --critic_depth 8 --actor_depth 8 $COMMON --wandb_group ant_hardest_maze &
$P train.py --env_id arm_push_easy --critic_depth 8 --actor_depth 8 $COMMON --wandb_group arm_push_easy &
$P train.py --env_id arm_binpick_easy --critic_depth 8 --actor_depth 8 $COMMON --wandb_group arm_binpick_easy &
$P train.py --env_id arm_reach --critic_depth 8 --actor_depth 8 $COMMON --wandb_group arm_reach &
$P train.py --env_id arm_push_hard --critic_depth 8 --actor_depth 8 $COMMON --wandb_group arm_push_hard &

echo "Launched 8 experiments on 8 GPUs"
wait
echo "All done!"
