#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=multi_task
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/multi-%j.out
#SBATCH --error=logs/multi-%j.err

cd /home/u2169145/code/scaling-crl

export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\n" ":")$LD_LIBRARY_PATH
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

nvidia-smi
echo "Node: $(hostname)"

P=.venv/bin/python
COMMON="--num_epochs 100 --total_env_steps 100000000 --batch_size 512 --num_envs 512 --save_buffer 0"
SKIP="--actor_skip_connections 4 --critic_skip_connections 4"

$P train.py --env_id ant --critic_depth 8 --actor_depth 8 $SKIP $COMMON --wandb_group ant &
$P train.py --env_id ant_big_maze --critic_depth 8 --actor_depth 8 $SKIP $COMMON --wandb_group ant_big_maze &
$P train.py --env_id ant_u_maze --critic_depth 8 --actor_depth 8 $SKIP $COMMON --wandb_group ant_u_maze &
$P train.py --env_id arm_push_easy --critic_depth 8 --actor_depth 8 $SKIP $COMMON --wandb_group arm_push_easy &
$P train.py --env_id arm_binpick_easy --critic_depth 8 --actor_depth 8 $SKIP $COMMON --wandb_group arm_binpick_easy &

echo "Launched 5 experiments"
wait
echo "All done!"
