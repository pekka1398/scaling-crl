#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=scaling_depth
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/parallel-%j.out
#SBATCH --error=logs/parallel-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e24136160@gs.ncku.edu.tw

cd /home/u2169145/code/scaling-crl

export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\n" ":")$LD_LIBRARY_PATH
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

nvidia-smi
echo "Node: $(hostname)"

# 3 depths in parallel
.venv/bin/python train.py --env_id ant_big_maze --critic_depth 4 --actor_depth 4 --num_epochs 100 --total_env_steps 100000000 --batch_size 512 --num_envs 512 --save_buffer 0 --wandb_group depth4 &
PID1=$!

.venv/bin/python train.py --env_id ant_big_maze --critic_depth 8 --actor_depth 8 --actor_skip_connections 4 --critic_skip_connections 4 --num_epochs 100 --total_env_steps 100000000 --batch_size 512 --num_envs 512 --save_buffer 0 --wandb_group depth8 &
PID2=$!

.venv/bin/python train.py --env_id ant_big_maze --critic_depth 16 --actor_depth 16 --actor_skip_connections 4 --critic_skip_connections 4 --num_epochs 100 --total_env_steps 100000000 --batch_size 512 --num_envs 512 --save_buffer 0 --wandb_group depth16 &
PID3=$!

wait $PID1 $PID2 $PID3
echo "All 3 done!"
