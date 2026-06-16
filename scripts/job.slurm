#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=scaling_crl
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=e24136160@gs.ncku.edu.tw

cd /home/u2169145/code/scaling-crl

# CUDA paths
export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\n" ":")$LD_LIBRARY_PATH

# Wandb offline (compute nodes may not have internet)
export WANDB_MODE=offline

nvidia-smi
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

.venv/bin/python train.py     --env_id ant     --critic_depth 4     --actor_depth 4     --num_epochs 5     --total_env_steps 1000000     --batch_size 128     --num_envs 64     --save_buffer 0

echo "Done!"
