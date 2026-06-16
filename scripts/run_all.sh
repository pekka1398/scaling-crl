#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=run_all
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH --output=logs/run_all-%j.out
#SBATCH --error=logs/run_all-%j.err

cd /home/u2169145/code/scaling-crl
export LD_LIBRARY_PATH=$(find .venv -path "*/nvidia/*/lib" -type d | tr "\n" ":")$LD_LIBRARY_PATH
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

P=.venv/bin/python
COMMON="--num_epochs NUM_EPOCHS --total_env_steps TOTAL_STEPS --batch_size 512 --num_envs 512 --save_buffer 0 --actor_skip_connections 4 --critic_skip_connections 4"

run_task() {
    local ENV_ID=$1
    local DEPTH=$2
    local EPOCHS=$3
    local STEPS=$4
    echo "=========================================="
    echo "Starting: $ENV_ID (depth=$DEPTH, epochs=$EPOCHS, steps=$STEPS)"
    echo "Time: $(date)"
    echo "=========================================="
    $P train.py --env_id $ENV_ID --critic_depth $DEPTH --actor_depth $DEPTH --num_epochs $EPOCHS --total_env_steps $STEPS --batch_size 512 --num_envs 512 --save_buffer 0 --actor_skip_connections 4 --critic_skip_connections 4 --wandb_group $ENV_ID 2>&1 | tee logs/${ENV_ID}.log
    echo "Finished: $ENV_ID at $(date)"
}

# Paper best settings
run_task ant_big_maze     32 100 100000000
run_task ant_u_maze       64 100 100000000
run_task ant_hardest_maze 32 200 200000000
run_task arm_push_easy    32 100 100000000
run_task arm_binpick_easy 32 100 100000000

echo "All done!"
