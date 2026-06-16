#!/bin/bash
#SBATCH --account=MST114560
#SBATCH --job-name=data_proc
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
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

# Warmup: just enough to trigger JIT compile (1 epoch, 100K steps = very fast)
warmup() {
    local ENV=$1
    local DEPTH=$2
    echo "[warmup] $ENV d=$DEPTH ..."
    CUDA_VISIBLE_DEVICES=0 $P train.py \
        --env_id $ENV --critic_depth $DEPTH --actor_depth $DEPTH \
        --num_epochs 1 --total_env_steps 100000 $COMMON \
        --wandb_group warmup > /dev/null 2>&1
    echo "[warmup] $ENV d=$DEPTH done"
}

echo "=== Phase 1: XLA Compilation Cache ==="
echo "Compiling 24 environments (this takes ~30 min)..."
warmup ant 8
warmup ant_big_maze 8
warmup ant_u_maze 8
warmup ant_hardest_maze 8
warmup arm_push_easy 8
warmup arm_push_hard 8
warmup arm_binpick_easy 8
warmup arm_binpick_hard 8
warmup arm_reach 8
warmup arm_grasp 8
warmup ant_ball 8
warmup ant_push 8
warmup ant 32
warmup ant_big_maze 32
warmup ant_u_maze 32
warmup ant_hardest_maze 32
warmup arm_push_easy 32
warmup arm_push_hard 32
warmup arm_binpick_easy 32
warmup arm_binpick_hard 32
warmup arm_reach 32
warmup arm_grasp 32
warmup ant_ball 32
warmup ant_push 32
echo "=== All XLA compiled and cached ==="

# Training
run_exp() {
    local GPU_ID=$1
    local ENV=$2
    local DEPTH=$3
    local EPOCHS=$4
    local STEPS=$5
    local LOGFILE="$LOGDIR/${ENV}_d${DEPTH}.log"
    echo "[$(date +%H:%M:%S)] $ENV d=$DEPTH GPU=$GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID $P train.py \
        --env_id $ENV --critic_depth $DEPTH --actor_depth $DEPTH \
        --num_epochs $EPOCHS --total_env_steps $STEPS $COMMON \
        --wandb_group ${ENV}_d${DEPTH} > $LOGFILE 2>&1 &
}

monitor() {
    while true; do
        sleep 1800
        echo "=== [$(date)] Status ==="
        for f in $LOGDIR/*.log; do
            [ -f "$f" ] || continue
            name=$(basename $f .log)
            if pgrep -f "env_id ${name%%_*}" > /dev/null 2>&1; then
                ep=$(grep -o "epoch [0-9]* out of" $f 2>/dev/null | tail -1 | awk '{print $2}')
                echo "  $name: ep $ep"
            else
                grep -q "Done:" $f 2>/dev/null && echo "  $name: OK" || echo "  $name: FAIL"
            fi
        done
        pgrep -c -f "train.py" > /dev/null 2>&1 || break
    done
}
monitor &
MON=$!

echo "=== Phase 2: Training ==="
run_exp 0 ant 8 100 100000000
run_exp 1 ant_big_maze 8 100 100000000
run_exp 2 ant_u_maze 8 100 100000000
run_exp 3 ant_hardest_maze 8 200 200000000
run_exp 4 arm_push_easy 8 100 100000000
run_exp 5 arm_push_hard 8 100 100000000
run_exp 6 arm_binpick_easy 8 100 100000000
run_exp 7 arm_binpick_hard 8 100 100000000
wait

run_exp 0 arm_reach 8 100 100000000
run_exp 1 arm_grasp 8 100 100000000
run_exp 2 ant_ball 8 100 100000000
run_exp 3 ant_push 8 100 100000000
run_exp 4 ant 32 100 100000000
run_exp 5 ant_big_maze 32 100 100000000
run_exp 6 ant_u_maze 32 100 100000000
run_exp 7 ant_hardest_maze 32 200 200000000
wait

run_exp 0 arm_push_easy 32 100 100000000
run_exp 1 arm_push_hard 32 100 100000000
run_exp 2 arm_binpick_easy 32 100 100000000
run_exp 3 arm_binpick_hard 32 100 100000000
run_exp 4 arm_reach 32 100 100000000
run_exp 5 arm_grasp 32 100 100000000
run_exp 6 ant_ball 32 100 100000000
run_exp 7 ant_push 32 100 100000000
wait

kill $MON 2>/dev/null
echo "=== Done ==="
