# Scaling-CRL Infrastructure Status
Updated: 2026-06-17 11:22

## Current Jobs

| Job | Partition | Node | Status | Content |
|-----|-----------|------|--------|---------|
| 109547 | 8gpus | 25a-hgpn083 | RUNNING | arm_push_easy_d32 ~ ant_push_d32 (depth=32 batch) |

## Completed Experiments (from previous runs)

### depth=8 (12/12 completed)
- ant_d8 ✅, ant_big_maze_d8 ✅, ant_u_maze_d8 ✅, ant_hardest_maze_d8 ✅
- ant_ball_d8 ✅, ant_push_d8 ✅
- arm_push_easy_d8 ⏳ (23/100, killed), arm_push_hard_d8 ⏳ (24/100, killed)
- arm_binpick_easy_d8 ⏳ (19/100, killed), arm_binpick_hard_d8 ⏳ (20/100, killed)
- arm_reach_d8 ⏳ (76/100, killed), arm_grasp_d8 ⏳ (35/100, killed)

### depth=32 (in progress, Job 109547)
- ant_d32 ✅, ant_big_maze_d32 ✅, ant_u_maze_d32 ✅, ant_hardest_maze_d32 ✅
- ant_ball_d32 ✅
- ant_push_d32 ⏳ (83/100)
- arm_*_d32 ⏳ (5-15/100)

## Pending Experiments (humanoid_experiments.yaml)

### Humanoid experiments (18 new)
- humanoid: d8, d32, d64 × 2 seeds = 6
- humanoid_u_maze: d8, d32, d64 × 2 seeds = 6
- humanoid_big_maze: d8, d32, d64 × 2 seeds = 6

### Resume arm_d8 (6 experiments)
- arm_push_easy_d8, arm_push_hard_d8, arm_binpick_easy_d8
- arm_binpick_hard_d8, arm_reach_d8, arm_grasp_d8
- All have checkpoints in runs/*_20260617-010023/

## Job Plan (3 jobs × 8 GPU)

| Job | Experiments | GPU |
|-----|-------------|-----|
| A | humanoid_d8×2 + humanoid_d32×2 + humanoid_d64×2 + humanoid_u_maze_d8×2 | 8 |
| B | humanoid_u_maze_d32×2 + humanoid_u_maze_d64×2 + humanoid_big_maze_d8×2 + humanoid_big_maze_d32×2 | 8 |
| C | humanoid_big_maze_d64×2 + resume arm_d8×6 | 8 |

Total: 24 experiments, 3 jobs, 24 GPU
+ Job 109547 (8 GPU) = 32 GPU (account limit)

## Submit Command

```bash
cd ~/code/scaling-crl/infra
python3 launcher.py --type humanoid --yaml humanoid_experiments.yaml --mem 200G
```

## Key Settings

- 1 CPU per job
- 8 GPU per job
- 200G memory per job
- Compile check with --no-capture-vis
- Training with --no-capture-vis
- JAX cache: /home/u2169145/.cache/jax (shared across jobs)
- Log naming: {env}_d{depth}_s{seed}.log

## Account Usage

- Account: MST114560
- User: u2169145
- Max GPU: 32
- FairShare: very high (almost no usage)

## Notes

- Different seeds share JAX compile cache (same model architecture)
- Resume experiments hit existing cache (compile ~0s)
- Humanoid obs_dim=268, action=17 (much larger than ant's 31/8)
- depth=64 compile takes ~10-15 min (260 configs autotuning)
