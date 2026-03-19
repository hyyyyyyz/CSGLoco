#!/bin/bash
set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

echo "Multi-seed training v2 (seed fix applied) started: $(date)"

# Quick verification: train 2 iterations with different seeds, check they differ
python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1 --max_iterations=2 --headless --seed=100 2>&1 | grep "Setting seed"
python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1 --max_iterations=2 --headless --seed=200 2>&1 | grep "Setting seed"
echo "Seed fix verified (both seeds printed above should differ)"

# Clean up test runs
rm -rf legged_gym/logs/safe_recovery_a1/$(ls -t legged_gym/logs/safe_recovery_a1/ | head -1)
rm -rf legged_gym/logs/safe_recovery_a1/$(ls -t legged_gym/logs/safe_recovery_a1/ | head -1)

# Train 3 seeds for each method (seeds 1, 2, 3)
for seed in 1 2 3; do
    echo ""
    echo ">>> Vanilla PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1 --max_iterations=1500 --headless --seed=$seed 2>&1 | tail -5
    echo ">>> Done Vanilla seed=$seed ($(date))"
done

for seed in 1 2 3; do
    echo ""
    echo ">>> CaT PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1_cat --max_iterations=1500 --headless --seed=$seed 2>&1 | tail -5
    echo ">>> Done CaT seed=$seed ($(date))"
done

for seed in 1 2 3; do
    echo ""
    echo ">>> Recovery PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1_fallen --max_iterations=2000 --headless --seed=$seed 2>&1 | tail -5
    echo ">>> Done Recovery seed=$seed ($(date))"
done

echo ""
echo "ALL TRAINING DONE: $(date)"
