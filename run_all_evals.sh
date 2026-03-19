#!/bin/bash
set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

OUTDIR="safe_recovery_eval_v2"
mkdir -p $OUTDIR

echo "=========================================="
echo "SafeRecovery Full Evaluation Suite"
echo "Started: $(date)"
echo "=========================================="

# === EXPERIMENT 1: Multi-seed standard evaluation ===
echo ""
echo ">>> EXPERIMENT 1: Multi-seed standard evaluation (80-150N, 10k steps x 128 envs)"

# The Mar18 runs correspond to seeds 1,2,3 from batch_train_3seed_v2.sh
# Vanilla: Mar18_16-16-27_ (s1), Mar18_16-53-02_ (s2), Mar18_17-29-18_ (s3)
# CaT:     Mar18_18-05-33_ (s1), Mar18_18-42-10_ (s2), Mar18_19-18-36_ (s3)
# Recovery: Mar18_19-54-41_ (s1), Mar18_20-44-07_ (s2), Mar18_21-33-50_ (s3)

VANILLA_RUNS="Mar18_16-16-27_ Mar18_16-53-02_ Mar18_17-29-18_"
CAT_RUNS="Mar18_18-05-33_ Mar18_18-42-10_ Mar18_19-18-36_"
RECOVERY_RUNS="Mar18_19-54-41_ Mar18_20-44-07_ Mar18_21-33-50_"

for seed_idx in 1 2 3; do
    VANILLA_RUN=$(echo $VANILLA_RUNS | cut -d' ' -f$seed_idx)
    CAT_RUN=$(echo $CAT_RUNS | cut -d' ' -f$seed_idx)
    RECOVERY_RUN=$(echo $RECOVERY_RUNS | cut -d' ' -f$seed_idx)

    echo ""
    echo "--- Seed $seed_idx ---"

    echo "  Evaluating Vanilla seed=$seed_idx ($VANILLA_RUN)..."
    python legged_gym/legged_gym/scripts/eval_saferecovery.py \
        --task=safe_recovery_a1 --load_run=$VANILLA_RUN \
        --num_eval_steps=10000 --num_envs=128 \
        --force_min=80 --force_max=150 \
        --output=$OUTDIR/vanilla_seed${seed_idx}.json \
        2>&1 | tail -3

    echo "  Evaluating CaT seed=$seed_idx ($CAT_RUN)..."
    python legged_gym/legged_gym/scripts/eval_saferecovery.py \
        --task=safe_recovery_a1_cat --load_run=$CAT_RUN \
        --num_eval_steps=10000 --num_envs=128 \
        --force_min=80 --force_max=150 \
        --output=$OUTDIR/cat_seed${seed_idx}.json \
        2>&1 | tail -3

    echo "  Evaluating Recovery seed=$seed_idx ($RECOVERY_RUN)..."
    python legged_gym/legged_gym/scripts/eval_saferecovery.py \
        --task=safe_recovery_a1_fallen --load_run=$RECOVERY_RUN \
        --num_eval_steps=10000 --num_envs=128 \
        --force_min=80 --force_max=150 \
        --output=$OUTDIR/recovery_seed${seed_idx}.json \
        2>&1 | tail -3
done

echo ""
echo ">>> EXPERIMENT 1 DONE: $(date)"

# === EXPERIMENT 2: Controlled fallen-start evaluation ===
echo ""
echo ">>> EXPERIMENT 2: Controlled fallen-start evaluation (all envs start fallen)"
echo "This tests recovery capability with large sample counts."

for seed_idx in 1 2 3; do
    VANILLA_RUN=$(echo $VANILLA_RUNS | cut -d' ' -f$seed_idx)
    CAT_RUN=$(echo $CAT_RUNS | cut -d' ' -f$seed_idx)
    RECOVERY_RUN=$(echo $RECOVERY_RUNS | cut -d' ' -f$seed_idx)

    echo "  Fallen-start Vanilla seed=$seed_idx..."
    python legged_gym/legged_gym/scripts/eval_saferecovery.py \
        --task=safe_recovery_a1 --load_run=$VANILLA_RUN \
        --num_eval_steps=5000 --num_envs=128 \
        --force_min=0 --force_max=0 \
        --fallen_start \
        --output=$OUTDIR/vanilla_fallen_seed${seed_idx}.json \
        2>&1 | tail -3

    echo "  Fallen-start CaT seed=$seed_idx..."
    python legged_gym/legged_gym/scripts/eval_saferecovery.py \
        --task=safe_recovery_a1_cat --load_run=$CAT_RUN \
        --num_eval_steps=5000 --num_envs=128 \
        --force_min=0 --force_max=0 \
        --fallen_start \
        --output=$OUTDIR/cat_fallen_seed${seed_idx}.json \
        2>&1 | tail -3

    echo "  Fallen-start Recovery seed=$seed_idx..."
    python legged_gym/legged_gym/scripts/eval_saferecovery.py \
        --task=safe_recovery_a1_fallen --load_run=$RECOVERY_RUN \
        --num_eval_steps=5000 --num_envs=128 \
        --force_min=0 --force_max=0 \
        --fallen_start \
        --output=$OUTDIR/recovery_fallen_seed${seed_idx}.json \
        2>&1 | tail -3
done

echo ""
echo ">>> EXPERIMENT 2 DONE: $(date)"

# === EXPERIMENT 3: CaT native vs fair protocol ===
echo ""
echo ">>> EXPERIMENT 3: CaT native vs fair dual-protocol evaluation"

for seed_idx in 1 2 3; do
    CAT_RUN=$(echo $CAT_RUNS | cut -d' ' -f$seed_idx)

    echo "  CaT native (termination ON) seed=$seed_idx..."
    python legged_gym/legged_gym/scripts/eval_saferecovery.py \
        --task=safe_recovery_a1_cat --load_run=$CAT_RUN \
        --num_eval_steps=10000 --num_envs=128 \
        --force_min=80 --force_max=150 \
        --native_termination \
        --output=$OUTDIR/cat_native_seed${seed_idx}.json \
        2>&1 | tail -3
done

echo ""
echo ">>> EXPERIMENT 3 DONE: $(date)"

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE: $(date)"
echo "Results in: $OUTDIR/"
echo "=========================================="
ls -la $OUTDIR/
