#!/usr/bin/env bash
set -euo pipefail

ROOT="/data6/wenchangxi/community_note/communitynotes_mf_calibrated_pipeline"
cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL

mkdir -p artifacts/logs artifacts/comparison_tables

for N in 12 24 36 48; do
  LOG="artifacts/logs/feature_ablation_targeted_combo_n${N}_20260513"
  nohup python src/run_feature_ablation.py \
    --repo-root "$ROOT" \
    --agent-counts "$N" \
    --model-tag gpt54nano \
    --date-tag 20260507 \
    --run-tag run1 \
    --folds 5 \
    --inner-folds 4 \
    --seed 42 \
    --coverage-targets 0.65 \
    --bootstrap-repeats 200 \
    --output-root "artifacts/feature_ablation_targeted_combo_parallel_20260513" \
    --comparison-summary-csv "artifacts/comparison_tables/feature_ablation_targeted_combo_n${N}_summary_20260513.csv" \
    --comparison-delta-csv "artifacts/comparison_tables/feature_ablation_targeted_combo_n${N}_deltas_20260513.csv" \
    > "${LOG}.stdout.log" 2> "${LOG}.stderr.log" &
  echo $! > "${LOG}.pid"
  echo "started N=$N pid=$(cat "${LOG}.pid")"
done
