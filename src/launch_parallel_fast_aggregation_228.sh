#!/usr/bin/env bash
set -euo pipefail

cd /data6/wenchangxi/community_note
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL

RUN_DIR="/data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_20260513"
OUT_DIR="$RUN_DIR/tristate_aggregator_parallel_fast_20260513"
mkdir -p "$OUT_DIR"

nohup python src/explore_binary_agent_tristate_parallel_fast_228.py \
  --run-dir "$RUN_DIR" \
  --jobs 24 \
  > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log" &

echo $! > "$OUT_DIR/job.pid"
echo "started pid=$(cat "$OUT_DIR/job.pid") outdir=$OUT_DIR"
