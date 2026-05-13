#!/usr/bin/env bash
set -euo pipefail

cd /data6/wenchangxi/community_note
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL

OUTDIR="/data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_20260513"
mkdir -p "$OUTDIR"

nohup python src/run_16agent_binaryrating_balanced_228.py \
  --sample-csv /data6/wenchangxi/community_note/analysis/llm_16agent_rawrating_balanced_1to1to1_promptv3_20260513/pilot_notes.csv \
  --cluster-summary-csv /data6/wenchangxi/community_note/analysis/official_mfcore_rater_clustering_20260510_201855/k_search_2_32_step1/cluster_summary_k16.csv \
  --outdir "$OUTDIR" \
  --max-notes 0 \
  --selection-mode sample_order \
  --model gpt-5.4-nano \
  --base-url https://api.gpt.ge/v1 \
  --temperature 0.2 \
  --max-tokens 280 \
  --concurrency 32 \
  --progress-every 50 \
  --save-every 200 \
  --timeout 90 \
  --max-retries 3 \
  > "$OUTDIR/stdout.log" 2> "$OUTDIR/stderr.log" &

echo $! > "$OUTDIR/job.pid"
echo "started pid=$(cat "$OUTDIR/job.pid") outdir=$OUTDIR"
