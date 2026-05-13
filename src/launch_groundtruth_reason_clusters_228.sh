#!/usr/bin/env bash
set -euo pipefail

cd /data6/wenchangxi/community_note
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL

OUT="/data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_20260513/groundtruth_reason_clusters_20260513"
mkdir -p "$OUT"
nohup python src/cluster_groundtruth_reasons_228.py > "$OUT/stdout.log" 2> "$OUT/stderr.log" &
echo $! > "$OUT/job.pid"
echo "started pid=$(cat "$OUT/job.pid") outdir=$OUT"
