#!/usr/bin/env bash
set -euo pipefail

cd /data6/wenchangxi/community_note

OUTDIR="analysis/representative_20k_sample_20260511/post_fetch"
mkdir -p "$OUTDIR"

if [[ -z "${X_BEARER_TOKEN:-}" && -z "${TWITTER_BEARER_TOKEN:-}" ]]; then
  echo "Missing X_BEARER_TOKEN or TWITTER_BEARER_TOKEN." >&2
  echo "Export an official X API bearer token first, then rerun this script." >&2
  exit 2
fi

nohup /data6/wenchangxi/.conda/envs/DL/bin/python src/fetch_sample_posts.py \
  --provider auto \
  --workers 8 \
  --batch-size 100 \
  --timeout 30 \
  --preflight-size 5 \
  --resume \
  --outdir "$OUTDIR" \
  > "$OUTDIR/stdout.log" \
  2> "$OUTDIR/stderr.log" &

echo "$!" > "$OUTDIR/post_fetch.pid"
echo "Started post fetch PID $(cat "$OUTDIR/post_fetch.pid")"
echo "Progress: $OUTDIR/post_fetch_progress.log"
