from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path("/data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_20260513/groundtruth_reason_clusters_20260513")
summary = pd.read_csv(BASE / "reason_cluster_summary.csv")
notes = pd.read_csv(BASE / "helpful_nh_notes_with_reason_clusters.csv", dtype={"noteId": str}, low_memory=False)

print("Counts:")
print(notes[["true_label_text", "reason_cluster"]].value_counts().sort_index())

for _, r in summary.sort_values(["label", "cluster"]).iterrows():
    print()
    print("=" * 100)
    print(f"{r['label']} cluster {int(r['cluster'])} | n={int(r['n'])} | share={float(r['share']):.3f}")
    print("Helpful reasons:", r["top_helpful_reasons"])
    print("Not-helpful reasons:", r["top_not_helpful_reasons"])
    print("Meta tags:", r["top_note_meta_tags"])
    print("Examples:")
    examples = json.loads(r["example_json"])
    for e in examples[:5]:
        print(f"- noteId={e['noteId']}")
        print(f"  POST: {e['post'][:400]}")
        print(f"  NOTE: {e['note'][:520]}")
