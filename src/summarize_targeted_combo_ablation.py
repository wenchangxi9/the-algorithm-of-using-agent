from pathlib import Path

import pandas as pd


ROOT = Path("/data6/wenchangxi/community_note/communitynotes_mf_calibrated_pipeline")
TABLE_DIR = ROOT / "artifacts/comparison_tables"
OUT = TABLE_DIR / "feature_ablation_targeted_combo_combined_view_20260513.csv"

KEEP = [
    "vote_only",
    "vote_plus_quality",
    "vote_plus_cluster_quality",
    "vote_plus_quality_plus_cluster_quality",
    "vote_plus_confidence_plus_quality_plus_cluster_quality",
    "vote_plus_uncertainty_plus_quality_plus_cluster_quality",
    "summary_all",
    "hybrid_all",
]

rows = []
for path in sorted(TABLE_DIR.glob("feature_ablation_targeted_combo_n*_summary_20260513.csv")):
    df = pd.read_csv(path)
    df = df[df["method"].isin(KEEP)].copy()
    rows.append(df)

combined = pd.concat(rows, ignore_index=True)
combined["accuracy_pct"] = combined["accuracy"] * 100
combined["coverage_pct"] = combined["coverage"] * 100
combined["accuracy_se_pct"] = (
    (combined["accuracy_ci95_high"] - combined["accuracy_ci95_low"]) / (2 * 1.96) * 100
)
combined = combined.sort_values(["metric", "coverage_target", "agent_count", "accuracy"], ascending=[True, True, True, False])
combined.to_csv(OUT, index=False, encoding="utf-8-sig")

show = combined[
    [
        "agent_count",
        "metric",
        "coverage_target",
        "method",
        "family",
        "accuracy_pct",
        "accuracy_se_pct",
        "coverage_pct",
        "n_features",
    ]
]
print(show.to_string(index=False))
print(f"Saved to {OUT}")
