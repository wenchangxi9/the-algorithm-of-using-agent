import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data6/wenchangxi/community_note/src")
from run_improved_aggregation_2000 import build_features  # noqa: E402


RUN = Path("/data6/wenchangxi/community_note/analysis/llm_16agent_tristate_2000_20260512")
OUT = RUN / "improved_aggregation_20260512" / "feature_signal_diagnostics.json"


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


df, feature_sets = build_features(RUN)
y_status = df["true_label_3way"].to_numpy(int)
y_resolved = (y_status != 1).astype(int)

out = {"n": int(len(df)), "resolved_rate": float(y_resolved.mean())}

simple_scores = {
    "resolved_vote_share": df["resolved_vote_share"].to_numpy(float),
    "one_minus_nmr_share": (1 - df["share_nmr"]).to_numpy(float),
    "top_label_share": df["top_label_share"].to_numpy(float),
    "quality_mean": df["quality_mean"].to_numpy(float),
    "failure_mean": -df["failure_mean"].to_numpy(float),
}
out["simple_resolved_detection"] = {}
for name, score in simple_scores.items():
    out["simple_resolved_detection"][name] = {
        "roc_auc": float(roc_auc_score(y_resolved, score)),
        "average_precision": float(average_precision_score(y_resolved, score)),
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
out["lr_resolved_detection"] = {}
for feature_set in ["agent_summary", "agent_full", "agent_plus_metadata"]:
    cols = feature_sets[feature_set]
    prob = np.zeros(len(df), dtype=float)
    for train_idx, test_idx in cv.split(df, y_resolved):
        model = make_model()
        model.fit(df.iloc[train_idx][cols], y_resolved[train_idx])
        prob[test_idx] = model.predict_proba(df.iloc[test_idx][cols])[:, 1]
    out["lr_resolved_detection"][feature_set] = {
        "roc_auc": float(roc_auc_score(y_resolved, prob)),
        "average_precision": float(average_precision_score(y_resolved, prob)),
    }

mask = y_status != 1
resolved_df = df[mask].reset_index(drop=True)
y_helpful = (resolved_df["true_label_3way"].to_numpy(int) == 2).astype(int)
out["direction_h_vs_nh"] = {
    "n_resolved": int(len(resolved_df)),
    "helpful_rate": float(y_helpful.mean()),
}
if len(np.unique(y_helpful)) == 2:
    cv_dir = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    for feature_set in ["agent_summary", "agent_full", "agent_plus_metadata"]:
        cols = feature_sets[feature_set]
        prob = np.zeros(len(resolved_df), dtype=float)
        for train_idx, test_idx in cv_dir.split(resolved_df, y_helpful):
            model = make_model()
            model.fit(resolved_df.iloc[train_idx][cols], y_helpful[train_idx])
            prob[test_idx] = model.predict_proba(resolved_df.iloc[test_idx][cols])[:, 1]
        out["direction_h_vs_nh"][feature_set] = {
            "roc_auc": float(roc_auc_score(y_helpful, prob)),
            "average_precision": float(average_precision_score(y_helpful, prob)),
        }

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
