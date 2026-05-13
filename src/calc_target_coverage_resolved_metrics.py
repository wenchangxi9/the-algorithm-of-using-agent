#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--target-coverage", type=float, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def predict_h_or_nh(row: pd.Series) -> int:
    if int(row["vote_helpful"]) >= int(row["vote_not_helpful"]):
        return 2
    return 0


def metric_block(resolved: pd.DataFrame, total: int) -> dict:
    if resolved.empty:
        return {
            "resolved_notes": 0,
            "coverage": 0.0,
            "resolved_correct": 0,
            "resolved_accuracy": None,
            "resolved_note_ids": [],
        }
    correct = resolved["predicted_h_or_nh"] == resolved["true_label_3way"]
    cols = [
        "noteId",
        "true_label_text",
        "predicted_h_or_nh_text",
        "resolved_score",
        "vote_helpful",
        "vote_need_more_ratings",
        "vote_not_helpful",
    ]
    return {
        "resolved_notes": int(len(resolved)),
        "coverage": float(len(resolved) / total),
        "resolved_correct": int(correct.sum()),
        "resolved_accuracy": float(correct.mean()),
        "resolved_note_ids": resolved[cols].to_dict(orient="records"),
    }


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    df["resolved_score"] = (df["vote_helpful"] + df["vote_not_helpful"]) / df["n_votes"]
    df["predicted_h_or_nh"] = df.apply(predict_h_or_nh, axis=1)
    df["predicted_h_or_nh_text"] = df["predicted_h_or_nh"].map({2: "HELPFUL", 0: "NOT_HELPFUL"})
    df = df.sort_values(["resolved_score", "mean_confidence"], ascending=[False, False]).reset_index(drop=True)

    n = len(df)
    target_k_round = max(1, int(round(args.target_coverage * n)))
    target_k_ceil = max(1, int(math.ceil(args.target_coverage * n)))

    exact_round = df.head(target_k_round).copy()
    exact_ceil = df.head(target_k_ceil).copy()

    threshold = float(df.iloc[target_k_ceil - 1]["resolved_score"])
    tie_inclusive = df[df["resolved_score"] >= threshold].copy()

    out = {
        "total_notes": int(n),
        "target_coverage": float(args.target_coverage),
        "target_k_round": int(target_k_round),
        "target_k_ceil": int(target_k_ceil),
        "resolved_score_definition": "(vote_helpful + vote_not_helpful) / n_votes; higher means less likely to be Need More Ratings",
        "binary_label_rule_for_resolved": "HELPFUL if vote_helpful >= vote_not_helpful else NOT_HELPFUL",
        "exact_top_k_round": metric_block(exact_round, n),
        "exact_top_k_ceil": metric_block(exact_ceil, n),
        "tie_inclusive_at_ceil_threshold": {
            "threshold": threshold,
            **metric_block(tie_inclusive, n),
        },
        "all_scores": df[
            [
                "noteId",
                "true_label_text",
                "resolved_score",
                "predicted_h_or_nh_text",
                "vote_helpful",
                "vote_need_more_ratings",
                "vote_not_helpful",
                "mean_confidence",
            ]
        ].to_dict(orient="records"),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
