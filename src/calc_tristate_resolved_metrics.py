#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("analysis/llm_16agent_tristate_pilot_20_20260512/note_vote_summary.csv"),
    )
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    df["is_full_correct"] = df["true_label_3way"] == df["predicted_label_3way_majority"]

    # Agent-vote status rule:
    # predicted NEEDS_MORE_RATINGS is unresolved; predicted HELPFUL / NOT_HELPFUL is resolved.
    resolved = df[df["predicted_label_3way_majority"].isin([0, 2])].copy()
    resolved["is_resolved_correct"] = resolved["true_label_3way"] == resolved["predicted_label_3way_majority"]

    by_true = (
        df.groupby("true_label_text", dropna=False)
        .agg(
            notes=("noteId", "size"),
            strict_correct=("is_full_correct", "sum"),
        )
        .reset_index()
    )
    by_true["strict_accuracy"] = by_true["strict_correct"] / by_true["notes"]

    if len(resolved):
        by_true_resolved = (
            resolved.groupby("true_label_text", dropna=False)
            .agg(
                resolved_notes=("noteId", "size"),
                resolved_correct=("is_resolved_correct", "sum"),
            )
            .reset_index()
        )
        by_true_resolved["resolved_accuracy"] = (
            by_true_resolved["resolved_correct"] / by_true_resolved["resolved_notes"]
        )
    else:
        by_true_resolved = pd.DataFrame()

    confusion = pd.crosstab(
        df["true_label_text"],
        df["predicted_rating_majority"],
        margins=True,
    )
    resolved_confusion = (
        pd.crosstab(
            resolved["true_label_text"],
            resolved["predicted_rating_majority"],
            margins=True,
        )
        if len(resolved)
        else pd.DataFrame()
    )

    out = {
        "total_notes": int(len(df)),
        "full_strict_correct": int(df["is_full_correct"].sum()),
        "full_strict_accuracy": float(df["is_full_correct"].mean()),
        "resolved_rule": "resolved iff majority agent prediction is HELPFUL or NOT_HELPFUL; majority NEEDS_MORE_RATINGS is unresolved",
        "resolved_notes": int(len(resolved)),
        "unresolved_notes": int(len(df) - len(resolved)),
        "coverage": float(len(resolved) / len(df)) if len(df) else None,
        "resolved_correct": int(resolved["is_resolved_correct"].sum()) if len(resolved) else 0,
        "resolved_accuracy": float(resolved["is_resolved_correct"].mean()) if len(resolved) else None,
        "by_true_status": by_true.to_dict(orient="records"),
        "by_true_status_resolved_only": by_true_resolved.to_dict(orient="records"),
        "confusion": confusion.to_dict(),
        "resolved_confusion": resolved_confusion.to_dict() if len(resolved_confusion) else {},
        "resolved_note_ids": resolved[
            [
                "noteId",
                "true_label_text",
                "predicted_rating_majority",
                "vote_helpful",
                "vote_need_more_ratings",
                "vote_not_helpful",
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
