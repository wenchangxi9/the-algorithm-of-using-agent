from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the core Community Notes MF-continuous multi-agent results: "
            "raw majority, official-style MF resolution, and calibrated nested-CV aggregation."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--agent-counts", type=str, default="12,24,36,48")
    parser.add_argument("--model-tag", type=str, default="gpt54nano")
    parser.add_argument("--date-tag", type=str, default="20260507")
    parser.add_argument("--run-tag", type=str, default="run1")
    parser.add_argument("--resolved-coverage-target", type=float, default=0.65)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/comparison_tables/core_results_summary.csv"),
    )
    return parser.parse_args()


def parse_counts(text: str) -> list[int]:
    counts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not counts:
        raise ValueError("No agent counts were supplied.")
    return counts


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def raw_majority_metrics(notes_csv: Path) -> dict[str, float | int]:
    df = pd.read_csv(notes_csv, low_memory=False)
    y_true = numeric_series(df["true_label"])
    y_pred = numeric_series(df["llm_pred_label"])
    keep = y_true.notna() & y_pred.notna()
    y_true = y_true[keep].astype(int)
    y_pred = y_pred[keep].astype(int)
    if len(y_true) == 0:
        return {"accuracy": np.nan, "coverage": 0.0, "resolved_notes": 0}
    return {
        "accuracy": float((y_true.to_numpy() == y_pred.to_numpy()).mean()),
        "coverage": float(len(y_true) / len(df)),
        "resolved_notes": int(len(y_true)),
    }


def official_style_metrics(summary_csv: Path) -> dict[str, float | int]:
    df = pd.read_csv(summary_csv, low_memory=False)
    if df.empty:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "coverage": 0.0, "resolved_notes": 0}
    row = df.iloc[0]
    coverage_col = "note_coverage" if "note_coverage" in df.columns else "note_coverage_mean"
    coverage = float(row[coverage_col])
    accuracy_col = "accuracy" if "accuracy" in df.columns else "accuracy_mean"
    balanced_col = "balanced_accuracy" if "balanced_accuracy" in df.columns else "balanced_accuracy_mean"
    return {
        "accuracy": float(row[accuracy_col]),
        "balanced_accuracy": float(row[balanced_col]),
        "coverage": coverage,
        "resolved_notes": int(round(coverage * 258)),
    }


def calibrated_full_metrics(method_summary_csv: Path) -> dict[str, float | int | str]:
    df = pd.read_csv(method_summary_csv, low_memory=False)
    calibrated = df[df["method"].astype(str).str.contains("logreg_nested_cv", regex=False)].copy()
    if calibrated.empty:
        calibrated = df.copy()
    row = calibrated.sort_values(["accuracy", "balanced_accuracy"], ascending=False).iloc[0]
    return {
        "method": str(row["method"]),
        "accuracy": float(row["accuracy"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "coverage": 1.0,
        "resolved_notes": 258,
    }


def calibrated_resolved_metrics(selective_csv: Path, target: float) -> dict[str, float | int | str]:
    df = pd.read_csv(selective_csv, low_memory=False)
    df["coverage_target_delta"] = (pd.to_numeric(df["coverage_target"]) - target).abs()
    closest = float(df.sort_values("coverage_target_delta").iloc[0]["coverage_target"])
    subset = df[np.isclose(pd.to_numeric(df["coverage_target"]), closest)].copy()
    calibrated = subset[subset["method"].astype(str).str.contains("logreg_nested_cv", regex=False)].copy()
    if calibrated.empty:
        calibrated = subset.copy()
    row = calibrated.sort_values(["pooled_accuracy", "pooled_balanced_accuracy"], ascending=False).iloc[0]
    return {
        "method": str(row["method"]),
        "coverage_target": closest,
        "accuracy": float(row["pooled_accuracy"]),
        "balanced_accuracy": float(row["pooled_balanced_accuracy"]),
        "coverage": float(row["coverage_mean"]),
        "resolved_notes": int(row["pooled_resolved_notes"]),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    rows: list[dict[str, float | int | str]] = []

    for count in parse_counts(args.agent_counts):
        variant = f"mf_continuous_n{count:03d}"
        run_name = f"{variant}_{args.model_tag}_{args.date_tag}_{args.run_tag}"
        llm_dir = repo_root / "artifacts" / "llm_runs" / run_name
        official_dir = repo_root / "artifacts" / "official_style_results" / run_name
        calibrated_dir = repo_root / "artifacts" / "calibrated_aggregation" / f"{variant}_optimized_aggregation_{args.date_tag}"

        raw = raw_majority_metrics(llm_dir / "note_predictions.csv")
        rows.append(
            {
                "agent_count": count,
                "metric": "raw_majority_full",
                "method": "majority_vote",
                "accuracy": raw["accuracy"],
                "balanced_accuracy": "",
                "coverage": raw["coverage"],
                "resolved_notes": raw["resolved_notes"],
                "notes_basis": "evaluated notes with a majority prediction",
            }
        )

        official = official_style_metrics(official_dir / "agent_count_official_summary.csv")
        rows.append(
            {
                "agent_count": count,
                "metric": "official_style_mf_resolved",
                "method": "rank1_mf_resolution",
                "accuracy": official["accuracy"],
                "balanced_accuracy": official["balanced_accuracy"],
                "coverage": official["coverage"],
                "resolved_notes": official["resolved_notes"],
                "notes_basis": "resolved subset",
            }
        )

        full = calibrated_full_metrics(calibrated_dir / "nested_cv_method_summary.csv")
        rows.append(
            {
                "agent_count": count,
                "metric": "calibrated_full_nested_cv",
                "method": full["method"],
                "accuracy": full["accuracy"],
                "balanced_accuracy": full["balanced_accuracy"],
                "coverage": full["coverage"],
                "resolved_notes": full["resolved_notes"],
                "notes_basis": "all notes, outer 5-fold nested CV",
            }
        )

        resolved = calibrated_resolved_metrics(
            calibrated_dir / "nested_cv_selective_summary_by_coverage.csv",
            args.resolved_coverage_target,
        )
        rows.append(
            {
                "agent_count": count,
                "metric": f"calibrated_resolved_nested_cv_target_{resolved['coverage_target']:.2f}",
                "method": resolved["method"],
                "accuracy": resolved["accuracy"],
                "balanced_accuracy": resolved["balanced_accuracy"],
                "coverage": resolved["coverage"],
                "resolved_notes": resolved["resolved_notes"],
                "notes_basis": "selectively resolved subset, outer 5-fold nested CV",
            }
        )

    summary = pd.DataFrame(rows).sort_values(["agent_count", "metric"]).reset_index(drop=True)
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(output_csv)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
