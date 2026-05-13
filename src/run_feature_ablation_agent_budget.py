from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from optimize_llm_multiagent_aggregation import (
    apply_selective_thresholds,
    best_selective_thresholds,
    binary_metrics,
    build_feature_table,
    fit_predict_logreg,
    inner_select_logreg,
    train_oof_scores_for_spec,
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    family: str
    description: str
    feature_cols: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe feature-group ablations for the calibrated "
            "Community Notes multi-agent aggregator."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--agent-counts", type=str, default="12,24,36,48")
    parser.add_argument("--model-tag", type=str, default="gpt54nano")
    parser.add_argument("--date-tag", type=str, default="20260507")
    parser.add_argument("--run-tag", type=str, default="run1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage-targets", type=str, default="0.65")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/feature_ablation"),
    )
    parser.add_argument(
        "--comparison-summary-csv",
        type=Path,
        default=Path("artifacts/comparison_tables/feature_ablation_summary.csv"),
    )
    parser.add_argument(
        "--comparison-delta-csv",
        type=Path,
        default=Path("artifacts/comparison_tables/feature_ablation_deltas.csv"),
    )
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No values were supplied.")
    return values


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No coverage targets were supplied.")
    return values


def target_key(target: float) -> str:
    return f"{target:.2f}".replace(".", "p")


def existing(cols: list[str], available: set[str]) -> list[str]:
    return [col for col in cols if col in available]


def feature_groups(summary_cols: list[str], agent_cols: list[str]) -> dict[str, list[str]]:
    available = set(summary_cols)
    return {
        "vote_share": existing(["llm_helpful_share"], available),
        "vote_uncertainty": existing(
            [
                "llm_total_votes",
                "helpful_vote_margin_from_half",
                "helpful_vote_entropy",
            ],
            available,
        ),
        "confidence": existing(
            [
                "llm_mean_confidence",
                "confidence_weighted_helpful_share",
            ],
            available,
        ),
        "quality": existing(
            [
                "llm_mean_addresses_core_claim",
                "llm_mean_changes_reader_understanding",
                "llm_mean_note_needed",
                "llm_mean_evidence_strength",
                "quality_signal_mean",
            ],
            available,
        ),
        "failure": existing(
            [
                "llm_misses_key_points_rate",
                "llm_too_minor_rate",
                "failure_signal_mean",
            ],
            available,
        ),
        "cluster_disagreement": existing(
            [
                "equal_cluster_helpful_share",
                "cluster_helpful_share_std",
                "cluster_helpful_share_min",
                "cluster_helpful_share_max",
            ],
            available,
        ),
        "cluster_quality": existing(
            [
                "equal_cluster_note_needed",
                "equal_cluster_changes_reader_understanding",
                "equal_cluster_evidence_strength",
                "equal_cluster_misses_key_points_rate",
                "equal_cluster_too_minor_rate",
            ],
            available,
        ),
        "agent_votes": list(agent_cols),
    }


def unique_cols(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for col in cols:
        if col not in seen:
            seen.add(col)
            output.append(col)
    return output


def build_ablation_specs(summary_cols: list[str], agent_cols: list[str]) -> tuple[list[AblationSpec], pd.DataFrame]:
    groups = feature_groups(summary_cols, agent_cols)
    summary_all = list(summary_cols)
    hybrid_all = unique_cols(summary_cols + agent_cols)

    specs: list[AblationSpec] = []

    def add(name: str, family: str, description: str, cols: list[str]) -> None:
        cols = unique_cols(cols)
        if cols:
            specs.append(AblationSpec(name=name, family=family, description=description, feature_cols=cols))

    vote = groups["vote_share"]
    add("vote_only", "single_addition", "Only aggregate Helpful vote share.", vote)
    add(
        "vote_plus_uncertainty",
        "single_addition",
        "Vote share plus vote-count, margin, and entropy features.",
        vote + groups["vote_uncertainty"],
    )
    add(
        "vote_plus_confidence",
        "single_addition",
        "Vote share plus mean confidence and confidence-weighted Helpful share.",
        vote + groups["confidence"],
    )
    add(
        "vote_plus_quality",
        "single_addition",
        "Vote share plus structured note-quality judgments.",
        vote + groups["quality"],
    )
    add(
        "vote_plus_failure",
        "single_addition",
        "Vote share plus explicit failure signals.",
        vote + groups["failure"],
    )
    add(
        "vote_plus_cluster_disagreement",
        "single_addition",
        "Vote share plus between-cluster disagreement statistics.",
        vote + groups["cluster_disagreement"],
    )
    add(
        "vote_plus_cluster_quality",
        "single_addition",
        "Vote share plus cluster-level quality statistics.",
        vote + groups["cluster_quality"],
    )
    add(
        "vote_plus_quality_plus_cluster_quality",
        "targeted_combo",
        "Vote share plus structured quality judgments and cluster-level quality statistics.",
        vote + groups["quality"] + groups["cluster_quality"],
    )
    add(
        "vote_plus_confidence_plus_quality_plus_cluster_quality",
        "targeted_combo",
        "Vote share plus confidence, structured quality judgments, and cluster-level quality statistics.",
        vote + groups["confidence"] + groups["quality"] + groups["cluster_quality"],
    )
    add(
        "vote_plus_uncertainty_plus_quality_plus_cluster_quality",
        "targeted_combo",
        "Vote share plus uncertainty, structured quality judgments, and cluster-level quality statistics.",
        vote + groups["vote_uncertainty"] + groups["quality"] + groups["cluster_quality"],
    )
    add(
        "agent_votes_only",
        "agent_identity",
        "Per-agent binary vote indicators without summary quality features.",
        groups["agent_votes"],
    )
    add(
        "vote_plus_agent_votes",
        "agent_identity",
        "Vote share plus per-agent binary vote indicators.",
        vote + groups["agent_votes"],
    )

    cumulative = vote.copy()
    for group_name in [
        "vote_uncertainty",
        "confidence",
        "quality",
        "failure",
        "cluster_disagreement",
        "cluster_quality",
    ]:
        cumulative = unique_cols(cumulative + groups[group_name])
        add(
            f"cumulative_through_{group_name}",
            "cumulative",
            f"Cumulative feature set through {group_name}.",
            cumulative,
        )

    add("summary_all", "full_model", "All non-agent summary features.", summary_all)
    add("hybrid_all", "full_model", "All summary features plus per-agent vote indicators.", hybrid_all)

    removable_groups = [
        "vote_uncertainty",
        "confidence",
        "quality",
        "failure",
        "cluster_disagreement",
        "cluster_quality",
    ]
    for group_name in removable_groups:
        remove = set(groups[group_name])
        add(
            f"summary_minus_{group_name}",
            "leave_one_group_out",
            f"All summary features except {group_name}.",
            [col for col in summary_all if col not in remove],
        )
    if agent_cols:
        add(
            "hybrid_minus_agent_votes",
            "leave_one_group_out",
            "Hybrid feature set without per-agent vote indicators.",
            summary_all,
        )

    group_rows = []
    for group_name, cols in groups.items():
        group_rows.append(
            {
                "group": group_name,
                "n_features": len(cols),
                "features": json.dumps(cols, ensure_ascii=False),
            }
        )
    return specs, pd.DataFrame(group_rows)


def bootstrap_binary_accuracy(
    y_true: np.ndarray,
    pred: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        values.append(float((y_true[idx] == pred[idx]).mean()))
    arr = np.asarray(values, dtype=float)
    return {
        "accuracy_ci95_low": float(np.quantile(arr, 0.025)),
        "accuracy_ci95_high": float(np.quantile(arr, 0.975)),
    }


def bootstrap_selective_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    acc_values = []
    cov_values = []
    n = len(y_true)
    resolved = pred >= 0
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        mask = resolved[idx]
        cov_values.append(float(mask.mean()))
        if mask.any():
            acc_values.append(float((y_true[idx][mask] == pred[idx][mask]).mean()))
    acc_arr = np.asarray(acc_values, dtype=float)
    cov_arr = np.asarray(cov_values, dtype=float)
    return {
        "accuracy_ci95_low": float(np.quantile(acc_arr, 0.025)) if len(acc_arr) else np.nan,
        "accuracy_ci95_high": float(np.quantile(acc_arr, 0.975)) if len(acc_arr) else np.nan,
        "coverage_ci95_low": float(np.quantile(cov_arr, 0.025)) if len(cov_arr) else np.nan,
        "coverage_ci95_high": float(np.quantile(cov_arr, 0.975)) if len(cov_arr) else np.nan,
    }


def run_ablation_cv(
    feature_df: pd.DataFrame,
    specs: list[AblationSpec],
    folds: int,
    inner_folds: int,
    seed: int,
    coverage_targets: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = feature_df["true_label"].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = feature_df[["noteId", "true_label"]].copy()
    fold_rows: list[dict[str, float | int | str]] = []
    selective_rows: list[dict[str, float | int | str]] = []
    spec_rows: list[dict[str, float | int | str]] = []

    for spec in specs:
        oof[f"{spec.name}_score"] = np.nan
        oof[f"{spec.name}_pred"] = -1
        for target in coverage_targets:
            oof[f"{spec.name}_selective_{target_key(target)}_pred"] = -1

    for fold, (train_idx, test_idx) in enumerate(cv.split(feature_df, y), start=1):
        for spec in specs:
            selected = inner_select_logreg(
                feature_df,
                y,
                train_idx,
                spec.feature_cols,
                spec.name,
                seed + fold,
                inner_folds,
            )
            train_scores = train_oof_scores_for_spec(
                feature_df,
                y,
                train_idx,
                selected,
                seed + 1000 + fold,
                inner_folds,
            )
            prob, pred = fit_predict_logreg(feature_df, y, train_idx, test_idx, selected)
            oof.loc[test_idx, f"{spec.name}_score"] = prob
            oof.loc[test_idx, f"{spec.name}_pred"] = pred

            row = binary_metrics(y[test_idx], pred)
            row.update(
                {
                    "fold": fold,
                    "method": spec.name,
                    "family": spec.family,
                    "threshold": selected.threshold,
                    "c": selected.c,
                    "class_weight": selected.class_weight or "none",
                    "inner_accuracy": selected.inner_accuracy,
                    "inner_balanced_accuracy": selected.inner_balanced_accuracy,
                    "n_features": len(spec.feature_cols),
                }
            )
            fold_rows.append(row)

            spec_rows.append(
                {
                    "fold": fold,
                    "method": spec.name,
                    "family": spec.family,
                    "threshold": selected.threshold,
                    "c": selected.c,
                    "class_weight": selected.class_weight or "none",
                    "inner_accuracy": selected.inner_accuracy,
                    "inner_balanced_accuracy": selected.inner_balanced_accuracy,
                    "n_features": len(spec.feature_cols),
                    "features": json.dumps(spec.feature_cols, ensure_ascii=False),
                }
            )

            for target in coverage_targets:
                low, high = best_selective_thresholds(y[train_idx], train_scores, target)
                selective = apply_selective_thresholds(y[test_idx], prob, low, high)
                selective_pred = np.full(len(test_idx), -1, dtype=int)
                selective_pred[prob <= low] = 0
                selective_pred[prob >= high] = 1
                oof.loc[test_idx, f"{spec.name}_selective_{target_key(target)}_pred"] = selective_pred
                selective.update(
                    {
                        "fold": fold,
                        "method": spec.name,
                        "family": spec.family,
                        "coverage_target": target,
                        "low_threshold": low,
                        "high_threshold": high,
                        "n_features": len(spec.feature_cols),
                    }
                )
                selective_rows.append(selective)

    return (
        oof,
        pd.DataFrame(fold_rows),
        pd.DataFrame(selective_rows),
        pd.DataFrame(spec_rows),
    )


def summarize_ablation(
    agent_count: int,
    specs: list[AblationSpec],
    oof: pd.DataFrame,
    selective_fold_df: pd.DataFrame,
    coverage_targets: list[float],
    bootstrap_repeats: int,
    seed: int,
) -> pd.DataFrame:
    y = oof["true_label"].to_numpy(dtype=int)
    spec_map = {spec.name: spec for spec in specs}
    rows: list[dict[str, float | int | str]] = []
    for spec in specs:
        pred = oof[f"{spec.name}_pred"].to_numpy(dtype=int)
        full = binary_metrics(y, pred)
        ci = bootstrap_binary_accuracy(y, pred, bootstrap_repeats, seed + agent_count * 101 + len(rows))
        rows.append(
            {
                "agent_count": agent_count,
                "metric": "full",
                "coverage_target": "",
                "method": spec.name,
                "family": spec.family,
                "description": spec.description,
                "n_features": len(spec.feature_cols),
                "accuracy": full["accuracy"],
                "balanced_accuracy": full["balanced_accuracy"],
                "f1": full["f1"],
                "recall_not_helpful": full["recall_not_helpful"],
                "recall_helpful": full["recall_helpful"],
                "coverage": 1.0,
                "resolved_notes": len(y),
                **ci,
                "coverage_ci95_low": 1.0,
                "coverage_ci95_high": 1.0,
            }
        )

    for (method, target), group in selective_fold_df.groupby(["method", "coverage_target"], dropna=False):
        spec = spec_map[str(method)]
        pooled_tp = int(group["tp"].sum())
        pooled_tn = int(group["tn"].sum())
        pooled_fp = int(group["fp"].sum())
        pooled_fn = int(group["fn"].sum())
        pooled_y = np.array([1] * (pooled_tp + pooled_fn) + [0] * (pooled_tn + pooled_fp), dtype=int)
        pooled_pred = np.array([1] * pooled_tp + [0] * pooled_fn + [0] * pooled_tn + [1] * pooled_fp, dtype=int)
        metrics = binary_metrics(pooled_y, pooled_pred) if len(pooled_y) else binary_metrics(np.array([], dtype=int), np.array([], dtype=int))
        resolved_pred = oof[f"{method}_selective_{target_key(float(target))}_pred"].to_numpy(dtype=int)
        ci = bootstrap_selective_metrics(y, resolved_pred, bootstrap_repeats, seed + agent_count * 503 + len(rows))
        rows.append(
            {
                "agent_count": agent_count,
                "metric": "resolved",
                "coverage_target": float(target),
                "method": method,
                "family": spec.family,
                "description": spec.description,
                "n_features": len(spec.feature_cols),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "recall_not_helpful": metrics["recall_not_helpful"],
                "recall_helpful": metrics["recall_helpful"],
                "coverage": float(group["resolved_notes"].sum() / len(y)),
                "resolved_notes": int(group["resolved_notes"].sum()),
                **ci,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["agent_count", "metric", "coverage_target", "family", "accuracy"],
        ascending=[True, True, True, True, False],
    )


def add_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent_count, metric, target), group in summary_df.groupby(["agent_count", "metric", "coverage_target"], dropna=False):
        group = group.copy()
        lookup = group.set_index("method")
        vote_acc = float(lookup.loc["vote_only", "accuracy"]) if "vote_only" in lookup.index else np.nan
        summary_acc = float(lookup.loc["summary_all", "accuracy"]) if "summary_all" in lookup.index else np.nan
        hybrid_acc = float(lookup.loc["hybrid_all", "accuracy"]) if "hybrid_all" in lookup.index else np.nan
        for _, row in group.iterrows():
            payload = row.to_dict()
            payload["delta_accuracy_vs_vote_only"] = float(row["accuracy"]) - vote_acc if not np.isnan(vote_acc) else np.nan
            payload["delta_accuracy_vs_summary_all"] = float(row["accuracy"]) - summary_acc if not np.isnan(summary_acc) else np.nan
            payload["delta_accuracy_vs_hybrid_all"] = float(row["accuracy"]) - hybrid_acc if not np.isnan(hybrid_acc) else np.nan
            rows.append(payload)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    counts = parse_ints(args.agent_counts)
    coverage_targets = parse_floats(args.coverage_targets)
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[pd.DataFrame] = []
    all_delta_rows: list[pd.DataFrame] = []
    all_group_rows: list[pd.DataFrame] = []

    for count in counts:
        variant = f"mf_continuous_n{count:03d}"
        run_name = f"{variant}_{args.model_tag}_{args.date_tag}_{args.run_tag}"
        llm_dir = repo_root / "artifacts" / "llm_runs" / run_name
        output_dir = output_root / f"{variant}_feature_ablation_{args.date_tag}"
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_df, summary_cols, agent_cols, _hybrid_cols = build_feature_table(
            llm_dir / "note_predictions.csv",
            llm_dir / "agent_votes.csv",
        )
        specs, group_df = build_ablation_specs(summary_cols, agent_cols)
        group_df.insert(0, "agent_count", count)

        oof, fold_df, selective_fold_df, selected_specs_df = run_ablation_cv(
            feature_df=feature_df,
            specs=specs,
            folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
            coverage_targets=coverage_targets,
        )
        summary_df = summarize_ablation(
            agent_count=count,
            specs=specs,
            oof=oof,
            selective_fold_df=selective_fold_df,
            coverage_targets=coverage_targets,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        delta_df = add_deltas(summary_df)

        feature_df.to_csv(output_dir / "feature_ablation_feature_table.csv", index=False, encoding="utf-8-sig")
        oof.to_csv(output_dir / "feature_ablation_oof_predictions.csv", index=False, encoding="utf-8-sig")
        fold_df.to_csv(output_dir / "feature_ablation_fold_metrics.csv", index=False, encoding="utf-8-sig")
        selective_fold_df.to_csv(output_dir / "feature_ablation_selective_fold_metrics.csv", index=False, encoding="utf-8-sig")
        selected_specs_df.to_csv(output_dir / "feature_ablation_selected_model_specs.csv", index=False, encoding="utf-8-sig")
        group_df.to_csv(output_dir / "feature_ablation_feature_groups.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(output_dir / "feature_ablation_summary.csv", index=False, encoding="utf-8-sig")
        delta_df.to_csv(output_dir / "feature_ablation_deltas.csv", index=False, encoding="utf-8-sig")

        metadata = {
            "agent_count": count,
            "variant": variant,
            "llm_dir": str(llm_dir),
            "notes_total": int(len(feature_df)),
            "folds": int(args.folds),
            "inner_folds": int(args.inner_folds),
            "seed": int(args.seed),
            "coverage_targets": coverage_targets,
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "ablation_specs": [
                {
                    "name": spec.name,
                    "family": spec.family,
                    "description": spec.description,
                    "n_features": len(spec.feature_cols),
                    "features": spec.feature_cols,
                }
                for spec in specs
            ],
        }
        with (output_dir / "feature_ablation_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        all_summary_rows.append(summary_df)
        all_delta_rows.append(delta_df)
        all_group_rows.append(group_df)

    combined_summary = pd.concat(all_summary_rows, ignore_index=True)
    combined_delta = pd.concat(all_delta_rows, ignore_index=True)
    combined_groups = pd.concat(all_group_rows, ignore_index=True)

    summary_path = args.comparison_summary_csv if args.comparison_summary_csv.is_absolute() else repo_root / args.comparison_summary_csv
    delta_path = args.comparison_delta_csv if args.comparison_delta_csv.is_absolute() else repo_root / args.comparison_delta_csv
    group_path = summary_path.with_name("feature_ablation_feature_groups.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    combined_delta.to_csv(delta_path, index=False, encoding="utf-8-sig")
    combined_groups.to_csv(group_path, index=False, encoding="utf-8-sig")

    print(summary_path)
    print(combined_summary.to_string(index=False))
    print(delta_path)


if __name__ == "__main__":
    main()
