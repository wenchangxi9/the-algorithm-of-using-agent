from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUMMARY_FEATURES = [
    "llm_helpful_share",
    "llm_total_votes",
    "llm_mean_confidence",
    "llm_mean_addresses_core_claim",
    "llm_mean_changes_reader_understanding",
    "llm_mean_note_needed",
    "llm_mean_evidence_strength",
    "llm_misses_key_points_rate",
    "llm_too_minor_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize LLM multi-agent aggregation with leakage-safe cross validation. "
            "This consumes completed note_predictions.csv and agent_votes.csv files; it does not call an LLM."
        )
    )
    parser.add_argument("--note-predictions-csv", type=Path, required=True)
    parser.add_argument("--agent-votes-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selective-coverages",
        type=str,
        default="0.25,0.50,0.65,0.75",
        help="Minimum coverage targets for selective/resolved prediction.",
    )
    return parser.parse_args()


def safe_cluster_payload(raw: object) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def finite_mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def finite_std(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanstd(arr))


def extract_cluster_features(series: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for raw in series:
        payload = safe_cluster_payload(raw)
        values = list(payload.values())
        helpful = [float(v.get("helpful_share", np.nan)) for v in values]
        note_needed = [float(v.get("note_needed", np.nan)) for v in values]
        changes = [float(v.get("changes_reader_understanding", np.nan)) for v in values]
        evidence = [float(v.get("evidence_strength", np.nan)) for v in values]
        misses = [float(v.get("misses_key_points_rate", np.nan)) for v in values]
        too_minor = [float(v.get("too_minor_rate", np.nan)) for v in values]
        rows.append(
            {
                "equal_cluster_helpful_share": finite_mean(helpful),
                "cluster_helpful_share_std": finite_std(helpful),
                "cluster_helpful_share_min": float(np.nanmin(helpful)) if len(helpful) else np.nan,
                "cluster_helpful_share_max": float(np.nanmax(helpful)) if len(helpful) else np.nan,
                "equal_cluster_note_needed": finite_mean(note_needed),
                "equal_cluster_changes_reader_understanding": finite_mean(changes),
                "equal_cluster_evidence_strength": finite_mean(evidence),
                "equal_cluster_misses_key_points_rate": finite_mean(misses),
                "equal_cluster_too_minor_rate": finite_mean(too_minor),
            }
        )
    return pd.DataFrame(rows)


def build_feature_table(note_predictions_csv: Path, agent_votes_csv: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    note_df = pd.read_csv(note_predictions_csv.resolve(), low_memory=False)
    note_df = note_df[note_df["true_label"].isin([0, 1])].copy()
    note_df["noteId"] = note_df["noteId"].astype(str)
    note_df = note_df.reset_index(drop=True)

    for col in SUMMARY_FEATURES:
        if col not in note_df.columns:
            note_df[col] = np.nan

    cluster_features = extract_cluster_features(note_df.get("cluster_vote_profile_json", pd.Series([""] * len(note_df))))
    feature_df = pd.concat(
        [
            note_df[["noteId", "true_label", *SUMMARY_FEATURES]].reset_index(drop=True),
            cluster_features.reset_index(drop=True),
        ],
        axis=1,
    )
    feature_df["helpful_vote_margin_from_half"] = (feature_df["llm_helpful_share"].astype(float) - 0.5).abs()
    share = feature_df["llm_helpful_share"].clip(1e-6, 1 - 1e-6).astype(float)
    feature_df["helpful_vote_entropy"] = -(share * np.log(share) + (1 - share) * np.log(1 - share))
    feature_df["quality_signal_mean"] = feature_df[
        [
            "llm_mean_addresses_core_claim",
            "llm_mean_changes_reader_understanding",
            "llm_mean_note_needed",
            "llm_mean_evidence_strength",
        ]
    ].mean(axis=1)
    feature_df["failure_signal_mean"] = feature_df[["llm_misses_key_points_rate", "llm_too_minor_rate"]].mean(axis=1)

    votes = pd.read_csv(agent_votes_csv.resolve(), low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes = votes[votes["predicted_label"].isin([0, 1])].copy()
    votes["predicted_label"] = votes["predicted_label"].astype(float)
    votes["confidence_num"] = pd.to_numeric(votes.get("confidence", np.nan), errors="coerce")

    vote_pivot = votes.pivot_table(index="noteId", columns="agent_id", values="predicted_label", aggfunc="mean")
    vote_pivot = vote_pivot.add_prefix("agent_vote__").reset_index()

    conf_weight = votes.copy()
    conf_weight["weight"] = conf_weight["confidence_num"].clip(lower=0).fillna(0.0)
    weighted = (
        conf_weight.assign(weighted_helpful=lambda df: df["predicted_label"] * df["weight"])
        .groupby("noteId", as_index=False)
        .agg(weighted_helpful=("weighted_helpful", "sum"), total_weight=("weight", "sum"))
    )
    weighted["confidence_weighted_helpful_share"] = np.divide(
        weighted["weighted_helpful"],
        weighted["total_weight"],
        out=np.full(len(weighted), np.nan, dtype=float),
        where=weighted["total_weight"].to_numpy(dtype=float) > 0,
    )
    weighted = weighted[["noteId", "confidence_weighted_helpful_share"]]

    feature_df = feature_df.merge(vote_pivot, on="noteId", how="left")
    feature_df = feature_df.merge(weighted, on="noteId", how="left")

    summary_cols = [
        col
        for col in feature_df.columns
        if col
        not in {
            "noteId",
            "true_label",
        }
        and not col.startswith("agent_vote__")
    ]
    agent_cols = [col for col in feature_df.columns if col.startswith("agent_vote__")]
    hybrid_cols = summary_cols + agent_cols
    return feature_df, summary_cols, agent_cols, hybrid_cols


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = tn + fp + fn + tp
    recall_not_helpful = float(tn / (tn + fp)) if (tn + fp) else 0.0
    recall_helpful = float(tp / (tp + fn)) if (tp + fn) else 0.0
    return {
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "balanced_accuracy": float((recall_not_helpful + recall_helpful) / 2) if total else 0.0,
        "f1": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "recall_not_helpful": recall_not_helpful,
        "recall_helpful": recall_helpful,
    }


def best_threshold(y_true: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> tuple[float, dict[str, float | int]]:
    best: tuple[tuple[float, float, float], float, dict[str, float | int]] | None = None
    for threshold in thresholds:
        pred = (score >= threshold).astype(int)
        metrics = binary_metrics(y_true, pred)
        key = (
            float(metrics["accuracy"]),
            float(metrics["balanced_accuracy"]),
            -abs(float(pred.mean()) - float(y_true.mean())),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return best[1], best[2]


def make_model(c: float, class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=5000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_cols: list[str]
    c: float
    class_weight: str | None
    threshold: float
    inner_accuracy: float
    inner_balanced_accuracy: float


def inner_select_logreg(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    feature_cols: list[str],
    name: str,
    seed: int,
    inner_folds: int,
) -> ModelSpec:
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    class_weights: list[str | None] = [None, "balanced"]
    prob_thresholds = np.arange(0.20, 0.81, 0.01)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    best: tuple[tuple[float, float, float], ModelSpec] | None = None
    X_train = X.iloc[train_idx][feature_cols]
    y_train = y[train_idx]

    for c in c_grid:
        for class_weight in class_weights:
            oof_prob = np.zeros(len(train_idx), dtype=float)
            for inner_train_pos, inner_val_pos in inner_cv.split(X_train, y_train):
                model = make_model(c, class_weight)
                model.fit(X_train.iloc[inner_train_pos], y_train[inner_train_pos])
                oof_prob[inner_val_pos] = model.predict_proba(X_train.iloc[inner_val_pos])[:, 1]

            threshold, metrics = best_threshold(y_train, oof_prob, prob_thresholds)
            key = (
                float(metrics["accuracy"]),
                float(metrics["balanced_accuracy"]),
                -abs(threshold - 0.5),
            )
            spec = ModelSpec(
                name=name,
                feature_cols=feature_cols,
                c=c,
                class_weight=class_weight,
                threshold=threshold,
                inner_accuracy=float(metrics["accuracy"]),
                inner_balanced_accuracy=float(metrics["balanced_accuracy"]),
            )
            if best is None or key > best[0]:
                best = (key, spec)
    assert best is not None
    return best[1]


def fit_predict_logreg(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    model = make_model(spec.c, spec.class_weight)
    model.fit(X.iloc[train_idx][spec.feature_cols], y[train_idx])
    prob = model.predict_proba(X.iloc[test_idx][spec.feature_cols])[:, 1]
    pred = (prob >= spec.threshold).astype(int)
    return prob, pred


def train_oof_scores_for_spec(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    spec: ModelSpec,
    seed: int,
    inner_folds: int,
) -> np.ndarray:
    X_train = X.iloc[train_idx][spec.feature_cols]
    y_train = y[train_idx]
    scores = np.zeros(len(train_idx), dtype=float)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    for inner_train_pos, inner_val_pos in inner_cv.split(X_train, y_train):
        model = make_model(spec.c, spec.class_weight)
        model.fit(X_train.iloc[inner_train_pos], y_train[inner_train_pos])
        scores[inner_val_pos] = model.predict_proba(X_train.iloc[inner_val_pos])[:, 1]
    return scores


def best_selective_thresholds(
    y_true: np.ndarray,
    score: np.ndarray,
    min_coverage: float,
) -> tuple[float, float]:
    best: tuple[tuple[float, float, float], float, float] | None = None
    low_grid = np.arange(0.00, 0.51, 0.01)
    high_grid = np.arange(0.50, 0.96, 0.01)
    for low in low_grid:
        for high in high_grid:
            if low >= high:
                continue
            pred = np.full(len(score), -1, dtype=int)
            pred[score <= low] = 0
            pred[score >= high] = 1
            mask = pred >= 0
            coverage = float(mask.mean())
            if coverage < min_coverage or not mask.any():
                continue
            metrics = binary_metrics(y_true[mask], pred[mask])
            key = (
                float(metrics["accuracy"]),
                float(metrics["balanced_accuracy"]),
                coverage,
            )
            if best is None or key > best[0]:
                best = (key, float(low), float(high))
    if best is None:
        return 0.0, 1.0
    return best[1], best[2]


def apply_selective_thresholds(
    y_true: np.ndarray,
    score: np.ndarray,
    low: float,
    high: float,
) -> dict[str, float | int]:
    pred = np.full(len(score), -1, dtype=int)
    pred[score <= low] = 0
    pred[score >= high] = 1
    mask = pred >= 0
    metrics = binary_metrics(y_true[mask], pred[mask]) if mask.any() else binary_metrics(np.array([], dtype=int), np.array([], dtype=int))
    metrics["coverage"] = float(mask.mean())
    metrics["resolved_notes"] = int(mask.sum())
    return metrics


def run_cross_validation(
    feature_df: pd.DataFrame,
    summary_cols: list[str],
    agent_cols: list[str],
    hybrid_cols: list[str],
    folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = feature_df["true_label"].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    raw_thresholds = np.arange(0.0, 0.81, 0.01)
    method_names = [
        "majority_threshold_050",
        "raw_share_threshold_cv",
        "confidence_weighted_threshold_cv",
        "summary_logreg_nested_cv",
        "agent_vote_logreg_nested_cv",
        "hybrid_logreg_nested_cv",
    ]

    oof = feature_df[["noteId", "true_label"]].copy()
    for method in method_names:
        oof[f"{method}_score"] = np.nan
        oof[f"{method}_pred"] = -1
    fold_rows: list[dict[str, float | int | str]] = []
    selective_fold_rows: list[dict[str, float | int | str]] = []
    spec_rows: list[dict[str, float | int | str | None]] = []
    selective_coverage_targets = [0.25, 0.50, 0.65, 0.75]

    for fold, (train_idx, test_idx) in enumerate(cv.split(feature_df, y), start=1):
        train_share = feature_df.iloc[train_idx]["llm_helpful_share"].fillna(0.0).to_numpy(dtype=float)
        test_share = feature_df.iloc[test_idx]["llm_helpful_share"].fillna(0.0).to_numpy(dtype=float)

        majority_pred = (test_share >= 0.5).astype(int)
        oof.loc[test_idx, "majority_threshold_050_score"] = test_share
        oof.loc[test_idx, "majority_threshold_050_pred"] = majority_pred
        row = binary_metrics(y[test_idx], majority_pred)
        row.update({"fold": fold, "method": "majority_threshold_050", "threshold": 0.5})
        fold_rows.append(row)
        for target in selective_coverage_targets:
            low, high = best_selective_thresholds(y[train_idx], train_share, target)
            selective = apply_selective_thresholds(y[test_idx], test_share, low, high)
            selective.update(
                {
                    "fold": fold,
                    "method": "majority_threshold_050",
                    "coverage_target": target,
                    "low_threshold": low,
                    "high_threshold": high,
                }
            )
            selective_fold_rows.append(selective)

        threshold, train_metrics = best_threshold(y[train_idx], train_share, raw_thresholds)
        pred = (test_share >= threshold).astype(int)
        oof.loc[test_idx, "raw_share_threshold_cv_score"] = test_share
        oof.loc[test_idx, "raw_share_threshold_cv_pred"] = pred
        row = binary_metrics(y[test_idx], pred)
        row.update(
            {
                "fold": fold,
                "method": "raw_share_threshold_cv",
                "threshold": threshold,
                "inner_accuracy": train_metrics["accuracy"],
                "inner_balanced_accuracy": train_metrics["balanced_accuracy"],
            }
        )
        fold_rows.append(row)
        for target in selective_coverage_targets:
            low, high = best_selective_thresholds(y[train_idx], train_share, target)
            selective = apply_selective_thresholds(y[test_idx], test_share, low, high)
            selective.update(
                {
                    "fold": fold,
                    "method": "raw_share_threshold_cv",
                    "coverage_target": target,
                    "low_threshold": low,
                    "high_threshold": high,
                }
            )
            selective_fold_rows.append(selective)

        train_weighted_raw = (
            feature_df.iloc[train_idx]["confidence_weighted_helpful_share"].to_numpy(dtype=float)
        )
        test_weighted_raw = (
            feature_df.iloc[test_idx]["confidence_weighted_helpful_share"].to_numpy(dtype=float)
        )
        train_weighted = np.where(np.isnan(train_weighted_raw), train_share, train_weighted_raw)
        test_weighted = np.where(np.isnan(test_weighted_raw), test_share, test_weighted_raw)
        threshold, train_metrics = best_threshold(y[train_idx], train_weighted, raw_thresholds)
        pred = (test_weighted >= threshold).astype(int)
        oof.loc[test_idx, "confidence_weighted_threshold_cv_score"] = test_weighted
        oof.loc[test_idx, "confidence_weighted_threshold_cv_pred"] = pred
        row = binary_metrics(y[test_idx], pred)
        row.update(
            {
                "fold": fold,
                "method": "confidence_weighted_threshold_cv",
                "threshold": threshold,
                "inner_accuracy": train_metrics["accuracy"],
                "inner_balanced_accuracy": train_metrics["balanced_accuracy"],
            }
        )
        fold_rows.append(row)
        for target in selective_coverage_targets:
            low, high = best_selective_thresholds(y[train_idx], train_weighted, target)
            selective = apply_selective_thresholds(y[test_idx], test_weighted, low, high)
            selective.update(
                {
                    "fold": fold,
                    "method": "confidence_weighted_threshold_cv",
                    "coverage_target": target,
                    "low_threshold": low,
                    "high_threshold": high,
                }
            )
            selective_fold_rows.append(selective)

        for name, cols in [
            ("summary_logreg_nested_cv", summary_cols),
            ("agent_vote_logreg_nested_cv", agent_cols),
            ("hybrid_logreg_nested_cv", hybrid_cols),
        ]:
            if not cols:
                continue
            spec = inner_select_logreg(feature_df, y, train_idx, cols, name, seed + fold, inner_folds)
            train_scores = train_oof_scores_for_spec(feature_df, y, train_idx, spec, seed + 1000 + fold, inner_folds)
            prob, pred = fit_predict_logreg(feature_df, y, train_idx, test_idx, spec)
            oof.loc[test_idx, f"{name}_score"] = prob
            oof.loc[test_idx, f"{name}_pred"] = pred
            row = binary_metrics(y[test_idx], pred)
            row.update(
                {
                    "fold": fold,
                    "method": name,
                    "threshold": spec.threshold,
                    "c": spec.c,
                    "class_weight": spec.class_weight or "none",
                    "inner_accuracy": spec.inner_accuracy,
                    "inner_balanced_accuracy": spec.inner_balanced_accuracy,
                }
            )
            fold_rows.append(row)
            for target in selective_coverage_targets:
                low, high = best_selective_thresholds(y[train_idx], train_scores, target)
                selective = apply_selective_thresholds(y[test_idx], prob, low, high)
                selective.update(
                    {
                        "fold": fold,
                        "method": name,
                        "coverage_target": target,
                        "low_threshold": low,
                        "high_threshold": high,
                    }
                )
                selective_fold_rows.append(selective)
            spec_rows.append(
                {
                    "fold": fold,
                    "method": name,
                    "c": spec.c,
                    "class_weight": spec.class_weight or "none",
                    "threshold": spec.threshold,
                    "inner_accuracy": spec.inner_accuracy,
                    "inner_balanced_accuracy": spec.inner_balanced_accuracy,
                    "n_features": len(cols),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    selective_fold_df = pd.DataFrame(selective_fold_rows)
    spec_df = pd.DataFrame(spec_rows)
    return oof, fold_df, selective_fold_df, spec_df


def summarize_oof(oof: pd.DataFrame) -> pd.DataFrame:
    y = oof["true_label"].to_numpy(dtype=int)
    rows = []
    for col in oof.columns:
        if not col.endswith("_pred"):
            continue
        method = col[: -len("_pred")]
        pred = oof[col].to_numpy(dtype=int)
        if (pred < 0).any():
            continue
        metrics = binary_metrics(y, pred)
        metrics["method"] = method
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy"], ascending=False).reset_index(drop=True)


def summarize_selective_folds(selective_fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "recall_not_helpful",
        "recall_helpful",
        "coverage",
        "resolved_notes",
    ]
    rows = []
    for (method, target), group in selective_fold_df.groupby(["method", "coverage_target"], dropna=False):
        pooled_tp = int(group["tp"].sum())
        pooled_tn = int(group["tn"].sum())
        pooled_fp = int(group["fp"].sum())
        pooled_fn = int(group["fn"].sum())
        pooled_y = np.array([1] * (pooled_tp + pooled_fn) + [0] * (pooled_tn + pooled_fp), dtype=int)
        pooled_pred = np.array([1] * pooled_tp + [0] * pooled_fn + [0] * pooled_tn + [1] * pooled_fp, dtype=int)
        pooled = binary_metrics(pooled_y, pooled_pred) if len(pooled_y) else {}
        row: dict[str, float | int | str] = {
            "method": str(method),
            "coverage_target": float(target),
            "folds": int(len(group)),
            "pooled_accuracy": float(pooled.get("accuracy", 0.0)),
            "pooled_balanced_accuracy": float(pooled.get("balanced_accuracy", 0.0)),
            "pooled_f1": float(pooled.get("f1", 0.0)),
            "pooled_tn": pooled_tn,
            "pooled_fp": pooled_fp,
            "pooled_fn": pooled_fn,
            "pooled_tp": pooled_tp,
            "pooled_resolved_notes": int(group["resolved_notes"].sum()),
        }
        for col in metric_cols:
            values = group[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["coverage_target", "pooled_accuracy", "pooled_balanced_accuracy"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def selective_summary(
    oof: pd.DataFrame,
    method: str,
    coverage_targets: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = oof["true_label"].to_numpy(dtype=int)
    score = oof[f"{method}_score"].to_numpy(dtype=float)
    rows = []
    for high in np.arange(0.50, 0.96, 0.01):
        for low in np.arange(0.00, 0.51, 0.01):
            pred = np.full(len(score), -1, dtype=int)
            pred[score >= high] = 1
            pred[score <= low] = 0
            mask = pred >= 0
            if not mask.any():
                continue
            metrics = binary_metrics(y[mask], pred[mask])
            metrics.update(
                {
                    "method": method,
                    "low_threshold": float(low),
                    "high_threshold": float(high),
                    "coverage": float(mask.mean()),
                    "resolved_notes": int(mask.sum()),
                }
            )
            rows.append(metrics)
    curve_df = pd.DataFrame(rows)
    best_rows = []
    for target in coverage_targets:
        eligible = curve_df[curve_df["coverage"] >= target].copy()
        if eligible.empty:
            continue
        row = eligible.sort_values(
            ["accuracy", "balanced_accuracy", "coverage"],
            ascending=False,
        ).iloc[0]
        payload = row.to_dict()
        payload["coverage_target"] = target
        best_rows.append(payload)
    return curve_df, pd.DataFrame(best_rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_targets = [float(v.strip()) for v in args.selective_coverages.split(",") if v.strip()]

    feature_df, summary_cols, agent_cols, hybrid_cols = build_feature_table(
        args.note_predictions_csv,
        args.agent_votes_csv,
    )
    oof, fold_df, selective_fold_df, spec_df = run_cross_validation(
        feature_df,
        summary_cols,
        agent_cols,
        hybrid_cols,
        folds=args.folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    summary_df = summarize_oof(oof)
    selective_fold_summary_df = summarize_selective_folds(selective_fold_df)
    best_method = str(summary_df.iloc[0]["method"])
    selective_curve_df, selective_best_df = selective_summary(oof, best_method, coverage_targets)

    feature_df.to_csv(output_dir / "aggregation_feature_table.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(output_dir / "nested_cv_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(output_dir / "nested_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    selective_fold_df.to_csv(output_dir / "nested_cv_selective_fold_metrics.csv", index=False, encoding="utf-8-sig")
    selective_fold_summary_df.to_csv(
        output_dir / "nested_cv_selective_summary_by_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    spec_df.to_csv(output_dir / "nested_cv_selected_model_specs.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "nested_cv_method_summary.csv", index=False, encoding="utf-8-sig")
    selective_curve_df.to_csv(output_dir / "selective_prediction_curve.csv", index=False, encoding="utf-8-sig")
    selective_best_df.to_csv(output_dir / "selective_prediction_best_by_coverage.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "note_predictions_csv": str(args.note_predictions_csv),
        "agent_votes_csv": str(args.agent_votes_csv),
        "notes_total": int(len(feature_df)),
        "folds": int(args.folds),
        "inner_folds": int(args.inner_folds),
        "seed": int(args.seed),
        "summary_features": summary_cols,
        "agent_vote_features": int(len(agent_cols)),
        "best_method": best_method,
        "method_summary": summary_df.to_dict(orient="records"),
        "nested_cv_selective_summary_by_coverage": selective_fold_summary_df.to_dict(orient="records"),
        "posthoc_oof_selective_best_by_coverage": selective_best_df.to_dict(orient="records"),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
