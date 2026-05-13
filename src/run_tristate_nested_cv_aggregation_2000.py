from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_TO_INT = {"NOT_HELPFUL": 0, "NEEDS_MORE_RATINGS": 1, "HELPFUL": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
RESOLVED_LABELS = {0, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/data6/wenchangxi/community_note/analysis/llm_16agent_tristate_2000_20260512"),
    )
    parser.add_argument("--target-coverage", type=float, default=0.1125)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_features(agent_votes_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    votes = pd.read_csv(agent_votes_csv, low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["predicted_label_3way"] = pd.to_numeric(votes["predicted_label_3way"], errors="coerce")
    votes = votes[votes["predicted_label_3way"].isin([0, 1, 2])].copy()

    numeric_cols = [
        "confidence",
        "addresses_core_claim",
        "changes_reader_understanding",
        "note_needed",
        "evidence_strength",
        "misses_key_points",
        "too_minor_or_tangential",
    ]
    for col in numeric_cols:
        votes[col] = pd.to_numeric(votes[col], errors="coerce")

    base = (
        votes.groupby("noteId", as_index=False)
        .agg(
            tweetId=("tweetId", "first"),
            currentStatus=("currentStatus", "first"),
            true_label_3way=("true_label_3way", "first"),
            true_label_text=("true_label_text", "first"),
            n_votes=("agent_id", "size"),
            mean_confidence=("confidence", "mean"),
            std_confidence=("confidence", "std"),
            mean_addresses_core_claim=("addresses_core_claim", "mean"),
            mean_changes_reader_understanding=("changes_reader_understanding", "mean"),
            mean_note_needed=("note_needed", "mean"),
            mean_evidence_strength=("evidence_strength", "mean"),
            misses_key_points_rate=("misses_key_points", "mean"),
            too_minor_or_tangential_rate=("too_minor_or_tangential", "mean"),
        )
    )
    base["std_confidence"] = base["std_confidence"].fillna(0.0)
    base["true_label_3way"] = pd.to_numeric(base["true_label_3way"], errors="coerce").astype(int)

    counts = (
        votes.pivot_table(
            index="noteId",
            columns="predicted_label_3way",
            values="agent_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns={0: "vote_not_helpful", 1: "vote_need_more_ratings", 2: "vote_helpful"})
        .reset_index()
    )
    for col in ["vote_not_helpful", "vote_need_more_ratings", "vote_helpful"]:
        if col not in counts.columns:
            counts[col] = 0
    feature_df = base.merge(counts, on="noteId", how="left")
    for col in ["vote_not_helpful", "vote_need_more_ratings", "vote_helpful"]:
        feature_df[col] = feature_df[col].fillna(0).astype(float)
        feature_df[col.replace("vote_", "share_")] = feature_df[col] / feature_df["n_votes"]

    helpful = feature_df["share_helpful"].clip(1e-6, 1 - 1e-6)
    not_helpful = feature_df["share_not_helpful"].clip(1e-6, 1 - 1e-6)
    nmr = feature_df["share_need_more_ratings"].clip(1e-6, 1 - 1e-6)
    shares = np.vstack([not_helpful.to_numpy(), nmr.to_numpy(), helpful.to_numpy()]).T
    feature_df["vote_entropy_3way"] = -(shares * np.log(shares)).sum(axis=1)
    feature_df["resolved_vote_share"] = feature_df["share_helpful"] + feature_df["share_not_helpful"]
    feature_df["resolved_vote_margin"] = (feature_df["share_helpful"] - feature_df["share_not_helpful"]).abs()
    feature_df["quality_signal_mean"] = feature_df[
        [
            "mean_addresses_core_claim",
            "mean_changes_reader_understanding",
            "mean_note_needed",
            "mean_evidence_strength",
        ]
    ].mean(axis=1)
    feature_df["failure_signal_mean"] = feature_df[
        ["misses_key_points_rate", "too_minor_or_tangential_rate"]
    ].mean(axis=1)

    # Per-agent tri-state votes preserve who made which judgment.
    pivot = votes.pivot_table(
        index="noteId",
        columns="agent_id",
        values="predicted_label_3way",
        aggfunc="mean",
    )
    pivot = pivot.add_prefix("agent_vote_3way__").reset_index()
    feature_df = feature_df.merge(pivot, on="noteId", how="left")

    non_features = {"noteId", "tweetId", "currentStatus", "true_label_3way", "true_label_text"}
    feature_cols = [col for col in feature_df.columns if col not in non_features]
    return feature_df.sort_values("noteId").reset_index(drop=True), feature_cols


def full_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    total = len(y_true)
    out: dict[str, float | int] = {
        "accuracy": float((y_true == pred).mean()) if total else 0.0,
        "n": int(total),
    }
    for label_id, label in INT_TO_LABEL.items():
        mask = y_true == label_id
        out[f"recall_{label.lower()}"] = float((pred[mask] == label_id).mean()) if mask.any() else math.nan
        out[f"n_{label.lower()}"] = int(mask.sum())
    return out


def macro_recall(metrics: dict[str, float | int]) -> float:
    return float(
        np.nanmean(
            [
                metrics["recall_not_helpful"],
                metrics["recall_needs_more_ratings"],
                metrics["recall_helpful"],
            ]
        )
    )


def resolved_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    mask = pred >= 0
    resolved = int(mask.sum())
    total = len(y_true)
    if resolved == 0:
        return {
            "coverage": 0.0,
            "resolved_notes": 0,
            "resolved_accuracy_3way": None,
            "resolved_binary_accuracy_on_true_resolved": None,
            "true_resolved_coverage": 0.0,
        }
    correct_3way = y_true[mask] == pred[mask]
    true_resolved_mask = np.isin(y_true, list(RESOLVED_LABELS)) & mask
    binary_acc = (
        float((y_true[true_resolved_mask] == pred[true_resolved_mask]).mean())
        if true_resolved_mask.any()
        else None
    )
    return {
        "coverage": float(resolved / total),
        "resolved_notes": resolved,
        "resolved_accuracy_3way": float(correct_3way.mean()),
        "resolved_binary_accuracy_on_true_resolved": binary_acc,
        "true_resolved_coverage": float(true_resolved_mask.sum() / max(1, np.isin(y_true, list(RESOLVED_LABELS)).sum())),
        "true_resolved_notes_predicted_resolved": int(true_resolved_mask.sum()),
    }


@dataclass(frozen=True)
class Spec:
    c: float
    class_weight: str | None


def score_for_selection(prob: np.ndarray) -> np.ndarray:
    # Higher score means the model is more confident that the note is resolved
    # as either Helpful or Not Helpful. NMR probability is implicitly penalized.
    return np.maximum(prob[:, 0], prob[:, 2])


def resolved_pred_from_prob(prob: np.ndarray, threshold: float) -> np.ndarray:
    pred = np.full(len(prob), -1, dtype=int)
    label = np.where(prob[:, 2] >= prob[:, 0], 2, 0)
    mask = score_for_selection(prob) >= threshold
    pred[mask] = label[mask]
    return pred


def choose_threshold_by_target(y_true: np.ndarray, prob: np.ndarray, target_coverage: float) -> tuple[float, dict]:
    scores = score_for_selection(prob)
    candidates = sorted(set(float(x) for x in scores), reverse=True)
    if not candidates:
        return 1.0, {}
    best: tuple[tuple[float, float, float], float, dict] | None = None
    for threshold in candidates:
        pred = resolved_pred_from_prob(prob, threshold)
        m = resolved_metrics(y_true, pred)
        coverage = float(m["coverage"])
        # Prefer configurations close to the target coverage, then maximize resolved accuracy.
        closeness = -abs(coverage - target_coverage)
        acc = -1.0 if m["resolved_accuracy_3way"] is None else float(m["resolved_accuracy_3way"])
        binary_acc = (
            -1.0
            if m["resolved_binary_accuracy_on_true_resolved"] is None
            else float(m["resolved_binary_accuracy_on_true_resolved"])
        )
        key = (closeness, acc, binary_acc)
        if best is None or key > best[0]:
            best = (key, threshold, m)
    assert best is not None
    return best[1], best[2]


def choose_threshold_topk(y_true: np.ndarray, prob: np.ndarray, target_coverage: float) -> tuple[float, dict]:
    scores = score_for_selection(prob)
    if len(scores) == 0:
        return 1.0, {}
    k = max(1, int(round(target_coverage * len(scores))))
    ordered = np.sort(scores)[::-1]
    threshold = float(ordered[min(k - 1, len(ordered) - 1)])
    pred = resolved_pred_from_prob(prob, threshold)
    return threshold, resolved_metrics(y_true, pred)


def choose_model_spec(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    feature_cols: list[str],
    inner_folds: int,
    seed: int,
    target_coverage: float,
) -> tuple[Spec, np.ndarray, dict, float, dict]:
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    class_weights: list[str | None] = [None, "balanced"]
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    X_train = X.iloc[train_idx][feature_cols]
    y_train = y[train_idx]

    best: tuple[tuple[float, float, float, float], Spec, np.ndarray, dict, float, dict] | None = None
    for c in c_grid:
        for class_weight in class_weights:
            oof_prob = np.zeros((len(train_idx), 3), dtype=float)
            for inner_train_pos, inner_val_pos in inner_cv.split(X_train, y_train):
                model = make_model(c, class_weight)
                model.fit(X_train.iloc[inner_train_pos], y_train[inner_train_pos])
                oof_prob[inner_val_pos] = model.predict_proba(X_train.iloc[inner_val_pos])
            full_pred = oof_prob.argmax(axis=1)
            m = full_metrics(y_train, full_pred)
            threshold, resolved = choose_threshold_topk(y_train, oof_prob, target_coverage)
            balanced = np.nanmean(
                [
                    m["recall_not_helpful"],
                    m["recall_needs_more_ratings"],
                    m["recall_helpful"],
                ]
            )
            resolved_acc = (
                -1.0
                if resolved.get("resolved_accuracy_3way") is None
                else float(resolved["resolved_accuracy_3way"])
            )
            binary_acc = (
                -1.0
                if resolved.get("resolved_binary_accuracy_on_true_resolved") is None
                else float(resolved["resolved_binary_accuracy_on_true_resolved"])
            )
            coverage = float(resolved.get("coverage", 0.0))
            key = (
                resolved_acc,
                binary_acc,
                -abs(coverage - target_coverage),
                float(balanced),
            )
            spec = Spec(c=c, class_weight=class_weight)
            if best is None or key > best[0]:
                best = (key, spec, oof_prob, m, threshold, resolved)
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "nested_cv_tristate_aggregation_target_0p1125"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_df, feature_cols = build_features(run_dir / "agent_votes.csv")
    y = feature_df["true_label_3way"].to_numpy(dtype=int)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof = feature_df[["noteId", "true_label_3way", "true_label_text"]].copy()
    for col in [
        "prob_not_helpful",
        "prob_need_more_ratings",
        "prob_helpful",
        "full_pred",
        "resolved_pred",
        "resolved_score",
    ]:
        oof[col] = np.nan
    fold_rows = []
    spec_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(feature_df, y), start=1):
        spec, inner_oof_prob, inner_full, threshold, inner_resolved = choose_model_spec(
            feature_df,
            y,
            train_idx,
            feature_cols,
            inner_folds=args.inner_folds,
            seed=args.seed + fold,
            target_coverage=args.target_coverage,
        )
        model = make_model(spec.c, spec.class_weight)
        model.fit(feature_df.iloc[train_idx][feature_cols], y[train_idx])
        prob = model.predict_proba(feature_df.iloc[test_idx][feature_cols])
        full_pred = prob.argmax(axis=1)
        resolved_pred = resolved_pred_from_prob(prob, threshold)

        oof.loc[test_idx, "prob_not_helpful"] = prob[:, 0]
        oof.loc[test_idx, "prob_need_more_ratings"] = prob[:, 1]
        oof.loc[test_idx, "prob_helpful"] = prob[:, 2]
        oof.loc[test_idx, "full_pred"] = full_pred
        oof.loc[test_idx, "resolved_pred"] = resolved_pred
        oof.loc[test_idx, "resolved_score"] = score_for_selection(prob)

        row = {
            "fold": fold,
            "c": spec.c,
            "class_weight": spec.class_weight or "none",
            "threshold": threshold,
            "inner_full_accuracy": inner_full["accuracy"],
            "inner_resolved_coverage": inner_resolved.get("coverage"),
            "inner_resolved_accuracy_3way": inner_resolved.get("resolved_accuracy_3way"),
            **{f"test_full_{k}": v for k, v in full_metrics(y[test_idx], full_pred).items()},
            **{f"test_resolved_{k}": v for k, v in resolved_metrics(y[test_idx], resolved_pred).items()},
        }
        fold_rows.append(row)
        spec_rows.append(
            {
                "fold": fold,
                "c": spec.c,
                "class_weight": spec.class_weight or "none",
                "threshold": threshold,
                "n_features": len(feature_cols),
                "features": json.dumps(feature_cols, ensure_ascii=False),
            }
        )

    oof["full_pred"] = oof["full_pred"].astype(int)
    oof["resolved_pred"] = oof["resolved_pred"].astype(int)
    oof["status_pred"] = oof["resolved_pred"].where(oof["resolved_pred"] >= 0, 1).astype(int)
    oof["full_pred_text"] = oof["full_pred"].map(INT_TO_LABEL)
    oof["resolved_pred_text"] = oof["resolved_pred"].map(INT_TO_LABEL).fillna("UNRESOLVED")
    oof["status_pred_text"] = oof["status_pred"].map(INT_TO_LABEL)

    full = full_metrics(y, oof["full_pred"].to_numpy(dtype=int))
    full["macro_recall"] = macro_recall(full)
    status = full_metrics(y, oof["status_pred"].to_numpy(dtype=int))
    status["macro_recall"] = macro_recall(status)
    resolved = resolved_metrics(y, oof["resolved_pred"].to_numpy(dtype=int))

    confusion_full = pd.crosstab(
        oof["true_label_text"], oof["full_pred_text"], margins=True
    )
    confusion_status = pd.crosstab(
        oof["true_label_text"], oof["status_pred_text"], margins=True
    )
    confusion_resolved = pd.crosstab(
        oof["true_label_text"], oof["resolved_pred_text"], margins=True
    )
    true_dist = oof["true_label_text"].value_counts().to_dict()
    pred_resolved_dist = oof["resolved_pred_text"].value_counts().to_dict()

    metadata = {
        "run_dir": str(run_dir),
        "n_notes": int(len(oof)),
        "n_features": int(len(feature_cols)),
        "folds": int(args.folds),
        "inner_folds": int(args.inner_folds),
        "seed": int(args.seed),
        "target_coverage": float(args.target_coverage),
        "true_resolved_coverage": float(np.isin(y, list(RESOLVED_LABELS)).mean()),
        "full_metrics": full,
        "status_3way_metrics_unresolved_as_nmr": status,
        "resolved_metrics": resolved,
        "true_distribution": true_dist,
        "resolved_prediction_distribution": pred_resolved_dist,
        "confusion_full": confusion_full.to_dict(),
        "confusion_status_unresolved_as_nmr": confusion_status.to_dict(),
        "confusion_resolved": confusion_resolved.to_dict(),
        "fold_metrics": fold_rows,
    }

    feature_df.to_csv(out_dir / "aggregation_feature_table.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(out_dir / "nested_cv_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out_dir / "nested_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(spec_rows).to_csv(out_dir / "nested_cv_selected_specs.csv", index=False, encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
