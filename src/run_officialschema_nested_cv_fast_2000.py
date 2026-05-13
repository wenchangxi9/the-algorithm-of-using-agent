from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_officialschema_nested_cv_aggregation_2000 import INT_TO_LABEL, build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/data6/wenchangxi/community_note/analysis/llm_16agent_rawrating_2000_20260512_officialschema"),
    )
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
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }
    for label_id, label in INT_TO_LABEL.items():
        mask = y_true == label_id
        out[f"recall_{label.lower()}"] = float((y_pred[mask] == label_id).mean()) if mask.any() else np.nan
        out[f"n_{label.lower()}"] = int(mask.sum())
    return out


def choose_spec(X_train: pd.DataFrame, y_train: np.ndarray, inner_folds: int, seed: int) -> tuple[float, str | None, dict]:
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weights: list[str | None] = [None, "balanced"]
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best = None
    for c in c_grid:
        for weight in weights:
            oof = np.zeros(len(y_train), dtype=int)
            for tr, va in inner.split(X_train, y_train):
                model = make_model(c, weight)
                model.fit(X_train.iloc[tr], y_train[tr])
                oof[va] = model.predict(X_train.iloc[va])
            m = metric_row(y_train, oof)
            key = (m["accuracy"], m["balanced_accuracy"])
            if best is None or key > best[0]:
                best = (key, c, weight, m)
    return best[1], best[2], best[3]


def nested_cv(df: pd.DataFrame, feature_cols: list[str], folds: int, inner_folds: int, seed: int) -> tuple[pd.DataFrame, list[dict], dict]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    X = df[feature_cols]
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros((len(df), 3), dtype=float)
    fold_rows = []
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        c, weight, inner_metrics = choose_spec(X.iloc[tr].reset_index(drop=True), y[tr], inner_folds, seed + fold)
        model = make_model(c, weight)
        model.fit(X.iloc[tr], y[tr])
        fold_pred = model.predict(X.iloc[te])
        fold_prob = model.predict_proba(X.iloc[te])
        pred[te] = fold_pred
        aligned = np.zeros((len(te), 3), dtype=float)
        for pos, cls in enumerate(model.named_steps["clf"].classes_):
            aligned[:, int(cls)] = fold_prob[:, pos]
        prob[te] = aligned
        fold_rows.append(
            {
                "fold": fold,
                "c": c,
                "class_weight": weight or "none",
                **{f"inner_{k}": v for k, v in inner_metrics.items()},
                **{f"test_{k}": v for k, v in metric_row(y[te], fold_pred).items()},
            }
        )
    oof = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    oof["pred_label_3way"] = pred
    oof["pred_label_text"] = oof["pred_label_3way"].map(INT_TO_LABEL)
    oof["prob_not_helpful"] = prob[:, 0]
    oof["prob_need_more_ratings"] = prob[:, 1]
    oof["prob_helpful"] = prob[:, 2]
    return oof, fold_rows, metric_row(y, pred)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "officialschema_nested_cv_fast_20260512"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    rows = []
    all_fold_rows = []
    all_predictions = df[["noteId", "true_label_3way", "true_label_text"]].copy()

    # Raw baselines.
    raw_majority = df[["vote_not_helpful", "vote_somewhat_helpful", "vote_helpful"]].to_numpy(dtype=float).argmax(axis=1)
    baselines = {
        "always_nmr": np.full(len(df), 1, dtype=int),
        "raw_vote_majority_somewhat_as_nmr": raw_majority,
        "raw_score_thresholds_0p33_0p67": np.where(
            df["mean_raw_score"].to_numpy() >= 2 / 3,
            2,
            np.where(df["mean_raw_score"].to_numpy() <= 1 / 3, 0, 1),
        ),
    }
    y = df["true_label_3way"].to_numpy(dtype=int)
    for name, pred in baselines.items():
        row = {"method": name, "family": "baseline", "n_features": 0, **metric_row(y, pred)}
        rows.append(row)
        all_predictions[name] = pred

    for feature_set_name in ["summary", "full_agent"]:
        cols = feature_sets[feature_set_name]
        oof, fold_rows, m = nested_cv(df, cols, args.folds, args.inner_folds, args.seed)
        method = f"nested_lr_{feature_set_name}"
        rows.append({"method": method, "family": "nested_lr", "feature_set": feature_set_name, "n_features": len(cols), **m})
        all_predictions[method] = oof["pred_label_3way"].to_numpy()
        all_predictions[f"{method}_pred_text"] = oof["pred_label_text"].to_numpy()
        all_predictions[f"{method}_prob_not_helpful"] = oof["prob_not_helpful"].to_numpy()
        all_predictions[f"{method}_prob_nmr"] = oof["prob_need_more_ratings"].to_numpy()
        all_predictions[f"{method}_prob_helpful"] = oof["prob_helpful"].to_numpy()
        for row in fold_rows:
            row["method"] = method
            all_fold_rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["accuracy", "balanced_accuracy"], ascending=False)
    for col in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful"]:
        summary[f"{col}_pct"] = pd.to_numeric(summary[col], errors="coerce") * 100.0
    summary.to_csv(out_dir / "officialschema_nested_cv_fast_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_fold_rows).to_csv(out_dir / "officialschema_nested_cv_fast_folds.csv", index=False, encoding="utf-8-sig")
    all_predictions.to_csv(out_dir / "officialschema_nested_cv_fast_oof_predictions.csv", index=False, encoding="utf-8-sig")

    best_method = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(
        all_predictions["true_label_text"],
        all_predictions[best_method].map(INT_TO_LABEL) if best_method in all_predictions else all_predictions[f"{best_method}_pred_text"],
        margins=True,
    )
    confusion.to_csv(out_dir / "best_method_confusion.csv", encoding="utf-8-sig")
    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_notes": int(len(df)),
        "folds": int(args.folds),
        "inner_folds": int(args.inner_folds),
        "true_distribution": df["true_label_text"].value_counts().to_dict(),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "best_method": best_method,
        "best": summary.iloc[0].to_dict(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "method",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "recall_not_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_helpful_pct",
        "n_not_helpful",
        "n_needs_more_ratings",
        "n_helpful",
    ]
    print(summary[cols].to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
