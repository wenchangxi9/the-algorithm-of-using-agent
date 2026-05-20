from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_officialschema_nested_cv_aggregation_2000 import build_features


LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("artifacts/05_agent_runs_2000"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--acc-drop-max", type=float, default=0.004)
    p.add_argument("--min-nmr-recall", type=float, default=0.85)
    p.add_argument("--min-change-rate", type=float, default=0.002)
    return p.parse_args()


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    recs = []
    for label_id, label in LABEL.items():
        mask = y_true == label_id
        rec = float((y_pred[mask] == label_id).mean()) if mask.any() else math.nan
        out[f"recall_{label.lower()}"] = rec
        out[f"n_{label.lower()}"] = int(mask.sum())
        recs.append(rec)
    out["min_recall"] = float(np.nanmin(recs))
    out["h_to_nh"] = int(((y_true == 2) & (y_pred == 0)).sum())
    out["nh_to_h"] = int(((y_true == 0) & (y_pred == 2)).sum())
    out["cross_error"] = int(out["h_to_nh"] + out["nh_to_h"])
    return out


def score_key(m: dict[str, float | int], baseline: dict[str, float | int], objective: str = "balanced") -> tuple[float, ...]:
    joint = 2.0 * m["accuracy"] * m["balanced_accuracy"] / max(m["accuracy"] + m["balanced_accuracy"], 1e-12)
    acc_delta = m["accuracy"] - baseline["accuracy"]
    bal_delta = m["balanced_accuracy"] - baseline["balanced_accuracy"]
    if objective == "balanced":
        return (
            bal_delta,
            acc_delta,
            m["balanced_accuracy"],
            m["accuracy"],
            m["recall_not_helpful"],
            m["recall_helpful"],
            joint,
        )
    if objective == "accuracy":
        return (
            acc_delta,
            bal_delta,
            m["accuracy"],
            m["balanced_accuracy"],
            m["recall_not_helpful"],
            m["recall_helpful"],
            joint,
        )
    raise ValueError(objective)


def add_meta_features(df: pd.DataFrame, fast_oof: pd.DataFrame) -> pd.DataFrame:
    fast_oof = fast_oof.copy()
    fast_oof["noteId"] = fast_oof["noteId"].astype(str)
    keep = [
        "noteId",
        "nested_lr_summary_prob_not_helpful",
        "nested_lr_summary_prob_nmr",
        "nested_lr_summary_prob_helpful",
        "nested_lr_full_agent_prob_not_helpful",
        "nested_lr_full_agent_prob_nmr",
        "nested_lr_full_agent_prob_helpful",
    ]
    fast_oof = fast_oof[[c for c in keep if c in fast_oof.columns]]

    df = df.copy()
    df["noteId"] = df["noteId"].astype(str)
    df = df.merge(fast_oof, on="noteId", how="left")

    for prefix in ["nested_lr_summary", "nested_lr_full_agent"]:
        nh = df[f"{prefix}_prob_not_helpful"].astype(float)
        nmr = df[f"{prefix}_prob_nmr"].astype(float)
        h = df[f"{prefix}_prob_helpful"].astype(float)
        probs = np.vstack([nh.to_numpy(), nmr.to_numpy(), h.to_numpy()]).T
        safe = np.clip(probs, 1e-9, 1.0)
        df[f"{prefix}_entropy"] = -(safe * np.log(safe)).sum(axis=1)
        df[f"{prefix}_margin"] = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
        df[f"{prefix}_resolved_mass"] = nh + h
        df[f"{prefix}_signed_margin"] = h - nh
        df[f"{prefix}_nmr_gap"] = nmr - np.maximum(nh, h)
    return df


def build_views(feature_sets: dict[str, list[str]], meta_cols: list[str]) -> dict[str, list[str]]:
    return {
        "summary": feature_sets["summary"],
        "summary_plus_meta": feature_sets["summary"] + meta_cols,
        "full_plus_meta": feature_sets["full_agent_plus_metadata"] + meta_cols,
    }


def make_model(c: float, class_weight, seed: int) -> Pipeline:
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
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def align_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), p.shape[1]), dtype=float)
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = p[:, i]
    return out


def align_binary_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    if p.shape[1] == 1:
        return np.zeros(len(X), dtype=float)
    idx = int(np.where(model.named_steps["clf"].classes_ == 1)[0][0]) if hasattr(model.named_steps["clf"], "classes_") else 1
    return p[:, idx]


@dataclass(frozen=True)
class Spec:
    view: str
    c: float
    class_weight: object | None


def pick_binary_spec(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    views: dict[str, list[str]],
    specs: list[Spec],
    inner_folds: int,
    seed: int,
    objective: str,
) -> tuple[Spec, dict[str, float | int], np.ndarray]:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best = None
    best_oof = None
    for spec in specs:
        cols = views[spec.view]
        oof_prob = np.zeros(len(y_train), dtype=float)
        for itr, iva in inner.split(X_train, y_train):
            model = make_model(spec.c, spec.class_weight, seed)
            model.fit(X_train.iloc[itr][cols], y_train[itr])
            oof_prob[iva] = align_binary_prob(model, X_train.iloc[iva][cols])
        pred = (oof_prob >= 0.5).astype(int)
        m = metric(y_train, pred)
        if best is None:
            best = (spec, m)
            best_oof = oof_prob
            continue
        key = score_key(m, best[1], objective)
        ref_key = score_key(best[1], m, objective)
        if key > ref_key:
            best = (spec, m)
            best_oof = oof_prob
    assert best is not None and best_oof is not None
    return best[0], best[1], best_oof


def pick_multiclass_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cols: list[str],
    inner_folds: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, float | int], np.ndarray]:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    c_grid = [0.3]
    weights = ["balanced"]
    best = None
    best_oof = None
    for c in c_grid:
        for w in weights:
            oof_prob = np.zeros((len(y_train), 3), dtype=float)
            for itr, iva in inner.split(X_train, y_train):
                model = make_model(c, w, seed)
                model.fit(X_train.iloc[itr][cols], y_train[itr])
                oof_prob[iva] = align_prob(model, X_train.iloc[iva][cols])
            pred = oof_prob.argmax(axis=1)
            m = metric(y_train, pred)
            if best is None or score_key(m, m, "balanced") > score_key(best[1], best[1], "balanced"):
                best = ({"c": c, "class_weight": w}, m)
                best_oof = oof_prob
    assert best is not None and best_oof is not None
    return best[0], best[1], best_oof


def search_gate(
    y_train: np.ndarray,
    baseline_prob: np.ndarray,
    baseline_pred: np.ndarray,
    resolved_prob: np.ndarray,
    direction_prob: np.ndarray,
    baseline_metrics: dict[str, float | int],
    min_nmr_recall: float,
    acc_drop_max: float,
    min_change_rate: float,
) -> tuple[dict[str, float], dict[str, float | int], np.ndarray]:
    t_resolved_grid = np.linspace(0.35, 0.80, 10)
    t_margin_grid = np.linspace(-0.10, 0.24, 9)
    t_conf_grid = np.linspace(0.50, 0.80, 7)
    best = None
    best_pred = None
    baseline_acc_floor = baseline_metrics["accuracy"] - acc_drop_max
    baseline_nmr = baseline_prob[:, 1]
    baseline_nmr_pred = baseline_pred == 1
    direction_label = (direction_prob >= 0.5).astype(int) * 2
    direction_label = np.where(direction_prob >= 0.5, 2, 0)
    direction_conf = np.maximum(direction_prob, 1.0 - direction_prob)

    for t_resolved in t_resolved_grid:
        for t_margin in t_margin_grid:
            for t_conf in t_conf_grid:
                pred = baseline_pred.copy()
                rescue = (
                    baseline_nmr_pred
                    & (resolved_prob >= t_resolved)
                    & ((resolved_prob - baseline_nmr) >= t_margin)
                    & (direction_conf >= t_conf)
                )
                if rescue.mean() < min_change_rate:
                    continue
                pred[rescue] = direction_label[rescue]
                m = metric(y_train, pred)
                if m["accuracy"] < baseline_acc_floor:
                    continue
                if m["recall_needs_more_ratings"] < min_nmr_recall:
                    continue
                key = (
                    m["balanced_accuracy"],
                    m["accuracy"],
                    m["recall_not_helpful"],
                    m["recall_helpful"],
                    m["min_recall"],
                )
                if best is None or key > best[0]:
                    best = (
                        key,
                        {
                            "t_resolved": float(t_resolved),
                            "t_margin": float(t_margin),
                            "t_conf": float(t_conf),
                            "baseline_acc_floor": float(baseline_acc_floor),
                        },
                        m,
                    )
                    best_pred = pred

    if best is None:
        pred = baseline_pred.copy()
        m = baseline_metrics
        best = (
            (
                m["balanced_accuracy"],
                m["accuracy"],
                m["recall_not_helpful"],
                m["recall_helpful"],
                m["min_recall"],
            ),
            {
                "t_resolved": 1.1,
                "t_margin": 1.1,
                "t_conf": 1.1,
                "baseline_acc_floor": float(baseline_acc_floor),
            },
            m,
        )
        best_pred = pred

    assert best_pred is not None
    return best[1], best[2], best_pred


def fit_predict_multiclass(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    cols: list[str],
    c: float,
    class_weight,
    seed: int,
) -> np.ndarray:
    model = make_model(c, class_weight, seed)
    model.fit(X_tr[cols], y_tr)
    return align_prob(model, X_te[cols])


def fit_predict_binary(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    cols: list[str],
    c: float,
    class_weight,
    seed: int,
) -> np.ndarray:
    model = make_model(c, class_weight, seed)
    model.fit(X_tr[cols], y_tr)
    return align_binary_prob(model, X_te[cols])


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "officialschema_xstyle_rescue_gate_20260519"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    fast_oof = pd.read_csv(run_dir / "officialschema_nested_cv_fast_20260512" / "officialschema_nested_cv_fast_oof_predictions.csv")
    df = add_meta_features(df, fast_oof)
    meta_cols = [c for c in df.columns if c.startswith("nested_lr_")]
    views = build_views(feature_sets, meta_cols)

    y = df["true_label_3way"].to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    baseline_cols = views["full_plus_meta"]
    baseline_pred = np.zeros(len(df), dtype=int)
    baseline_prob = np.zeros((len(df), 3), dtype=float)
    rescue_pred = np.zeros(len(df), dtype=int)
    rescue_prob = np.zeros(len(df), dtype=float)
    direction_prob = np.zeros(len(df), dtype=float)
    rows = []

    # Candidate pools are intentionally small to keep the search conservative and stable.
    binary_specs = [
        Spec("summary_plus_meta", 0.3, "balanced"),
        Spec("summary_plus_meta", 1.0, "balanced"),
        Spec("full_plus_meta", 0.3, "balanced"),
        Spec("full_plus_meta", 1.0, "balanced"),
        Spec("full_plus_meta", 0.3, {0: 1.0, 1: 1.20}),
    ]

    for fold, (tr, te) in enumerate(outer.split(df, y), start=1):
        X_tr = df.iloc[tr].reset_index(drop=True)
        y_tr = y[tr]
        X_te = df.iloc[te].reset_index(drop=True)

        inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold)

        # Baseline anchor: the best known full_plus_meta multiclass LR.
        base_oof_prob = np.zeros((len(y_tr), 3), dtype=float)
        for itr, iva in inner.split(X_tr, y_tr):
            model = make_model(0.3, "balanced", seed=args.seed + fold)
            model.fit(X_tr.iloc[itr][baseline_cols], y_tr[itr])
            base_oof_prob[iva] = align_prob(model, X_tr.iloc[iva][baseline_cols])
        base_oof_pred = base_oof_prob.argmax(axis=1)
        base_inner_m = metric(y_tr, base_oof_pred)

        # Rescue head 1: predicts whether the note is resolved (NH/H) vs NMR.
        resolved_y = (y_tr != 1).astype(int)
        resolved_spec, resolved_inner_m, resolved_oof = pick_binary_spec(
            X_tr,
            resolved_y,
            views=views,
            specs=binary_specs,
            inner_folds=args.inner_folds,
            seed=args.seed + 11 + fold,
            objective="balanced",
        )

        # Rescue head 2: predicts H vs NH on resolved examples only.
        resolved_mask = y_tr != 1
        direction_y = (y_tr[resolved_mask] == 2).astype(int)
        direction_specs = [
            Spec("summary_plus_meta", 0.3, "balanced"),
            Spec("summary_plus_meta", 1.0, "balanced"),
            Spec("full_plus_meta", 0.3, "balanced"),
            Spec("full_plus_meta", 1.0, "balanced"),
            Spec("full_plus_meta", 0.3, {0: 1.0, 1: 1.20}),
        ]
        direction_spec, direction_inner_m, direction_oof_resolved = pick_binary_spec(
            X_tr.iloc[resolved_mask].reset_index(drop=True),
            direction_y,
            views=views,
            specs=direction_specs,
            inner_folds=max(2, min(args.inner_folds, int(direction_y.sum() > 0) + 2)),
            seed=args.seed + 23 + fold,
            objective="balanced",
        )

        # Build OOF-style probabilities on the training slice for gate tuning.
        direction_oof = np.full(len(y_tr), 0.5, dtype=float)
        direction_oof[resolved_mask] = direction_oof_resolved

        gate_spec, gate_inner_m, gate_oof_pred = search_gate(
            y_train=y_tr,
            baseline_prob=base_oof_prob,
            baseline_pred=base_oof_pred,
            resolved_prob=resolved_oof,
            direction_prob=direction_oof,
            baseline_metrics=base_inner_m,
            min_nmr_recall=args.min_nmr_recall,
            acc_drop_max=args.acc_drop_max,
            min_change_rate=args.min_change_rate,
        )

        # Fit final fold models on the full outer training set.
        base_model = make_model(0.3, "balanced", seed=args.seed + fold)
        base_model.fit(X_tr[baseline_cols], y_tr)
        base_prob_te = align_prob(base_model, X_te[baseline_cols])
        base_pred_te = base_prob_te.argmax(axis=1)

        resolved_model = make_model(resolved_spec.c, resolved_spec.class_weight, seed=args.seed + 101 + fold)
        resolved_model.fit(X_tr[views[resolved_spec.view]], (y_tr != 1).astype(int))
        resolved_prob_te = align_binary_prob(resolved_model, X_te[views[resolved_spec.view]])

        direction_model = make_model(direction_spec.c, direction_spec.class_weight, seed=args.seed + 202 + fold)
        direction_model.fit(X_tr[views[direction_spec.view]][resolved_mask], (y_tr[resolved_mask] == 2).astype(int))
        direction_prob_te = align_binary_prob(direction_model, X_te[views[direction_spec.view]])

        baseline_nmr = base_prob_te[:, 1]
        direction_label_te = np.where(direction_prob_te >= 0.5, 2, 0)
        direction_conf_te = np.maximum(direction_prob_te, 1.0 - direction_prob_te)
        rescue_mask_te = (
            (base_pred_te == 1)
            & (resolved_prob_te >= gate_spec["t_resolved"])
            & ((resolved_prob_te - baseline_nmr) >= gate_spec["t_margin"])
            & (direction_conf_te >= gate_spec["t_conf"])
        )

        final_te = base_pred_te.copy()
        final_te[rescue_mask_te] = direction_label_te[rescue_mask_te]

        baseline_pred[te] = base_pred_te
        baseline_prob[te] = base_prob_te
        rescue_pred[te] = final_te
        rescue_prob[te] = resolved_prob_te
        direction_prob[te] = direction_prob_te

        rows.append(
            {
                "fold": fold,
                "baseline_cols": len(baseline_cols),
                "resolved_view": resolved_spec.view,
                "resolved_c": resolved_spec.c,
                "resolved_class_weight": resolved_spec.class_weight if isinstance(resolved_spec.class_weight, str) else json.dumps(resolved_spec.class_weight),
                "direction_view": direction_spec.view,
                "direction_c": direction_spec.c,
                "direction_class_weight": direction_spec.class_weight if isinstance(direction_spec.class_weight, str) else json.dumps(direction_spec.class_weight),
                **{f"inner_baseline_{k}": v for k, v in base_inner_m.items()},
                **{f"inner_resolved_{k}": v for k, v in resolved_inner_m.items()},
                **{f"inner_direction_{k}": v for k, v in direction_inner_m.items()},
                **{f"inner_gate_{k}": v for k, v in gate_inner_m.items()},
                **{f"test_baseline_{k}": v for k, v in metric(y[te], base_pred_te).items()},
                **{f"test_rescue_{k}": v for k, v in metric(y[te], final_te).items()},
            }
        )

    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    preds["baseline_full_plus_meta_lr_balanced"] = baseline_pred
    preds["xstyle_rescue_gate"] = rescue_pred

    summary_rows = [
        {
            "method": "baseline_full_plus_meta_lr_balanced",
            "family": "baseline",
            "feature_set": "full_plus_meta",
            **metric(y, baseline_pred),
        },
        {
            "method": "xstyle_rescue_gate",
            "family": "xstyle_gate",
            "feature_set": "baseline_plus_rescue_heads",
            **metric(y, rescue_pred),
        },
    ]
    summary = pd.DataFrame(summary_rows)
    for c in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful", "min_recall"]:
        summary[f"{c}_pct"] = pd.to_numeric(summary[c], errors="coerce") * 100.0
    summary = summary.sort_values(["accuracy", "balanced_accuracy", "min_recall"], ascending=[False, False, False])

    best = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(preds["true_label_text"], preds[best].map(LABEL), margins=True)

    preds.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    pd.DataFrame(
        {
            "noteId": preds["noteId"],
            "baseline_pred": baseline_pred,
            "rescue_pred": rescue_pred,
            "changed": baseline_pred != rescue_pred,
        }
    ).to_csv(out_dir / "prediction_changes.csv", index=False, encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "n_notes": int(len(df)),
                "folds": int(args.folds),
                "inner_folds": int(args.inner_folds),
                "seed": int(args.seed),
                "acc_drop_max": float(args.acc_drop_max),
                "min_nmr_recall": float(args.min_nmr_recall),
                "min_change_rate": float(args.min_change_rate),
                "best": summary.iloc[0].to_dict(),
                "feature_sets": {k: len(v) for k, v in views.items()},
                "baseline_cols": len(baseline_cols),
                "resolved_spec": {
                    "view": resolved_spec.view,
                    "c": resolved_spec.c,
                    "class_weight": resolved_spec.class_weight if isinstance(resolved_spec.class_weight, str) else json.dumps(resolved_spec.class_weight),
                },
                "direction_spec": {
                    "view": direction_spec.view,
                    "c": direction_spec.c,
                    "class_weight": direction_spec.class_weight if isinstance(direction_spec.class_weight, str) else json.dumps(direction_spec.class_weight),
                },
                "gate_spec": gate_spec,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary[[
        "method",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "recall_not_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_helpful_pct",
    ]].to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
