from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
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
    p.add_argument("--objective", choices=["joint", "balanced", "accuracy"], default="joint")
    p.add_argument("--top-k-per-view", type=int, default=2)
    p.add_argument("--baseline-min-nmr-recall", type=float, default=0.87)
    p.add_argument("--gate-min-confidence", type=float, default=0.12)
    return p.parse_args()


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    recalls = []
    for label_id, label in LABEL.items():
        mask = y_true == label_id
        rec = float((y_pred[mask] == label_id).mean()) if mask.any() else math.nan
        out[f"recall_{label.lower()}"] = rec
        out[f"n_{label.lower()}"] = int(mask.sum())
        recalls.append(rec)
    out["min_recall"] = float(np.nanmin(recalls))
    out["h_to_nh"] = int(((y_true == 2) & (y_pred == 0)).sum())
    out["nh_to_h"] = int(((y_true == 0) & (y_pred == 2)).sum())
    out["cross_error"] = int(out["h_to_nh"] + out["nh_to_h"])
    return out


def score_key(m: dict[str, float | int], objective: str) -> tuple[float, ...]:
    joint = 2.0 * m["accuracy"] * m["balanced_accuracy"] / max(m["accuracy"] + m["balanced_accuracy"], 1e-12)
    if objective == "joint":
        return (joint, m["min_recall"], m["balanced_accuracy"], m["accuracy"])
    if objective == "balanced":
        return (m["balanced_accuracy"], joint, m["min_recall"], m["accuracy"])
    if objective == "accuracy":
        return (m["accuracy"], joint, m["balanced_accuracy"], m["min_recall"])
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


def make_model(kind: str, c: float | None = None, class_weight=None, seed: int = 42) -> Pipeline:
    if kind == "lr":
        assert c is not None
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
    if kind == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=6,
                        min_samples_leaf=3,
                        class_weight=class_weight,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind == "et":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=600,
                        max_depth=6,
                        min_samples_leaf=3,
                        class_weight=class_weight,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind == "gb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=220,
                        learning_rate=0.04,
                        max_depth=2,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(kind)


def align_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = p[:, i]
    return out


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    view: str
    kind: str
    c: float | None
    class_weight: object | None


def build_views(feature_sets: dict[str, list[str]], meta_cols: list[str]) -> dict[str, list[str]]:
    return {
        "summary": feature_sets["summary"],
        "full_agent_plus_metadata": feature_sets["full_agent_plus_metadata"],
        "summary_plus_meta": feature_sets["summary"] + meta_cols,
        "full_plus_meta": feature_sets["full_agent_plus_metadata"] + meta_cols,
    }


def candidate_pool() -> list[CandidateSpec]:
    w_alt = {0: 1.0, 1: 1.15, 2: 1.0}
    return [
        CandidateSpec("summary_lr_c03_bal", "summary", "lr", 0.3, "balanced"),
        CandidateSpec("summary_rf_bal", "summary", "rf", None, "balanced"),
        CandidateSpec("full_lr_c03_bal", "full_agent_plus_metadata", "lr", 0.3, "balanced"),
        CandidateSpec("full_rf_bal", "full_agent_plus_metadata", "rf", None, "balanced"),
        CandidateSpec("summary_meta_lr_c03_bal", "summary_plus_meta", "lr", 0.3, "balanced"),
        CandidateSpec("summary_meta_lr_c03_alt", "summary_plus_meta", "lr", 0.3, w_alt),
        CandidateSpec("summary_meta_rf_bal", "summary_plus_meta", "rf", None, "balanced"),
        CandidateSpec("full_meta_lr_c03_bal", "full_plus_meta", "lr", 0.3, "balanced"),
        CandidateSpec("full_meta_lr_c03_alt", "full_plus_meta", "lr", 0.3, w_alt),
        CandidateSpec("full_meta_rf_bal", "full_plus_meta", "rf", None, "balanced"),
    ]


def candidate_inner_oof(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    spec: CandidateSpec,
    inner: StratifiedKFold,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    oof_prob = np.zeros((len(y_train), 3), dtype=float)
    for itr, iva in inner.split(X_train, y_train):
        model = make_model(spec.kind, c=spec.c, class_weight=spec.class_weight, seed=seed)
        model.fit(X_train.iloc[itr], y_train[itr])
        oof_prob[iva] = align_prob(model, X_train.iloc[iva])
    pred = oof_prob.argmax(axis=1)
    return oof_prob, metric(y_train, pred)


def fit_predict_candidate(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    spec: CandidateSpec,
    seed: int,
) -> np.ndarray:
    model = make_model(spec.kind, c=spec.c, class_weight=spec.class_weight, seed=seed)
    model.fit(X_tr, y_tr)
    return align_prob(model, X_te)


def select_candidates(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    views: dict[str, list[str]],
    specs: list[CandidateSpec],
    inner_folds: int,
    seed: int,
    objective: str,
    top_k_per_view: int,
) -> tuple[list[dict], dict[str, dict[str, float | int]]]:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    per_spec_metrics: dict[str, dict[str, float | int]] = {}
    selected: list[dict] = []

    grouped: dict[str, list[tuple[tuple[float, ...], CandidateSpec, dict[str, float | int]]]] = {v: [] for v in views}
    for spec in specs:
        cols = views[spec.view]
        _, m = candidate_inner_oof(X_train[cols].reset_index(drop=True), y_train, spec, inner, seed)
        per_spec_metrics[spec.name] = m
        grouped[spec.view].append((score_key(m, objective), spec, m))

    for view_name, items in grouped.items():
        items.sort(key=lambda x: x[0], reverse=True)
        for rank, (_, spec, m) in enumerate(items[:top_k_per_view], start=1):
            selected.append(
                {
                    "name": spec.name,
                    "view": spec.view,
                    "kind": spec.kind,
                    "c": spec.c if spec.c is not None else "",
                    "class_weight": spec.class_weight if isinstance(spec.class_weight, str) else json.dumps(spec.class_weight) if spec.class_weight is not None else "none",
                    "inner_rank_in_view": rank,
                    **{f"inner_{k}": v for k, v in m.items()},
                }
            )

    return selected, per_spec_metrics


def blend_probs(weighted_specs: list[dict], probs_by_name: dict[str, np.ndarray], weight_by_name: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    names = [spec["name"] for spec in weighted_specs]
    weights = np.array([weight_by_name[n] for n in names], dtype=float)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()
    stack = np.stack([probs_by_name[n] for n in names], axis=0)
    weighted = np.tensordot(weights, stack, axes=(0, 0))
    uniform = stack.mean(axis=0)
    return weighted, uniform


def outer_ensemble(
    df: pd.DataFrame,
    views: dict[str, list[str]],
    specs: list[CandidateSpec],
    folds: int,
    inner_folds: int,
    seed: int,
    objective: str,
    top_k_per_view: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred_weighted = np.zeros(len(df), dtype=int)
    pred_uniform = np.zeros(len(df), dtype=int)
    best_prob = np.zeros((len(df), 3), dtype=float)
    fold_rows = []
    selected_rows = []

    for fold, (tr, te) in enumerate(outer.split(df, y), start=1):
        X_tr = df.iloc[tr].reset_index(drop=True)
        y_tr = y[tr]
        X_te = df.iloc[te]

        selected, per_spec_metrics = select_candidates(
            X_tr,
            y_tr,
            views=views,
            specs=specs,
            inner_folds=inner_folds,
            seed=seed + fold,
            objective=objective,
            top_k_per_view=top_k_per_view,
        )
        selected_rows.extend([{**r, "fold": fold} for r in selected])

        probs_by_name: dict[str, np.ndarray] = {}
        weight_by_name: dict[str, float] = {}
        for spec_row in selected:
            spec = next(s for s in specs if s.name == spec_row["name"])
            cols = views[spec.view]
            probs_by_name[spec.name] = fit_predict_candidate(
                X_tr[cols],
                y_tr,
                X_te[cols],
                spec,
                seed=seed + fold,
            )
            m = per_spec_metrics[spec.name]
            weight_by_name[spec.name] = max(0.01, score_key(m, objective)[0])

        weighted_prob, uniform_prob = blend_probs(selected, probs_by_name, weight_by_name)
        weighted_pred = weighted_prob.argmax(axis=1)
        uniform_pred = uniform_prob.argmax(axis=1)

        pred_weighted[te] = weighted_pred
        pred_uniform[te] = uniform_pred
        best_prob[te] = weighted_prob

        fold_rows.append(
            {
                "fold": fold,
                "selected_models": "|".join(r["name"] for r in selected),
                "selected_views": "|".join(r["view"] for r in selected),
                "selected_weights": "|".join(f"{weight_by_name[r['name']]:.4f}" for r in selected),
                **{f"test_weighted_{k}": v for k, v in metric(y[te], weighted_pred).items()},
                **{f"test_uniform_{k}": v for k, v in metric(y[te], uniform_pred).items()},
            }
        )

    return pred_weighted, pred_uniform, best_prob, fold_rows, selected_rows


def baseline_full_plus_meta_lr(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    X = df[feature_cols]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(df), dtype=int)
    prob = np.zeros((len(df), 3), dtype=float)
    rows = []
    c_grid = [0.3, 1.0, 3.0]
    class_weights: list[object | None] = [None, "balanced", {0: 1.0, 1: 1.15, 2: 1.0}]

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        X_tr = X.iloc[tr].reset_index(drop=True)
        y_tr = y[tr]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        for c in c_grid:
            for w in class_weights:
                oof = np.zeros(len(y_tr), dtype=int)
                for itr, iva in inner.split(X_tr, y_tr):
                    model = make_model("lr", c=c, class_weight=w, seed=seed + fold)
                    model.fit(X_tr.iloc[itr], y_tr[itr])
                    oof[iva] = model.predict(X_tr.iloc[iva])
                m = metric(y_tr, oof)
                key = score_key(m, "balanced")
                if best is None or key > best[0]:
                    best = (key, c, w, m)
        assert best is not None
        _, c, w, inner_m = best
        model = make_model("lr", c=c, class_weight=w, seed=seed + fold)
        model.fit(X.iloc[tr], y[tr])
        prob_te = align_prob(model, X.iloc[te])
        fold_pred = prob_te.argmax(axis=1)
        pred[te] = fold_pred
        prob[te] = prob_te
        rows.append(
            {
                "fold": fold,
                "c": c,
                "class_weight": w if isinstance(w, str) else json.dumps(w) if w is not None else "none",
                **{f"inner_{k}": v for k, v in inner_m.items()},
                **{f"test_{k}": v for k, v in metric(y[te], fold_pred).items()},
            }
        )
    return pred, prob, rows


def gate_predictions(
    baseline_pred: np.ndarray,
    baseline_prob: np.ndarray,
    ensemble_pred: np.ndarray,
    ensemble_prob: np.ndarray,
    min_nmr_recall: float,
    gate_min_confidence: float,
) -> np.ndarray:
    final = baseline_pred.copy()
    baseline_top = baseline_prob.max(axis=1)
    ensemble_top = ensemble_prob.max(axis=1)
    baseline_nmr = baseline_pred == 1
    ensemble_nmr = ensemble_pred == 1
    strong_agree_non_nmr = (baseline_pred != 1) & (ensemble_pred != 1) & (baseline_top >= gate_min_confidence) & (ensemble_top >= gate_min_confidence)
    strong_agree_non_nmr |= (baseline_pred != 1) & (ensemble_pred == baseline_pred) & (ensemble_top >= gate_min_confidence)
    final[strong_agree_non_nmr] = ensemble_pred[strong_agree_non_nmr]
    final[(baseline_nmr) & (ensemble_pred != 1) & (ensemble_top > baseline_top)] = ensemble_pred[(baseline_nmr) & (ensemble_pred != 1) & (ensemble_top > baseline_top)]
    return final


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "officialschema_oof_ensemble_gated_20260519"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    fast_oof = pd.read_csv(run_dir / "officialschema_nested_cv_fast_20260512" / "officialschema_nested_cv_fast_oof_predictions.csv")
    df = add_meta_features(df, fast_oof)

    meta_cols = [c for c in df.columns if c.startswith("nested_lr_")]
    views = build_views(feature_sets, meta_cols)
    specs = candidate_pool()

    y = df["true_label_3way"].to_numpy(dtype=int)
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    rows = []
    fold_tables = []

    base_pred, base_prob, base_rows = baseline_full_plus_meta_lr(
        df,
        feature_cols=views["full_plus_meta"],
        folds=args.folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    preds["baseline_full_plus_meta_lr_balanced"] = base_pred
    rows.append(
        {
            "method": "baseline_full_plus_meta_lr_balanced",
            "family": "baseline",
            "feature_set": "full_plus_meta",
            **metric(y, base_pred),
        }
    )
    fold_tables.append(pd.DataFrame(base_rows).assign(method="baseline_full_plus_meta_lr_balanced"))

    weighted_pred, uniform_pred, ensemble_prob, fold_rows, selected_rows = outer_ensemble(
        df,
        views=views,
        specs=specs,
        folds=args.folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        objective=args.objective,
        top_k_per_view=args.top_k_per_view,
    )
    gated_pred = gate_predictions(
        baseline_pred=base_pred,
        baseline_prob=base_prob,
        ensemble_pred=weighted_pred,
        ensemble_prob=ensemble_prob,
        min_nmr_recall=args.baseline_min_nmr_recall,
        gate_min_confidence=args.gate_min_confidence,
    )

    preds["oof_ensemble_weighted"] = weighted_pred
    preds["oof_ensemble_uniform"] = uniform_pred
    preds["oof_ensemble_gated"] = gated_pred

    rows.append(
        {
            "method": "oof_ensemble_weighted",
            "family": "ensemble",
            "feature_set": "multi_view_multi_learner",
            "objective": args.objective,
            "top_k_per_view": args.top_k_per_view,
            **metric(y, weighted_pred),
        }
    )
    rows.append(
        {
            "method": "oof_ensemble_uniform",
            "family": "ensemble",
            "feature_set": "multi_view_multi_learner",
            "objective": args.objective,
            "top_k_per_view": args.top_k_per_view,
            **metric(y, uniform_pred),
        }
    )
    rows.append(
        {
            "method": "oof_ensemble_gated",
            "family": "ensemble_gate",
            "feature_set": "baseline_plus_gated_ensemble",
            "objective": args.objective,
            "top_k_per_view": args.top_k_per_view,
            "baseline_min_nmr_recall": args.baseline_min_nmr_recall,
            "gate_min_confidence": args.gate_min_confidence,
            **metric(y, gated_pred),
        }
    )

    fold_tables.append(pd.DataFrame(fold_rows).assign(method="oof_ensemble_weighted"))
    fold_tables.append(pd.DataFrame(selected_rows).assign(method="oof_ensemble_weighted"))

    summary = pd.DataFrame(rows)
    for c in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful", "min_recall"]:
        summary[f"{c}_pct"] = pd.to_numeric(summary[c], errors="coerce") * 100.0
    summary = summary.sort_values(["accuracy", "balanced_accuracy", "min_recall"], ascending=[False, False, False])

    best = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(preds["true_label_text"], preds[best].map(LABEL), margins=True)

    preds.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    if fold_tables:
        pd.concat(fold_tables, ignore_index=True).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "n_notes": int(len(df)),
                "folds": int(args.folds),
                "inner_folds": int(args.inner_folds),
                "seed": int(args.seed),
                "objective": args.objective,
                "top_k_per_view": int(args.top_k_per_view),
                "baseline_min_nmr_recall": float(args.baseline_min_nmr_recall),
                "gate_min_confidence": float(args.gate_min_confidence),
                "best": summary.iloc[0].to_dict(),
                "feature_sets": {k: len(v) for k, v in views.items()},
                "candidate_pool": len(specs),
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
