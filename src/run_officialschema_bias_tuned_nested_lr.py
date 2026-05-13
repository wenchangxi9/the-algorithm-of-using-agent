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

from run_officialschema_aggregation_exploration import build_features, load_votes


LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260513)
    return p.parse_args()


def metric(y, pred):
    out = {
        "accuracy": float((y == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "h_to_nh": int(((y == 2) & (pred == 0)).sum()),
        "nh_to_h": int(((y == 0) & (pred == 2)).sum()),
    }
    out["cross_error"] = out["h_to_nh"] + out["nh_to_h"]
    for k, name in LABEL.items():
        m = y == k
        out[f"recall_{name.lower()}"] = float((pred[m] == k).mean()) if m.any() else np.nan
        out[f"n_{name.lower()}"] = int(m.sum())
    return out


def key(m, objective):
    if objective == "balanced":
        return (m["balanced_accuracy"], m["accuracy"], -m["cross_error"])
    if objective == "hnh_average":
        return ((m["recall_helpful"] + m["recall_not_helpful"]) / 2, m["balanced_accuracy"], -m["cross_error"])
    if objective == "balanced_cross_penalty":
        return (m["balanced_accuracy"] - 0.0035 * m["cross_error"], m["balanced_accuracy"], m["accuracy"])
    if objective == "min_resolved":
        return (min(m["recall_helpful"], m["recall_not_helpful"]), m["balanced_accuracy"], -m["cross_error"])
    if objective == "cross_safe":
        return (-m["cross_error"], m["balanced_accuracy"], m["accuracy"])
    raise ValueError(objective)


def make_lr(c, weight):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=c, class_weight=weight, max_iter=5000, solver="lbfgs", random_state=42)),
    ])


def align_prob(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = p[:, i]
    return out


def decide(prob, bias, nmr_band, cross_margin):
    score = np.log(np.clip(prob, 1e-9, 1.0)) + np.asarray(bias)[None, :]
    pred = score.argmax(axis=1)
    h_score = score[:, 2]
    nh_score = score[:, 0]
    nmr_score = score[:, 1]
    # Optional safety: if H and NH are very close and NMR is competitive, abstain as NMR.
    if cross_margin > 0:
        pred[(np.abs(h_score - nh_score) < cross_margin) & (nmr_score >= np.minimum(h_score, nh_score) - nmr_band)] = 1
    return pred


def tune_inner(X, y, inner_folds, seed, objective):
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weights = [None, "balanced"]
    # Biases are log-prior adjustments. Negative NMR bias encourages decisive H/NH.
    bias_grid = []
    for b_nh in [-0.3, 0.0, 0.2, 0.4, 0.6]:
        for b_nmr in [-0.8, -0.5, -0.3, 0.0, 0.2]:
            for b_h in [-0.3, 0.0, 0.2, 0.4, 0.6]:
                bias_grid.append((b_nh, b_nmr, b_h))
    safety_grid = [(0.0, 0.0), (0.1, 0.2), (0.2, 0.2), (0.2, 0.4), (0.3, 0.4)]
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best = None
    for c in c_grid:
        for w in weights:
            oof = np.zeros((len(y), 3), dtype=float)
            for tr, va in inner.split(X, y):
                model = make_lr(c, w)
                model.fit(X.iloc[tr], y[tr])
                oof[va] = align_prob(model, X.iloc[va])
            for bias in bias_grid:
                for nmr_band, cross_margin in safety_grid:
                    pred = decide(oof, bias, nmr_band, cross_margin)
                    m = metric(y, pred)
                    k = key(m, objective)
                    if best is None or k > best[0]:
                        best = (k, c, w, bias, nmr_band, cross_margin, m)
    return best[1:]


def nested(df, feature_cols, folds, inner_folds, seed, objective):
    y = df["true_label_3way"].to_numpy(int)
    X = df[feature_cols]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    fold_rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        c, w, bias, nmr_band, cross_margin, inner_m = tune_inner(
            X.iloc[tr].reset_index(drop=True), y[tr], inner_folds, seed + fold, objective
        )
        model = make_lr(c, w)
        model.fit(X.iloc[tr], y[tr])
        prob = align_prob(model, X.iloc[te])
        pred[te] = decide(prob, bias, nmr_band, cross_margin)
        fold_rows.append({
            "fold": fold,
            "objective": objective,
            "c": c,
            "weight": w or "none",
            "bias_nh": bias[0],
            "bias_nmr": bias[1],
            "bias_h": bias[2],
            "nmr_band": nmr_band,
            "cross_margin": cross_margin,
            **{f"inner_{k}": v for k, v in inner_m.items()},
            **{f"test_{k}": v for k, v in metric(y[te], pred[te]).items()},
        })
    return pred, fold_rows


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "bias_tuned_nested_lr_20260513"
    out_dir.mkdir(parents=True, exist_ok=True)
    votes = load_votes(run_dir)
    df = build_features(votes)
    non_features = {"noteId", "tweetId", "true_label_3way", "true_label_text"}
    compact = [c for c in df.columns if c not in non_features and not c.startswith("agent_label__")]
    full = [c for c in df.columns if c not in non_features]
    y = df["true_label_3way"].to_numpy(int)
    rows = []
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    fold_tables = []
    configs = [
        ("compact", compact),
        ("full", full),
    ]
    for fs_name, cols in configs:
        for objective in ["balanced", "hnh_average", "balanced_cross_penalty", "min_resolved", "cross_safe"]:
            name = f"bias_lr_{fs_name}_{objective}"
            print("running", name, flush=True)
            pred, folds = nested(df, cols, args.folds, args.inner_folds, args.seed, objective)
            preds[name] = pred
            rows.append({"method": name, "feature_set": fs_name, "n_features": len(cols), **metric(y, pred)})
            fold_tables.append(pd.DataFrame(folds).assign(method=name))
            print("done", name, metric(y, pred), flush=True)
    summary = pd.DataFrame(rows)
    for c in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful"]:
        summary[f"{c}_pct"] = summary[c] * 100
    summary = summary.sort_values(["balanced_accuracy", "accuracy", "cross_error"], ascending=[False, False, True])
    best = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(df["true_label_text"], pd.Series(preds[best]).map(LABEL), margins=True)
    preds.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_tables, ignore_index=True).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "best": summary.iloc[0].to_dict(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["method", "accuracy_pct", "balanced_accuracy_pct", "recall_not_helpful_pct", "recall_needs_more_ratings_pct", "recall_helpful_pct", "h_to_nh", "nh_to_h", "cross_error"]
    print(summary[cols].to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())


if __name__ == "__main__":
    main()
