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


def make_model(c, weight):
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


def decide(prob, tau_h, tau_nh, tau_nmr, mode):
    h = prob[:, 2]
    nh = prob[:, 0]
    nmr = prob[:, 1]
    pred = np.full(len(prob), 1, dtype=int)
    if mode == "helpful_first":
        pred[h >= tau_h] = 2
        pred[(pred == 1) & (nh >= tau_nh)] = 0
        pred[(pred == 1) & (nmr >= tau_nmr)] = 1
    elif mode == "not_helpful_first":
        pred[nh >= tau_nh] = 0
        pred[(pred == 1) & (h >= tau_h)] = 2
        pred[(pred == 1) & (nmr >= tau_nmr)] = 1
    elif mode == "max_with_helpful_bias":
        score = np.vstack([nh, nmr, h + tau_h]).T
        pred = score.argmax(axis=1)
    else:
        raise ValueError(mode)
    return pred


def objective(m, target_h, mode):
    # Keep the requested helpful recall first, then maximize other quality.
    h_gap = -abs(m["recall_helpful"] - target_h) if m["recall_helpful"] >= target_h else -10 + m["recall_helpful"]
    if mode == "target_h_balanced":
        return (h_gap, m["balanced_accuracy"], m["accuracy"], -m["nh_to_h"])
    if mode == "target_h_resolved":
        resolved_avg = (m["recall_helpful"] + m["recall_not_helpful"]) / 2
        return (h_gap, resolved_avg, -m["nh_to_h"], m["balanced_accuracy"])
    if mode == "target_h_safe":
        return (h_gap, -m["nh_to_h"], -m["cross_error"], m["balanced_accuracy"])
    raise ValueError(mode)


def tune_inner(X, y, inner_folds, seed, target_h, obj_mode):
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    weights = [None, "balanced"]
    tau_h_grid = np.arange(0.12, 0.61, 0.04)
    tau_nh_grid = np.arange(0.24, 0.71, 0.06)
    tau_nmr_grid = [0.20, 0.30, 0.40, 0.50]
    bias_grid = np.arange(0.00, 0.61, 0.05)
    modes = ["helpful_first", "not_helpful_first", "max_with_helpful_bias"]
    best = None
    for c in c_grid:
        for w in weights:
            oof = np.zeros((len(y), 3), dtype=float)
            for tr, va in inner.split(X, y):
                model = make_model(c, w)
                model.fit(X.iloc[tr], y[tr])
                oof[va] = align_prob(model, X.iloc[va])
            for mode in modes:
                if mode == "max_with_helpful_bias":
                    for b in bias_grid:
                        pred = decide(oof, b, 0.0, 0.0, mode)
                        m = metric(y, pred)
                        key = objective(m, target_h, obj_mode)
                        if best is None or key > best[0]:
                            best = (key, c, w, mode, float(b), 0.0, 0.0, m)
                else:
                    for th in tau_h_grid:
                        for tnh in tau_nh_grid:
                            for tnmr in tau_nmr_grid:
                                pred = decide(oof, float(th), float(tnh), float(tnmr), mode)
                                m = metric(y, pred)
                                key = objective(m, target_h, obj_mode)
                                if best is None or key > best[0]:
                                    best = (key, c, w, mode, float(th), float(tnh), float(tnmr), m)
    return best[1:]


def nested(df, features, folds, inner_folds, seed, target_h, obj_mode):
    y = df["true_label_3way"].to_numpy(int)
    X = df[features]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        c, w, mode, th, tnh, tnmr, im = tune_inner(
            X.iloc[tr].reset_index(drop=True), y[tr], inner_folds, seed + fold, target_h, obj_mode
        )
        model = make_model(c, w)
        model.fit(X.iloc[tr], y[tr])
        prob = align_prob(model, X.iloc[te])
        pred[te] = decide(prob, th, tnh, tnmr, mode)
        rows.append({
            "fold": fold,
            "target_h": target_h,
            "obj_mode": obj_mode,
            "c": c,
            "weight": w or "none",
            "decision_mode": mode,
            "tau_h_or_bias": th,
            "tau_nh": tnh,
            "tau_nmr": tnmr,
            **{f"inner_{k}": v for k, v in im.items()},
            **{f"test_{k}": v for k, v in metric(y[te], pred[te]).items()},
        })
    return pred, rows


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "helpful_target_threshold_20260513"
    out_dir.mkdir(parents=True, exist_ok=True)
    votes = load_votes(run_dir)
    df = build_features(votes)
    non = {"noteId", "tweetId", "true_label_3way", "true_label_text"}
    compact = [c for c in df.columns if c not in non and not c.startswith("agent_label__")]
    full = [c for c in df.columns if c not in non]
    y = df["true_label_3way"].to_numpy(int)
    rows = []
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    fold_tables = []
    for fs_name, features in [("compact", compact), ("full", full)]:
        for target_h in [0.70, 0.80, 0.90]:
            for obj_mode in ["target_h_balanced", "target_h_resolved", "target_h_safe"]:
                name = f"helpful_target_{fs_name}_{target_h:.2f}_{obj_mode}"
                print("running", name, flush=True)
                pred, folds = nested(df, features, args.folds, args.inner_folds, args.seed, target_h, obj_mode)
                preds[name] = pred
                rows.append({"method": name, "feature_set": fs_name, "target_h": target_h, "obj_mode": obj_mode, **metric(y, pred)})
                fold_tables.append(pd.DataFrame(folds).assign(method=name))
                print("done", name, metric(y, pred), flush=True)
    summary = pd.DataFrame(rows)
    for c in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful"]:
        summary[f"{c}_pct"] = summary[c] * 100
    summary = summary.sort_values(["recall_helpful", "balanced_accuracy", "accuracy"], ascending=False)
    best = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(df["true_label_text"], pd.Series(preds[best]).map(LABEL), margins=True)
    preds.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_tables, ignore_index=True).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(json.dumps({"best": summary.iloc[0].to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["method", "accuracy_pct", "balanced_accuracy_pct", "recall_not_helpful_pct", "recall_needs_more_ratings_pct", "recall_helpful_pct", "h_to_nh", "nh_to_h", "cross_error"]
    print(summary[cols].head(20).to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())


if __name__ == "__main__":
    main()
