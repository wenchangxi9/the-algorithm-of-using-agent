from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--folds", type=int, default=5)
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
        out[f"recall_{name.lower()}"] = float((pred[m] == k).mean())
    return out


def obj(m, mode):
    if mode == "balanced":
        return (m["balanced_accuracy"], m["accuracy"], -m["cross_error"])
    if mode == "cross_safe":
        return (-m["cross_error"], m["balanced_accuracy"], m["accuracy"])
    if mode == "hnh_average":
        return ((m["recall_helpful"] + m["recall_not_helpful"]) / 2, m["balanced_accuracy"], -m["cross_error"])
    if mode == "balanced_cross_penalty":
        return (m["balanced_accuracy"] - 0.004 * m["cross_error"], m["balanced_accuracy"], m["accuracy"])
    raise ValueError(mode)


def vote_map(row, methods, weights):
    score = np.zeros(3, dtype=float)
    for m, w in zip(methods, weights):
        val = int(row[m])
        score[val] += w
    return int(score.argmax())


def apply_combo(df, methods, weights, rule, margin):
    scores = np.zeros((len(df), 3), dtype=float)
    for m, w in zip(methods, weights):
        vals = df[m].to_numpy(int)
        for k in [0, 1, 2]:
            scores[:, k] += w * (vals == k)
    pred = scores.argmax(axis=1)
    if rule == "margin_to_nmr":
        h, nmr, nh = scores[:, 2], scores[:, 1], scores[:, 0]
        pred[np.abs(h - nh) <= margin] = 1
    if rule == "any_cross_to_nmr":
        h_votes = scores[:, 2] > 0
        nh_votes = scores[:, 0] > 0
        nmr_votes = scores[:, 1] > 0
        pred[h_votes & nh_votes & nmr_votes] = 1
    return pred


def tune(train, y, candidate_methods, mode):
    weight_grid = [0.5, 1.0, 2.0]
    rules = [("plain", 0.0), ("margin_to_nmr", 0.0), ("margin_to_nmr", 0.5), ("any_cross_to_nmr", 0.0)]
    best = None
    # Explore small 2-3 method ensembles. Larger searches are slow and overfit on 228 notes.
    from itertools import combinations, product
    for r in [2, 3]:
        for methods in combinations(candidate_methods, r):
            for weights in product(weight_grid, repeat=r):
                for rule, margin in rules:
                    pred = apply_combo(train, methods, weights, rule, margin)
                    m = metric(y, pred)
                    key = obj(m, mode)
                    if best is None or key > best[0]:
                        best = (key, methods, weights, rule, margin, m)
    return best[1:]


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "meta_blend_exploration_20260513"
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = run_dir / "aggregation_exploration_20260513" / "oof_predictions.csv"
    fast = run_dir / "officialschema_nested_cv_fast_20260512" / "officialschema_nested_cv_fast_oof_predictions.csv"
    strong = run_dir / "officialschema_strong_aggregation_20260513" / "strong_oof_predictions.csv"
    dfs = []
    base = pd.read_csv(exp, low_memory=False)
    base["noteId"] = base["noteId"].astype(str)
    keep = ["noteId", "true_label_3way", "true_label_text"] + [
        c for c in base.columns if c not in {"noteId", "true_label_3way", "true_label_text"} and not c.endswith("_pred_text")
    ]
    merged = base[keep].copy()
    if fast.exists():
        f = pd.read_csv(fast, low_memory=False)
        f["noteId"] = f["noteId"].astype(str)
        for c in ["nested_lr_summary", "nested_lr_full_agent", "raw_score_thresholds_0p33_0p67", "raw_vote_majority_somewhat_as_nmr"]:
            if c in f.columns:
                merged = merged.merge(f[["noteId", c]].rename(columns={c: f"fast_{c}"}), on="noteId", how="left")
    if strong.exists():
        s = pd.read_csv(strong, low_memory=False)
        s["noteId"] = s["noteId"].astype(str)
        for c in [x for x in s.columns if x.startswith("strong_nested_") or x in {"raw_score_033_067", "raw_vote_majority"}]:
            if not c.endswith("_pred_text") and not "_prob_" in c:
                merged = merged.merge(s[["noteId", c]].rename(columns={c: f"strong_{c}"}), on="noteId", how="left")

    preferred = [
        "agent_reliability_cross_safe",
        "lr_full_hnh_focus",
        "lr_compact_balanced",
        "rule_grid_cross_safe",
        "fast_nested_lr_summary",
        "fast_nested_lr_full_agent",
        "strong_strong_nested_crosssafe_balanced",
        "strong_strong_nested_balanced_crosssafe",
    ]
    candidate_methods = [
        c for c in preferred
        if c in merged.columns and set(merged[c].dropna().astype(int).unique()).issubset({0, 1, 2})
    ]
    y = merged["true_label_3way"].to_numpy(int)
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    preds = merged[["noteId", "true_label_3way", "true_label_text"]].copy()
    for mode in ["balanced", "cross_safe", "hnh_average", "balanced_cross_penalty"]:
        pred = np.zeros(len(merged), dtype=int)
        fold_rows = []
        for fold, (tr, te) in enumerate(outer.split(merged, y), start=1):
            methods, weights, rule, margin, im = tune(merged.iloc[tr], y[tr], candidate_methods, mode)
            p = apply_combo(merged.iloc[te], methods, weights, rule, margin)
            pred[te] = p
            fold_rows.append({
                "fold": fold,
                "mode": mode,
                "methods": "|".join(methods),
                "weights": "|".join(map(str, weights)),
                "rule": rule,
                "margin": margin,
                **{f"inner_{k}": v for k, v in im.items()},
                **{f"test_{k}": v for k, v in metric(y[te], p).items()},
            })
        name = f"meta_blend_{mode}"
        preds[name] = pred
        rows.append({"method": name, **metric(y, pred)})
        pd.DataFrame(fold_rows).to_csv(out_dir / f"{name}_folds.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    for c in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful"]:
        summary[f"{c}_pct"] = summary[c] * 100
    summary = summary.sort_values(["balanced_accuracy", "accuracy", "cross_error"], ascending=[False, False, True])
    best = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(merged["true_label_text"], pd.Series(preds[best]).map(LABEL), margins=True)
    merged.to_csv(out_dir / "candidate_predictions.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "meta_blend_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "meta_blend_summary.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "candidate_methods": candidate_methods,
        "best": summary.iloc[0].to_dict(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = [
        "method", "accuracy_pct", "balanced_accuracy_pct",
        "recall_not_helpful_pct", "recall_needs_more_ratings_pct", "recall_helpful_pct",
        "h_to_nh", "nh_to_h", "cross_error",
    ]
    print(summary[cols].to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())


if __name__ == "__main__":
    main()
