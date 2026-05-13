from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL3 = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}
RAW = {"NOT_HELPFUL": 0, "HELPFUL": 1}
HELPFUL_REASONS = [
    "helpfulClear",
    "helpfulGoodSources",
    "helpfulAddressesClaim",
    "helpfulImportantContext",
    "helpfulUnbiasedLanguage",
]
NOT_HELPFUL_REASONS = [
    "notHelpfulIncorrect",
    "notHelpfulSourcesMissingOrUnreliable",
    "notHelpfulMissingKeyPoints",
    "notHelpfulHardToUnderstand",
    "notHelpfulArgumentativeOrBiased",
    "notHelpfulIrrelevantSources",
    "notHelpfulOpinionSpeculation",
    "notHelpfulNoteNotNeeded",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260513)
    return p.parse_args()


def clean01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1)


def load_features(run_dir: Path) -> pd.DataFrame:
    votes = pd.read_csv(run_dir / "agent_votes.csv", low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["agent_id"] = votes["agent_id"].astype(str)
    votes["raw_binary"] = votes["parsed_rating"].map(RAW)
    votes = votes[votes["raw_binary"].isin([0, 1])].copy()
    votes["true_label_3way"] = pd.to_numeric(votes["true_label_3way"], errors="coerce").astype(int)
    votes["score"] = pd.to_numeric(votes["predicted_rating_score"], errors="coerce")
    votes["confidence"] = pd.to_numeric(votes.get("confidence", 0), errors="coerce")
    votes["changes_reader_understanding"] = pd.to_numeric(
        votes.get("changes_reader_understanding", 0), errors="coerce"
    )
    for c in HELPFUL_REASONS + NOT_HELPFUL_REASONS:
        if c not in votes.columns:
            votes[c] = 0
        votes[c] = clean01(votes[c])
    votes["is_h"] = (votes["raw_binary"] == 1).astype(float)
    votes["is_nh"] = (votes["raw_binary"] == 0).astype(float)
    votes["helpful_reason_sum"] = votes[HELPFUL_REASONS].sum(axis=1)
    votes["not_helpful_reason_sum"] = votes[NOT_HELPFUL_REASONS].sum(axis=1)

    agg = {
        "true_label_3way": ("true_label_3way", "first"),
        "true_label_text": ("true_label_text", "first"),
        "n_votes": ("agent_id", "size"),
        "vote_h": ("is_h", "sum"),
        "vote_nh": ("is_nh", "sum"),
        "mean_score": ("score", "mean"),
        "std_score": ("score", "std"),
        "mean_confidence": ("confidence", "mean"),
        "std_confidence": ("confidence", "std"),
        "mean_understanding": ("changes_reader_understanding", "mean"),
        "std_understanding": ("changes_reader_understanding", "std"),
        "helpful_reason_sum": ("helpful_reason_sum", "mean"),
        "not_helpful_reason_sum": ("not_helpful_reason_sum", "mean"),
    }
    for c in HELPFUL_REASONS + NOT_HELPFUL_REASONS:
        agg[f"{c}_rate"] = (c, "mean")
    df = votes.groupby("noteId", as_index=False).agg(**agg)
    for c in ["std_score", "std_confidence", "std_understanding"]:
        df[c] = df[c].fillna(0)
    df["share_h"] = df["vote_h"] / df["n_votes"].clip(lower=1)
    df["share_nh"] = df["vote_nh"] / df["n_votes"].clip(lower=1)
    df["h_minus_nh"] = df["share_h"] - df["share_nh"]
    df["h_nh_margin"] = df["h_minus_nh"].abs()
    df["vote_entropy_binary"] = -(
        df["share_h"].clip(1e-9, 1) * np.log(df["share_h"].clip(1e-9, 1))
        + df["share_nh"].clip(1e-9, 1) * np.log(df["share_nh"].clip(1e-9, 1))
    )

    agent = votes.pivot_table(index="noteId", columns="agent_id", values="raw_binary", aggfunc="first")
    agent.columns = [f"agent_{c}_is_h" for c in agent.columns]
    df = df.merge(agent.reset_index(), on="noteId", how="left")
    return df.sort_values("noteId").reset_index(drop=True)


def model(c: float, weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight=weight, max_iter=5000, random_state=42)),
        ]
    )


def p_helpful(clf: Pipeline, x: pd.DataFrame) -> np.ndarray:
    classes = list(clf.named_steps["clf"].classes_)
    prob = clf.predict_proba(x)
    return prob[:, classes.index(1)]


def apply_dual_threshold(p: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where(p <= low, 0, np.where(p >= high, 2, 1))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "resolved_coverage": float(np.isin(pred, [0, 2]).mean()),
        "h_to_nh": int(((y == 2) & (pred == 0)).sum()),
        "nh_to_h": int(((y == 0) & (pred == 2)).sum()),
    }
    for k, name in LABEL3.items():
        mask = y == k
        out[f"recall_{name.lower()}"] = float((pred[mask] == k).mean())
        out[f"n_{name.lower()}"] = int(mask.sum())
    return out


def tune_thresholds(y: np.ndarray, p: np.ndarray) -> tuple[float, float, dict[str, float | int]]:
    best = None
    for low in np.arange(0.05, 0.56, 0.02):
        for high in np.arange(max(0.45, low + 0.04), 0.96, 0.02):
            pred = apply_dual_threshold(p, float(low), float(high))
            m = metrics(y, pred)
            key = (
                float(m["balanced_accuracy"]),
                float(m["accuracy"]),
                -abs(float(m["resolved_coverage"]) - 2 / 3),
                -int(m["h_to_nh"]) - int(m["nh_to_h"]),
            )
            if best is None or key > best[0]:
                best = (key, float(low), float(high), m)
    assert best is not None
    return best[1], best[2], best[3]


def nested_dual_threshold(df: pd.DataFrame, features: list[str], folds: int, inner_folds: int, seed: int):
    y = df["true_label_3way"].to_numpy(int)
    x = df[features]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    rows = []
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weights = [None, "balanced"]

    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        xtr = x.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        for c in c_grid:
            for w in weights:
                oof = np.zeros(len(ytr), dtype=float)
                ok = True
                for a, b in inner.split(xtr, ytr):
                    resolved = np.isin(ytr[a], [0, 2])
                    if resolved.sum() < 4 or len(np.unique(ytr[a][resolved])) < 2:
                        ok = False
                        break
                    ybin = (ytr[a][resolved] == 2).astype(int)
                    m = model(c, w)
                    m.fit(xtr.iloc[a].iloc[resolved], ybin)
                    oof[b] = p_helpful(m, xtr.iloc[b])
                if not ok:
                    continue
                low, high, inner_metrics = tune_thresholds(ytr, oof)
                key = (
                    float(inner_metrics["balanced_accuracy"]),
                    float(inner_metrics["accuracy"]),
                    -abs(float(inner_metrics["resolved_coverage"]) - 2 / 3),
                    -int(inner_metrics["h_to_nh"]) - int(inner_metrics["nh_to_h"]),
                )
                if best is None or key > best[0]:
                    best = (key, c, w, low, high, inner_metrics)

        assert best is not None
        _, c, w, low, high, inner_metrics = best
        resolved_outer = np.isin(y[tr], [0, 2])
        ybin_outer = (y[tr][resolved_outer] == 2).astype(int)
        m = model(c, w)
        m.fit(x.iloc[tr].iloc[resolved_outer], ybin_outer)
        pte = p_helpful(m, x.iloc[te])
        fold_pred = apply_dual_threshold(pte, low, high)
        pred[te] = fold_pred
        prob[te] = pte
        test_metrics = metrics(y[te], fold_pred)
        rows.append(
            {
                "fold": fold,
                "c": c,
                "class_weight": w or "none",
                "low_threshold": low,
                "high_threshold": high,
                **{f"inner_{k}": v for k, v in inner_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
    return pred, prob, pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out = run_dir / "binary_agent_tristate_dual_threshold_eval_20260513"
    out.mkdir(exist_ok=True)
    df = load_features(run_dir)
    non = {"noteId", "true_label_3way", "true_label_text"}
    full_features = [c for c in df.columns if c not in non]
    pred, prob, folds = nested_dual_threshold(df, full_features, args.folds, args.inner_folds, args.seed)
    y = df["true_label_3way"].to_numpy(int)
    summary = pd.DataFrame([{**metrics(y, pred), "method": "binary_agent_nested_lr_full_dual_threshold", "n_features": len(full_features)}])
    for c in [
        "accuracy",
        "balanced_accuracy",
        "resolved_coverage",
        "recall_not_helpful",
        "recall_needs_more_ratings",
        "recall_helpful",
    ]:
        summary[f"{c}_pct"] = summary[c] * 100
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    preds["p_helpful"] = prob
    preds["pred_label_3way"] = pred
    preds["pred_label_text"] = pd.Series(pred).map(LABEL3)
    cm = pd.DataFrame(
        confusion_matrix(y, pred, labels=[2, 1, 0]),
        index=["true_HELPFUL", "true_NMR", "true_NOT_HELPFUL"],
        columns=["pred_HELPFUL", "pred_NMR", "pred_NOT_HELPFUL"],
    )
    folds.to_csv(out / "fold_thresholds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    cm.to_csv(out / "confusion.csv", encoding="utf-8-sig")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out),
                "note": "Binary agent outputs are converted to tri-state note status by a nested-CV binary LR p(Helpful) model and dual thresholds: p<=low -> NOT_HELPFUL, low<p<high -> NEEDS_MORE_RATINGS, p>=high -> HELPFUL.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== Fold thresholds ===")
    print(folds[["fold", "c", "class_weight", "low_threshold", "high_threshold", "inner_accuracy", "inner_balanced_accuracy", "inner_resolved_coverage", "test_accuracy", "test_balanced_accuracy", "test_resolved_coverage"]].to_string(index=False))
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Confusion ===")
    print(cm.to_string())
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
