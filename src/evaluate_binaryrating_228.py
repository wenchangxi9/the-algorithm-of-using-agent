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
    votes["true_binary"] = votes["true_label_3way"].map({0: 0, 2: 1})
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

    agent = votes.pivot_table(index="noteId", columns="agent_id", values="raw_binary", aggfunc="first")
    agent.columns = [f"agent_{c}_is_h" for c in agent.columns]
    df = df.merge(agent.reset_index(), on="noteId", how="left")
    return df.sort_values("noteId").reset_index(drop=True)


def fit_lr(c: float, weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight=weight, max_iter=5000, random_state=42)),
        ]
    )


def nested_binary(df: pd.DataFrame, features: list[str], seed: int, folds: int) -> np.ndarray:
    mask = df["true_label_3way"].isin([0, 2]).to_numpy()
    y = df.loc[mask, "true_label_3way"].map({0: 0, 2: 1}).to_numpy(int)
    x = df.loc[mask, features]
    pred = np.zeros(len(y), dtype=int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weights = [None, "balanced"]
    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + fold)
        best = None
        xtr = x.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        for c in c_grid:
            for w in weights:
                oof = np.zeros(len(ytr), dtype=float)
                for a, b in inner.split(xtr, ytr):
                    model = fit_lr(c, w)
                    model.fit(xtr.iloc[a], ytr[a])
                    classes = list(model.named_steps["clf"].classes_)
                    prob = model.predict_proba(xtr.iloc[b])
                    h_idx = classes.index(1)
                    oof[b] = prob[:, h_idx]
                for t in np.arange(0.20, 0.81, 0.02):
                    p = (oof >= t).astype(int)
                    key = (balanced_accuracy_score(ytr, p), accuracy_score(ytr, p))
                    if best is None or key > best[0]:
                        best = (key, c, w, float(t))
        _, c, w, t = best
        model = fit_lr(c, w)
        model.fit(x.iloc[tr], y[tr])
        classes = list(model.named_steps["clf"].classes_)
        prob = model.predict_proba(x.iloc[te])[:, classes.index(1)]
        pred[te] = (prob >= t).astype(int)
    return mask, pred


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out = run_dir / "binary_aggregation_eval_20260513"
    out.mkdir(exist_ok=True)
    df = load_features(run_dir)
    non = {"noteId", "true_label_3way", "true_label_text"}
    vote_features = ["n_votes", "vote_h", "vote_nh", "mean_score", "std_score", "share_h", "share_nh", "h_minus_nh", "h_nh_margin"]
    full_features = [c for c in df.columns if c not in non]

    y3 = df["true_label_3way"].to_numpy(int)
    majority3 = np.where(df["share_h"].to_numpy() >= 0.5, 2, 0)
    rows = []
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    for name, p3 in {"raw_majority_binary": majority3}.items():
        mask = np.isin(y3, [0, 2])
        pb = (p3[mask] == 2).astype(int)
        yb = (y3[mask] == 2).astype(int)
        rows.append(
            {
                "method": name,
                "full_3way_accuracy": float((p3 == y3).mean()),
                "binary_resolved_accuracy": float(accuracy_score(yb, pb)),
                "binary_balanced_accuracy": float(balanced_accuracy_score(yb, pb)),
                "helpful_recall": float(((pb == 1) & (yb == 1)).sum() / max((yb == 1).sum(), 1)),
                "not_helpful_recall": float(((pb == 0) & (yb == 0)).sum() / max((yb == 0).sum(), 1)),
                "n_binary_eval": int(mask.sum()),
                "n_total": int(len(y3)),
            }
        )
        preds[name] = p3

    for name, features in {"nested_lr_vote": vote_features, "nested_lr_full": full_features}.items():
        mask, pred_bin = nested_binary(df, features, args.seed + len(rows), args.folds)
        p3 = np.full(len(df), -1, dtype=int)
        p3[mask] = np.where(pred_bin == 1, 2, 0)
        yb = (df.loc[mask, "true_label_3way"].to_numpy(int) == 2).astype(int)
        rows.append(
            {
                "method": name,
                "full_3way_accuracy": float((p3 == y3).mean()),
                "binary_resolved_accuracy": float(accuracy_score(yb, pred_bin)),
                "binary_balanced_accuracy": float(balanced_accuracy_score(yb, pred_bin)),
                "helpful_recall": float(((pred_bin == 1) & (yb == 1)).sum() / max((yb == 1).sum(), 1)),
                "not_helpful_recall": float(((pred_bin == 0) & (yb == 0)).sum() / max((yb == 0).sum(), 1)),
                "n_binary_eval": int(mask.sum()),
                "n_total": int(len(y3)),
            }
        )
        preds[name] = p3

    summary = pd.DataFrame(rows)
    for c in ["full_3way_accuracy", "binary_resolved_accuracy", "binary_balanced_accuracy", "helpful_recall", "not_helpful_recall"]:
        summary[f"{c}_pct"] = summary[c] * 100
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    confs = {}
    for method in preds.columns:
        if method in {"noteId", "true_label_3way", "true_label_text"}:
            continue
        mask = np.isin(y3, [0, 2])
        cm = confusion_matrix(y3[mask], preds.loc[mask, method], labels=[2, 0])
        confs[method] = cm.tolist()
    (out / "confusions.json").write_text(json.dumps(confs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
