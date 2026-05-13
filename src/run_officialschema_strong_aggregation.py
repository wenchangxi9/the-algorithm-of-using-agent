from __future__ import annotations

import argparse
import json
import math
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


LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}
RAW_LABEL_ID = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}
RESOLVED = {0, 2}

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
NUMERIC_AGENT_FIELDS = [
    "predicted_rating_score",
    "confidence",
    "changes_reader_understanding",
    "agree",
    "disagree",
    *HELPFUL_REASONS,
    *NOT_HELPFUL_REASONS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_indicator(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where(x >= 0, np.nan)


def entropy_from_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    arr = df[cols].fillna(0).clip(1e-9, 1.0).to_numpy(dtype=float)
    return pd.Series(-(arr * np.log(arr)).sum(axis=1), index=df.index)


def add_group_stats(base: pd.DataFrame, votes: pd.DataFrame, mask: pd.Series, prefix: str) -> pd.DataFrame:
    cols = ["confidence", "changes_reader_understanding", "predicted_rating_score", *HELPFUL_REASONS, *NOT_HELPFUL_REASONS]
    part = votes[mask].groupby("noteId")[cols].mean().add_prefix(prefix).reset_index()
    return base.merge(part, on="noteId", how="left")


def build_feature_table(run_dir: Path) -> pd.DataFrame:
    votes = pd.read_csv(run_dir / "agent_votes.csv", low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["agent_id"] = votes["agent_id"].astype(str)
    votes["raw_label_id"] = votes["parsed_rating"].map(RAW_LABEL_ID)
    votes = votes[votes["raw_label_id"].isin([0, 1, 2])].copy()
    for col in NUMERIC_AGENT_FIELDS:
        if col in votes.columns:
            votes[col] = clean_indicator(votes[col])
        else:
            votes[col] = np.nan
    votes["is_h"] = (votes["raw_label_id"] == 2).astype(float)
    votes["is_sh"] = (votes["raw_label_id"] == 1).astype(float)
    votes["is_nh"] = (votes["raw_label_id"] == 0).astype(float)
    votes["helpful_reason_sum"] = votes[HELPFUL_REASONS].sum(axis=1, skipna=True)
    votes["not_helpful_reason_sum"] = votes[NOT_HELPFUL_REASONS].sum(axis=1, skipna=True)
    votes["helpful_reason_any"] = (votes["helpful_reason_sum"] > 0).astype(float)
    votes["not_helpful_reason_any"] = (votes["not_helpful_reason_sum"] > 0).astype(float)
    votes["contradict_h_vote_has_nh_reason"] = ((votes["raw_label_id"] == 2) & (votes["not_helpful_reason_sum"] > 0)).astype(float)
    votes["contradict_nh_vote_has_h_reason"] = ((votes["raw_label_id"] == 0) & (votes["helpful_reason_sum"] > 0)).astype(float)
    votes["decisive_conf"] = (votes["confidence"].fillna(0) >= 75).astype(float)
    votes["high_understanding"] = (votes["changes_reader_understanding"].fillna(0) >= 65).astype(float)
    votes["low_understanding"] = (votes["changes_reader_understanding"].fillna(100) <= 35).astype(float)

    base = votes.groupby("noteId", as_index=False).agg(
        tweetId=("tweetId", "first"),
        currentStatus=("currentStatus", "first"),
        true_label_3way=("true_label_3way", "first"),
        true_label_text=("true_label_text", "first"),
        n_votes=("agent_id", "size"),
        vote_helpful=("is_h", "sum"),
        vote_somewhat=("is_sh", "sum"),
        vote_not_helpful=("is_nh", "sum"),
        mean_score=("predicted_rating_score", "mean"),
        std_score=("predicted_rating_score", "std"),
        mean_confidence=("confidence", "mean"),
        std_confidence=("confidence", "std"),
        mean_understanding=("changes_reader_understanding", "mean"),
        std_understanding=("changes_reader_understanding", "std"),
        agree_rate=("agree", "mean"),
        disagree_rate=("disagree", "mean"),
        helpful_reason_sum_mean=("helpful_reason_sum", "mean"),
        not_helpful_reason_sum_mean=("not_helpful_reason_sum", "mean"),
        helpful_reason_any_rate=("helpful_reason_any", "mean"),
        not_helpful_reason_any_rate=("not_helpful_reason_any", "mean"),
        contradiction_h_has_nh_rate=("contradict_h_vote_has_nh_reason", "mean"),
        contradiction_nh_has_h_rate=("contradict_nh_vote_has_h_reason", "mean"),
        decisive_conf_rate=("decisive_conf", "mean"),
        high_understanding_rate=("high_understanding", "mean"),
        low_understanding_rate=("low_understanding", "mean"),
        **{f"{c}_rate": (c, "mean") for c in HELPFUL_REASONS + NOT_HELPFUL_REASONS},
    )
    base["true_label_3way"] = pd.to_numeric(base["true_label_3way"], errors="coerce").astype(int)
    base["std_score"] = base["std_score"].fillna(0)
    base["std_confidence"] = base["std_confidence"].fillna(0)
    base["std_understanding"] = base["std_understanding"].fillna(0)
    for col in ["helpful", "somewhat", "not_helpful"]:
        base[f"share_{col}"] = base[f"vote_{col}"] / base["n_votes"].clip(lower=1)
    base["vote_entropy"] = entropy_from_cols(base, ["share_not_helpful", "share_somewhat", "share_helpful"])
    base["h_vs_nh_margin"] = (base["share_helpful"] - base["share_not_helpful"]).abs()
    base["h_minus_nh"] = base["share_helpful"] - base["share_not_helpful"]
    base["resolved_share"] = base["share_helpful"] + base["share_not_helpful"]
    base["somewhat_pressure"] = base["share_somewhat"] * (1 + base["vote_entropy"])
    base["helpful_support"] = (
        base["share_helpful"]
        + 0.25 * base["share_somewhat"]
        + 0.15 * base["helpful_reason_any_rate"]
        + 0.15 * base["helpfulImportantContext_rate"]
        + 0.15 * base["helpfulAddressesClaim_rate"]
        + 0.10 * base["helpfulGoodSources_rate"]
        + 0.003 * base["mean_understanding"].fillna(0)
    )
    base["not_helpful_support"] = (
        base["share_not_helpful"]
        + 0.25 * base["share_somewhat"]
        + 0.15 * base["not_helpful_reason_any_rate"]
        + 0.15 * base["notHelpfulMissingKeyPoints_rate"]
        + 0.15 * base["notHelpfulSourcesMissingOrUnreliable_rate"]
        + 0.15 * base["notHelpfulNoteNotNeeded_rate"]
        + 0.003 * (100 - base["mean_understanding"].fillna(50))
    )
    base["support_margin_signed"] = base["helpful_support"] - base["not_helpful_support"]
    base["support_margin_abs"] = base["support_margin_signed"].abs()

    base = add_group_stats(base, votes, votes["raw_label_id"] == 2, "h_vote_mean_")
    base = add_group_stats(base, votes, votes["raw_label_id"] == 1, "sh_vote_mean_")
    base = add_group_stats(base, votes, votes["raw_label_id"] == 0, "nh_vote_mean_")

    # Per-agent one-hot labels and raw numeric outputs.
    labels = votes.pivot_table(index="noteId", columns="agent_id", values="raw_label_id", aggfunc="first")
    labels = labels.add_prefix("agent_label__").reset_index()
    base = base.merge(labels, on="noteId", how="left")
    for col in [c for c in base.columns if c.startswith("agent_label__")]:
        for label_id, name in [(0, "nh"), (1, "sh"), (2, "h")]:
            base[f"{col}__is_{name}"] = (base[col] == label_id).astype(float)
    for metric in ["confidence", "changes_reader_understanding", "helpfulImportantContext", "helpfulAddressesClaim", "helpfulGoodSources", "notHelpfulMissingKeyPoints", "notHelpfulSourcesMissingOrUnreliable", "notHelpfulNoteNotNeeded"]:
        pivot = votes.pivot_table(index="noteId", columns="agent_id", values=metric, aggfunc="first")
        base = base.merge(pivot.add_prefix(f"agent_{metric}__").reset_index(), on="noteId", how="left")

    non_features = {"noteId", "tweetId", "currentStatus", "true_label_3way", "true_label_text"}
    drop_raw_ordinal = [c for c in base.columns if c.startswith("agent_label__") and "__is_" not in c]
    return base.drop(columns=drop_raw_ordinal).sort_values("noteId").reset_index(drop=True)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    out = {
        "accuracy": float((y == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)),
    }
    for label_id, label in LABEL.items():
        mask = y == label_id
        out[f"recall_{label.lower()}"] = float((pred[mask] == label_id).mean()) if mask.any() else np.nan
        out[f"n_{label.lower()}"] = int(mask.sum())
    out["h_to_nh"] = int(((y == 2) & (pred == 0)).sum())
    out["nh_to_h"] = int(((y == 0) & (pred == 2)).sum())
    out["cross_error"] = out["h_to_nh"] + out["nh_to_h"]
    return out


def model_factory(name: str, seed: int):
    if name.startswith("lr"):
        c = float(name.split("_")[1])
        weight = "balanced" if name.endswith("balanced") else None
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight=weight, max_iter=5000, solver="lbfgs", random_state=seed)),
        ])
    if name == "rf_balanced":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ])
    if name == "extra_balanced":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ])
    if name == "gb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(n_estimators=160, learning_rate=0.04, max_depth=2, random_state=seed)),
        ])
    raise ValueError(name)


def objective_key(m: dict, objective: str) -> tuple:
    if objective == "balanced_crosssafe":
        return (m["balanced_accuracy"], -m["cross_error"], m["accuracy"])
    if objective == "accuracy_crosssafe":
        return (m["accuracy"], -m["cross_error"], m["balanced_accuracy"])
    if objective == "crosssafe_balanced":
        return (-m["cross_error"], m["balanced_accuracy"], m["accuracy"])
    raise ValueError(objective)


def apply_margin_rule(prob: np.ndarray, base_pred: np.ndarray, margin: float, nmr_threshold: float) -> np.ndarray:
    out = base_pred.copy()
    h = prob[:, 2]
    nh = prob[:, 0]
    nmr = prob[:, 1]
    cross_risk = np.abs(h - nh) < margin
    enough_nmr = nmr >= nmr_threshold
    out[cross_risk & enough_nmr] = 1
    return out


def choose_inner(
    X: pd.DataFrame,
    y: np.ndarray,
    inner_folds: int,
    seed: int,
    objective: str,
) -> tuple[str, float, float, dict]:
    model_names = [
        "lr_0.03_balanced", "lr_0.1_balanced", "lr_0.3_balanced", "lr_1.0_balanced", "lr_3.0_balanced",
        "lr_0.1_none", "lr_0.3_none", "lr_1.0_none",
        "rf_balanced", "extra_balanced", "gb",
    ]
    margins = [0.0, 0.05, 0.10, 0.15, 0.20]
    nmr_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best = None
    for model_name in model_names:
        prob = np.zeros((len(y), 3), dtype=float)
        pred = np.zeros(len(y), dtype=int)
        for tr, va in inner.split(X, y):
            model = model_factory(model_name, seed)
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])
            aligned = np.zeros((len(va), 3), dtype=float)
            classes = model.named_steps["clf"].classes_
            for i, cls in enumerate(classes):
                aligned[:, int(cls)] = p[:, i]
            prob[va] = aligned
            pred[va] = aligned.argmax(axis=1)
        for margin in margins:
            for nmr_threshold in nmr_thresholds:
                adjusted = apply_margin_rule(prob, pred, margin, nmr_threshold)
                m = metrics(y, adjusted)
                key = objective_key(m, objective)
                if best is None or key > best[0]:
                    best = (key, model_name, margin, nmr_threshold, m)
    return best[1], best[2], best[3], best[4]


def nested_eval(df: pd.DataFrame, feature_cols: list[str], folds: int, inner_folds: int, seed: int, objective: str):
    y = df["true_label_3way"].to_numpy(dtype=int)
    X = df[feature_cols]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros((len(y), 3), dtype=float)
    fold_rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        model_name, margin, nmr_threshold, inner_metrics = choose_inner(
            X.iloc[tr].reset_index(drop=True),
            y[tr],
            inner_folds,
            seed + fold,
            objective,
        )
        model = model_factory(model_name, seed + fold)
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])
        aligned = np.zeros((len(te), 3), dtype=float)
        for i, cls in enumerate(model.named_steps["clf"].classes_):
            aligned[:, int(cls)] = p[:, i]
        raw_pred = aligned.argmax(axis=1)
        fold_pred = apply_margin_rule(aligned, raw_pred, margin, nmr_threshold)
        pred[te] = fold_pred
        prob[te] = aligned
        fold_rows.append({
            "fold": fold,
            "objective": objective,
            "model": model_name,
            "margin": margin,
            "nmr_threshold": nmr_threshold,
            **{f"inner_{k}": v for k, v in inner_metrics.items()},
            **{f"test_{k}": v for k, v in metrics(y[te], fold_pred).items()},
        })
    return pred, prob, fold_rows, metrics(y, pred)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "officialschema_strong_aggregation_20260513"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_feature_table(run_dir)
    non_features = {"noteId", "tweetId", "currentStatus", "true_label_3way", "true_label_text"}
    feature_cols = [c for c in df.columns if c not in non_features]
    y = df["true_label_3way"].to_numpy(dtype=int)

    rows = []
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    fold_tables = []

    raw_majority = df[["vote_not_helpful", "vote_somewhat", "vote_helpful"]].to_numpy(dtype=float).argmax(axis=1)
    baselines = {
        "raw_vote_majority": raw_majority,
        "raw_score_033_067": np.where(df["mean_score"].to_numpy() >= 2/3, 2, np.where(df["mean_score"].to_numpy() <= 1/3, 0, 1)),
        "support_margin_rule": np.where(
            df["support_margin_abs"].to_numpy() < 0.18,
            1,
            np.where(df["support_margin_signed"].to_numpy() > 0, 2, 0),
        ),
    }
    for name, pred in baselines.items():
        preds[name] = pred
        rows.append({"method": name, "family": "baseline", "n_features": 0, **metrics(y, pred)})

    for objective in ["balanced_crosssafe", "accuracy_crosssafe", "crosssafe_balanced"]:
        method = f"strong_nested_{objective}"
        pred, prob, fold_rows, m = nested_eval(df, feature_cols, args.folds, args.inner_folds, args.seed, objective)
        preds[method] = pred
        preds[f"{method}_pred_text"] = pd.Series(pred).map(LABEL)
        preds[f"{method}_prob_not_helpful"] = prob[:, 0]
        preds[f"{method}_prob_nmr"] = prob[:, 1]
        preds[f"{method}_prob_helpful"] = prob[:, 2]
        rows.append({"method": method, "family": "strong_nested", "n_features": len(feature_cols), **m})
        fold_tables.append(pd.DataFrame(fold_rows).assign(method=method))

    summary = pd.DataFrame(rows)
    for col in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful"]:
        summary[f"{col}_pct"] = summary[col] * 100.0
    summary = summary.sort_values(["balanced_accuracy", "accuracy", "cross_error"], ascending=[False, False, True])
    best_method = str(summary.iloc[0]["method"])
    best_pred = preds[best_method].to_numpy(dtype=int)
    confusion = pd.crosstab(df["true_label_text"], pd.Series(best_pred).map(LABEL), margins=True)

    df.to_csv(out_dir / "strong_feature_table.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "strong_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "strong_summary.csv", index=False, encoding="utf-8-sig")
    if fold_tables:
        pd.concat(fold_tables, ignore_index=True).to_csv(out_dir / "strong_fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_notes": int(len(df)),
        "n_features": int(len(feature_cols)),
        "best_method": best_method,
        "best": summary.iloc[0].to_dict(),
        "true_distribution": df["true_label_text"].value_counts().to_dict(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = [
        "method",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "recall_not_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_helpful_pct",
        "h_to_nh",
        "nh_to_h",
        "cross_error",
    ]
    print(summary[cols].to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
