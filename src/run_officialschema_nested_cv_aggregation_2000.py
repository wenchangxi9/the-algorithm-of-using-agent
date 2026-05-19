from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABEL_TO_INT = {"NOT_HELPFUL": 0, "NEEDS_MORE_RATINGS": 1, "HELPFUL": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
RAW_LABEL_SCORE = {"NOT_HELPFUL": 0.0, "SOMEWHAT_HELPFUL": 0.5, "HELPFUL": 1.0}

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


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
    parser.add_argument("--target-coverage", type=float, default=0.1125)
    parser.add_argument("--acc-drop-max", type=float, default=0.005)
    return parser.parse_args()


def make_lr(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def add_entropy(df: pd.DataFrame, cols: list[str], out_col: str) -> None:
    arr = df[cols].fillna(0.0).clip(1e-9, 1.0).to_numpy(dtype=float)
    df[out_col] = -(arr * np.log(arr)).sum(axis=1)


def build_features(run_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    votes = pd.read_csv(run_dir / "agent_votes.csv", low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["raw_label"] = votes["parsed_rating"].fillna(votes.get("helpfulnessLevel", "")).astype(str)
    votes["raw_score"] = votes["raw_label"].map(RAW_LABEL_SCORE)
    votes = votes[votes["raw_score"].isin([0.0, 0.5, 1.0])].copy()

    numeric_cols = [
        "predicted_rating_score",
        "raw_score",
        "agree",
        "disagree",
        "helpfulClear",
        "helpfulGoodSources",
        "helpfulAddressesClaim",
        "helpfulImportantContext",
        "helpfulUnbiasedLanguage",
        "notHelpfulIncorrect",
        "notHelpfulSourcesMissingOrUnreliable",
        "notHelpfulMissingKeyPoints",
        "notHelpfulHardToUnderstand",
        "notHelpfulArgumentativeOrBiased",
        "notHelpfulIrrelevantSources",
        "notHelpfulOpinionSpeculation",
        "notHelpfulNoteNotNeeded",
        "confidence",
        "changes_reader_understanding",
    ]
    for col in numeric_cols:
        votes[col] = safe_num(votes, col)

    votes["is_h"] = (votes["raw_label"] == "HELPFUL").astype(float)
    votes["is_sh"] = (votes["raw_label"] == "SOMEWHAT_HELPFUL").astype(float)
    votes["is_nh"] = (votes["raw_label"] == "NOT_HELPFUL").astype(float)

    grouped = votes.groupby("noteId", as_index=False)
    base = grouped.agg(
        tweetId=("tweetId", "first"),
        currentStatus=("currentStatus", "first"),
        true_label_3way=("true_label_3way", "first"),
        true_label_text=("true_label_text", "first"),
        n_votes=("agent_id", "size"),
        vote_helpful=("is_h", "sum"),
        vote_somewhat_helpful=("is_sh", "sum"),
        vote_not_helpful=("is_nh", "sum"),
        mean_raw_score=("raw_score", "mean"),
        std_raw_score=("raw_score", "std"),
        mean_confidence=("confidence", "mean"),
        std_confidence=("confidence", "std"),
        mean_changes_reader_understanding=("changes_reader_understanding", "mean"),
        agree_rate=("agree", "mean"),
        disagree_rate=("disagree", "mean"),
        helpful_clear_rate=("helpfulClear", "mean"),
        helpful_good_sources_rate=("helpfulGoodSources", "mean"),
        helpful_addresses_claim_rate=("helpfulAddressesClaim", "mean"),
        helpful_important_context_rate=("helpfulImportantContext", "mean"),
        helpful_unbiased_language_rate=("helpfulUnbiasedLanguage", "mean"),
        not_helpful_incorrect_rate=("notHelpfulIncorrect", "mean"),
        not_helpful_sources_missing_or_unreliable_rate=("notHelpfulSourcesMissingOrUnreliable", "mean"),
        not_helpful_missing_key_points_rate=("notHelpfulMissingKeyPoints", "mean"),
        not_helpful_hard_to_understand_rate=("notHelpfulHardToUnderstand", "mean"),
        not_helpful_argumentative_or_biased_rate=("notHelpfulArgumentativeOrBiased", "mean"),
        not_helpful_irrelevant_sources_rate=("notHelpfulIrrelevantSources", "mean"),
        not_helpful_opinion_speculation_rate=("notHelpfulOpinionSpeculation", "mean"),
        not_helpful_note_not_needed_rate=("notHelpfulNoteNotNeeded", "mean"),
    )
    base["true_label_3way"] = pd.to_numeric(base["true_label_3way"], errors="coerce").astype(int)
    base["std_raw_score"] = base["std_raw_score"].fillna(0.0)
    base["std_confidence"] = base["std_confidence"].fillna(0.0)
    for name in ["helpful", "somewhat_helpful", "not_helpful"]:
        base[f"share_{name}"] = base[f"vote_{name}"] / base["n_votes"].clip(lower=1)
    add_entropy(base, ["share_not_helpful", "share_somewhat_helpful", "share_helpful"], "raw_vote_entropy")
    base["helpful_vs_not_margin"] = (base["share_helpful"] - base["share_not_helpful"]).abs()
    base["positive_mass"] = base["share_helpful"] + 0.5 * base["share_somewhat_helpful"]
    base["negative_mass"] = base["share_not_helpful"] + 0.5 * base["share_somewhat_helpful"]
    base["resolved_raw_vote_share"] = base["share_helpful"] + base["share_not_helpful"]
    base["helpful_reason_mean"] = base[
        [
            "helpful_clear_rate",
            "helpful_good_sources_rate",
            "helpful_addresses_claim_rate",
            "helpful_important_context_rate",
            "helpful_unbiased_language_rate",
        ]
    ].mean(axis=1)
    base["not_helpful_reason_mean"] = base[
        [
            "not_helpful_incorrect_rate",
            "not_helpful_sources_missing_or_unreliable_rate",
            "not_helpful_missing_key_points_rate",
            "not_helpful_hard_to_understand_rate",
            "not_helpful_argumentative_or_biased_rate",
            "not_helpful_irrelevant_sources_rate",
            "not_helpful_opinion_speculation_rate",
            "not_helpful_note_not_needed_rate",
        ]
    ].mean(axis=1)

    total_conf = votes.groupby("noteId")["confidence"].sum().replace(0, np.nan)
    for raw_label, name in [
        ("NOT_HELPFUL", "not_helpful"),
        ("SOMEWHAT_HELPFUL", "somewhat_helpful"),
        ("HELPFUL", "helpful"),
    ]:
        weighted = votes[votes["raw_label"] == raw_label].groupby("noteId")["confidence"].sum() / total_conf
        base = base.merge(weighted.rename(f"conf_weighted_share_{name}").reset_index(), on="noteId", how="left")

    for raw_label, name in [
        ("NOT_HELPFUL", "nh"),
        ("SOMEWHAT_HELPFUL", "sh"),
        ("HELPFUL", "h"),
    ]:
        part = (
            votes[votes["raw_label"] == raw_label]
            .groupby("noteId")[
                [
                    "confidence",
                    "changes_reader_understanding",
                    "helpfulGoodSources",
                    "helpfulAddressesClaim",
                    "helpfulImportantContext",
                    "notHelpfulSourcesMissingOrUnreliable",
                    "notHelpfulMissingKeyPoints",
                    "notHelpfulNoteNotNeeded",
                ]
            ]
            .mean()
            .add_prefix(f"{name}_mean_")
            .reset_index()
        )
        base = base.merge(part, on="noteId", how="left")

    label_map = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}
    votes["raw_label_id"] = votes["raw_label"].map(label_map)
    label_pivot = votes.pivot_table(index="noteId", columns="agent_id", values="raw_label_id", aggfunc="first")
    label_pivot = label_pivot.add_prefix("agent_raw_label__").reset_index()
    base = base.merge(label_pivot, on="noteId", how="left")
    for col in [c for c in base.columns if c.startswith("agent_raw_label__")]:
        for label_id, label_name in [(0, "nh"), (1, "sh"), (2, "h")]:
            base[f"{col}__is_{label_name}"] = (base[col] == label_id).astype(float)

    for metric in [
        "confidence",
        "changes_reader_understanding",
        "helpfulGoodSources",
        "helpfulAddressesClaim",
        "helpfulImportantContext",
        "notHelpfulSourcesMissingOrUnreliable",
        "notHelpfulMissingKeyPoints",
        "notHelpfulNoteNotNeeded",
    ]:
        pivot = votes.pivot_table(index="noteId", columns="agent_id", values=metric, aggfunc="first")
        base = base.merge(pivot.add_prefix(f"agent_{metric}__").reset_index(), on="noteId", how="left")

    pilot = pd.read_csv(run_dir / "pilot_notes.csv", low_memory=False)
    pilot["noteId"] = pilot["noteId"].astype(str)
    meta_cols = [
        "noteId",
        "year",
        "classification",
        "primary_topic",
        "isMediaNote",
        "isCollaborativeNote",
        "misleadingManipulatedMedia",
        "misleadingFactualError",
        "misleadingOutdatedInformation",
        "misleadingMissingImportantContext",
        "misleadingUnverifiedClaimAsFact",
        "misleadingSatire",
        "notMisleadingOther",
        "notMisleadingFactuallyCorrect",
        "notMisleadingOutdatedButNotWhenWritten",
        "notMisleadingClearlySatire",
        "notMisleadingPersonalOpinion",
        "topic_count",
    ]
    meta = pilot[[c for c in meta_cols if c in pilot.columns]].drop_duplicates("noteId").copy()
    categorical = [c for c in ["classification", "primary_topic"] if c in meta.columns]
    if categorical:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        arr = enc.fit_transform(meta[categorical].fillna("UNKNOWN"))
        enc_df = pd.DataFrame(arr, columns=[f"meta_{x}" for x in enc.get_feature_names_out(categorical)])
        meta = pd.concat([meta.drop(columns=categorical).reset_index(drop=True), enc_df], axis=1)
    for col in meta.columns:
        if col != "noteId":
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    full = base.merge(meta, on="noteId", how="left")

    non_features = {"noteId", "tweetId", "currentStatus", "true_label_3way", "true_label_text"}
    summary_features = [
        c
        for c in base.columns
        if c not in non_features
        and not (c.startswith("agent_raw_label__") and "__is_" not in c)
        and not c.startswith("agent_confidence__")
        and not c.startswith("agent_changes_reader_understanding__")
        and not c.startswith("agent_helpful")
        and not c.startswith("agent_notHelpful")
    ]
    full_features = [
        c
        for c in base.columns
        if c not in non_features and not (c.startswith("agent_raw_label__") and "__is_" not in c)
    ]
    metadata_features = [c for c in full.columns if c not in base.columns and c != "noteId"]
    return full.sort_values("noteId").reset_index(drop=True), {
        "summary": summary_features,
        "full_agent": full_features,
        "full_agent_plus_metadata": full_features + metadata_features,
    }


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    recs = []
    for label_id, label in INT_TO_LABEL.items():
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


def align_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    prob = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = prob[:, i]
    return out


def direct_pred_from_prob(prob: np.ndarray) -> np.ndarray:
    return prob.argmax(axis=1).astype(int)


def guarded_nmr_pred(prob: np.ndarray, t_nmr: float, t_resolved_mass: float, t_margin: float) -> np.ndarray:
    p_nmr = prob[:, 1]
    p_nh = prob[:, 0]
    p_h = prob[:, 2]
    out = prob.argmax(axis=1).astype(int)
    resolved_mass = p_nh + p_h
    margin = np.sort(prob, axis=1)[:, -1] - np.sort(prob, axis=1)[:, -2]
    gate = (out != 1) & (p_nmr >= t_nmr) & (resolved_mass <= t_resolved_mass) & (margin <= t_margin)
    out[gate] = 1
    return out.astype(int)


def score_key(m: dict[str, float | int]) -> tuple[float, ...]:
    joint = 2.0 * m["accuracy"] * m["balanced_accuracy"] / max(m["accuracy"] + m["balanced_accuracy"], 1e-12)
    return (m["recall_needs_more_ratings"], m["balanced_accuracy"], m["accuracy"], joint, m["min_recall"])


def choose_spec(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    inner_folds: int,
    seed: int,
    acc_drop_max: float,
) -> tuple[dict[str, float], dict[str, float | int]]:
    t_nmr_grid = [0.46, 0.52, 0.58]
    t_resolved_mass_grid = [0.56, 0.62]
    t_margin_grid = [0.08, 0.12]

    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    oof_prob = np.zeros((len(y_train), 3), dtype=float)

    for tr, va in inner.split(X_train, y_train):
        model = make_lr(seed=seed)
        model.fit(X_train.iloc[tr], y_train[tr])
        oof_prob[va] = align_prob(model, X_train.iloc[va])

    base_pred = direct_pred_from_prob(oof_prob)
    base_m = metric(y_train, base_pred)
    acc_floor = base_m["accuracy"] - acc_drop_max

    best = None
    for t_nmr in t_nmr_grid:
        for t_resolved_mass in t_resolved_mass_grid:
            for t_margin in t_margin_grid:
                pred = guarded_nmr_pred(oof_prob, float(t_nmr), float(t_resolved_mass), float(t_margin))
                m = metric(y_train, pred)
                if m["accuracy"] < acc_floor:
                    continue
                key = score_key(m)
                if best is None or key > best[0]:
                    best = (
                        key,
                        {
                            "t_nmr": float(t_nmr),
                            "t_resolved_mass": float(t_resolved_mass),
                            "t_margin": float(t_margin),
                            "base_acc_inner": float(base_m["accuracy"]),
                            "base_balanced_inner": float(base_m["balanced_accuracy"]),
                            "acc_floor_inner": float(acc_floor),
                        },
                        m,
                    )

    if best is None:
        best = (
            score_key(base_m),
            {
                "t_nmr": 1.1,
                "t_resolved_mass": 0.0,
                "t_margin": 0.0,
                "base_acc_inner": float(base_m["accuracy"]),
                "base_balanced_inner": float(base_m["balanced_accuracy"]),
                "acc_floor_inner": float(acc_floor),
            },
            base_m,
        )

    return best[1], best[2]


def nested_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: int,
    inner_folds: int,
    seed: int,
    acc_drop_max: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    X = df[feature_cols]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred_base = np.zeros(len(df), dtype=int)
    pred_guard = np.zeros(len(df), dtype=int)
    rows: list[dict[str, float | int]] = []

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        spec, inner_m = choose_spec(
            X.iloc[tr].reset_index(drop=True),
            y[tr],
            inner_folds=inner_folds,
            seed=seed + fold,
            acc_drop_max=acc_drop_max,
        )

        model = make_lr(seed=seed + fold)
        model.fit(X.iloc[tr], y[tr])
        prob = align_prob(model, X.iloc[te])
        base_fold = direct_pred_from_prob(prob)
        guard_fold = guarded_nmr_pred(prob, spec["t_nmr"], spec["t_resolved_mass"], spec["t_margin"])

        pred_base[te] = base_fold
        pred_guard[te] = guard_fold

        base_test = metric(y[te], base_fold)
        guard_test = metric(y[te], guard_fold)
        rows.append(
            {
                "fold": fold,
                "t_nmr": spec["t_nmr"],
                "t_resolved_mass": spec["t_resolved_mass"],
                "t_margin": spec["t_margin"],
                "inner_base_acc": spec["base_acc_inner"],
                "inner_base_balanced": spec["base_balanced_inner"],
                "inner_acc_floor": spec["acc_floor_inner"],
                **{f"inner_guard_{k}": v for k, v in inner_m.items()},
                **{f"test_base_{k}": v for k, v in base_test.items()},
                **{f"test_guard_{k}": v for k, v in guard_test.items()},
            }
        )

    return pred_base, pred_guard, rows


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "officialschema_nested_cv_aggregation_20260512"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    y = df["true_label_3way"].to_numpy(dtype=int)
    predictions = df[["noteId", "true_label_3way", "true_label_text"]].copy()

    fast_path = run_dir / "officialschema_nested_cv_fast_20260512" / "officialschema_nested_cv_fast_oof_predictions.csv"
    if not fast_path.exists():
        raise FileNotFoundError(f"Missing fast OOF predictions: {fast_path}")
    fast_oof = pd.read_csv(fast_path, low_memory=False)
    df = add_meta_features(df, fast_oof)

    meta_cols = [c for c in df.columns if c.startswith("nested_lr_")]
    feature_cols = feature_sets["full_agent_plus_metadata"] + meta_cols

    base_pred, guard_pred, fold_rows = nested_cv(
        df=df,
        feature_cols=feature_cols,
        folds=args.folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        acc_drop_max=args.acc_drop_max,
    )

    predictions["direct_mc_full_plus_meta_balanced"] = base_pred
    predictions["guarded_nmr_gate_full_plus_meta"] = guard_pred

    rows = [
        {
            "method": "direct_mc_full_plus_meta_balanced",
            "family": "direct_multiclass",
            "feature_set": "full_agent_plus_metadata_plus_meta_probs",
            "objective": "balanced_accuracy",
            "n_features": len(feature_cols),
            "pred_resolved_coverage": float(np.isin(base_pred, [0, 2]).mean()),
            "strict_accuracy_on_pred_resolved": float((y[np.isin(base_pred, [0, 2])] == base_pred[np.isin(base_pred, [0, 2])]).mean())
            if np.isin(base_pred, [0, 2]).any()
            else None,
            **metric(y, base_pred),
        },
        {
            "method": "guarded_nmr_gate_full_plus_meta",
            "family": "guarded_nmr_gate",
            "feature_set": "full_agent_plus_metadata_plus_meta_probs",
            "objective": "nmr_recall_then_balanced_accuracy",
            "n_features": len(feature_cols),
            "pred_resolved_coverage": float(np.isin(guard_pred, [0, 2]).mean()),
            "strict_accuracy_on_pred_resolved": float((y[np.isin(guard_pred, [0, 2])] == guard_pred[np.isin(guard_pred, [0, 2])]).mean())
            if np.isin(guard_pred, [0, 2]).any()
            else None,
            **metric(y, guard_pred),
        },
    ]

    summary = pd.DataFrame(rows)
    for col in [
        "accuracy",
        "balanced_accuracy",
        "recall_not_helpful",
        "recall_needs_more_ratings",
        "recall_helpful",
        "min_recall",
        "pred_resolved_coverage",
        "strict_accuracy_on_pred_resolved",
    ]:
        summary[f"{col}_pct"] = pd.to_numeric(summary[col], errors="coerce") * 100.0

    base_acc = float(summary.loc[summary["method"] == "direct_mc_full_plus_meta_balanced", "accuracy_pct"].iloc[0])
    base_bal = float(summary.loc[summary["method"] == "direct_mc_full_plus_meta_balanced", "balanced_accuracy_pct"].iloc[0])
    base_nmr = float(summary.loc[summary["method"] == "direct_mc_full_plus_meta_balanced", "recall_needs_more_ratings_pct"].iloc[0])
    summary["delta_accuracy_pct"] = summary["accuracy_pct"] - base_acc
    summary["delta_balanced_accuracy_pct"] = summary["balanced_accuracy_pct"] - base_bal
    summary["delta_nmr_recall_pct"] = summary["recall_needs_more_ratings_pct"] - base_nmr
    summary = summary.sort_values(["accuracy", "balanced_accuracy"], ascending=[False, False]).reset_index(drop=True)

    best = str(summary.iloc[0]["method"])
    label_map = INT_TO_LABEL
    base_confusion = pd.crosstab(predictions["true_label_text"], predictions["direct_mc_full_plus_meta_balanced"].map(label_map), margins=True)
    guard_confusion = pd.crosstab(predictions["true_label_text"], predictions["guarded_nmr_gate_full_plus_meta"].map(label_map), margins=True)
    best_confusion = pd.crosstab(predictions["true_label_text"], predictions[best].map(label_map), margins=True)

    df.to_csv(out_dir / "officialschema_feature_table.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "officialschema_nested_cv_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "officialschema_nested_cv_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out_dir / "officialschema_nested_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    base_confusion.to_csv(out_dir / "base_confusion.csv", encoding="utf-8-sig")
    guard_confusion.to_csv(out_dir / "guard_confusion.csv", encoding="utf-8-sig")
    best_confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")

    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_notes": int(len(df)),
        "folds": int(args.folds),
        "inner_folds": int(args.inner_folds),
        "seed": int(args.seed),
        "target_coverage": float(args.target_coverage),
        "acc_drop_max": float(args.acc_drop_max),
        "true_distribution": df["true_label_text"].value_counts().to_dict(),
        "feature_set": "full_agent_plus_metadata_plus_meta_probs",
        "n_features": int(len(feature_cols)),
        "fast_oof_path": str(fast_path),
        "best_by_accuracy": summary.iloc[0].to_dict(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "method",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "delta_accuracy_pct",
        "delta_balanced_accuracy_pct",
        "delta_nmr_recall_pct",
        "pred_resolved_coverage_pct",
        "strict_accuracy_on_pred_resolved_pct",
        "recall_not_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_helpful_pct",
    ]
    print(summary[cols].to_string(index=False))
    print("\nBest confusion matrix:")
    print(best_confusion.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
