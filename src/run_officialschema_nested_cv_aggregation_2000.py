from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
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
RESOLVED = {0, 2}
RAW_LABEL_SCORE = {"NOT_HELPFUL": 0.0, "SOMEWHAT_HELPFUL": 0.5, "HELPFUL": 1.0}


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
    return parser.parse_args()


def make_lr(c: float, class_weight: str | None, multi_class: bool = True) -> Pipeline:
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
                    random_state=42,
                    solver="lbfgs",
                    multi_class="auto" if multi_class else "auto",
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
    for c in meta.columns:
        if c != "noteId":
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
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


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {}
    out["accuracy"] = float((y_true == pred).mean())
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    for label_id, label in INT_TO_LABEL.items():
        mask = y_true == label_id
        out[f"recall_{label.lower()}"] = float((pred[mask] == label_id).mean()) if mask.any() else math.nan
        out[f"n_{label.lower()}"] = int(mask.sum())
    pred_resolved = np.isin(pred, list(RESOLVED))
    true_resolved = np.isin(y_true, list(RESOLVED))
    out["pred_resolved_coverage"] = float(pred_resolved.mean())
    out["pred_resolved_notes"] = int(pred_resolved.sum())
    out["strict_accuracy_on_pred_resolved"] = (
        float((y_true[pred_resolved] == pred[pred_resolved]).mean()) if pred_resolved.any() else None
    )
    both = pred_resolved & true_resolved
    out["binary_accuracy_true_resolved_selected"] = (
        float((y_true[both] == pred[both]).mean()) if both.any() else None
    )
    out["true_resolved_selected"] = int(both.sum())
    out["true_resolved_total"] = int(true_resolved.sum())
    return out


def objective_key(m: dict[str, float | int | None], objective: str, target_coverage: float) -> tuple[float, ...]:
    if objective == "accuracy":
        return (float(m["accuracy"]), float(m["balanced_accuracy"]))
    if objective == "balanced":
        return (float(m["balanced_accuracy"]), float(m["accuracy"]))
    if objective == "target_coverage":
        acc = -1.0 if m["strict_accuracy_on_pred_resolved"] is None else float(m["strict_accuracy_on_pred_resolved"])
        return (-abs(float(m["pred_resolved_coverage"]) - target_coverage), acc, float(m["balanced_accuracy"]))
    if objective == "resolved_precision":
        acc = -1.0 if m["strict_accuracy_on_pred_resolved"] is None else float(m["strict_accuracy_on_pred_resolved"])
        return (acc, -abs(float(m["pred_resolved_coverage"]) - target_coverage), float(m["balanced_accuracy"]))
    raise ValueError(objective)


@dataclass(frozen=True)
class Spec:
    c: float
    class_weight: str | None


def nested_multiclass_lr(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: int,
    inner_folds: int,
    seed: int,
    objective: str,
    target_coverage: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    X = df[feature_cols]
    pred = np.zeros(len(df), dtype=int)
    prob_all = np.zeros((len(df), 3), dtype=float)
    rows = []
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weights: list[str | None] = [None, "balanced"]
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        for c in c_grid:
            for weight in weights:
                oof_pred = np.zeros(len(train_idx), dtype=int)
                for inner_train, inner_val in inner.split(X_train, y_train):
                    model = make_lr(c, weight, multi_class=True)
                    model.fit(X_train.iloc[inner_train], y_train[inner_train])
                    oof_pred[inner_val] = model.predict(X_train.iloc[inner_val])
                m = metrics(y_train, oof_pred)
                key = objective_key(m, objective, target_coverage)
                if best is None or key > best[0]:
                    best = (key, Spec(c, weight), m)
        spec = best[1]
        model = make_lr(spec.c, spec.class_weight, multi_class=True)
        model.fit(X.iloc[train_idx], y[train_idx])
        fold_prob = model.predict_proba(X.iloc[test_idx])
        fold_pred = model.predict(X.iloc[test_idx])
        pred[test_idx] = fold_pred
        # sklearn keeps classes sorted here, but align defensively.
        aligned = np.zeros((len(test_idx), 3), dtype=float)
        for pos, cls in enumerate(model.named_steps["clf"].classes_):
            aligned[:, int(cls)] = fold_prob[:, pos]
        prob_all[test_idx] = aligned
        rows.append(
            {
                "fold": fold,
                "objective": objective,
                "c": spec.c,
                "class_weight": spec.class_weight or "none",
                **{f"inner_{k}": v for k, v in best[2].items()},
                **{f"test_{k}": v for k, v in metrics(y[test_idx], fold_pred).items()},
            }
        )
    return pred, prob_all, rows


@dataclass(frozen=True)
class TwoStageSpec:
    c_resolved: float
    w_resolved: str | None
    c_direction: float
    w_direction: str | None
    resolved_threshold: float
    direction_threshold: float


def pred_two_stage(res_prob: np.ndarray, help_prob: np.ndarray, rt: float, dt: float) -> np.ndarray:
    out = np.full(len(res_prob), 1, dtype=int)
    mask = res_prob >= rt
    out[mask] = np.where(help_prob[mask] >= dt, 2, 0)
    return out


def threshold_search(
    y: np.ndarray,
    res_prob: np.ndarray,
    help_prob: np.ndarray,
    objective: str,
    target_coverage: float,
) -> tuple[float, float, dict]:
    res_candidates = [float("inf")] + list(np.quantile(res_prob, np.linspace(0, 1, 101))) + list(np.unique(res_prob))
    res_candidates = sorted(set(float(x) for x in res_candidates), reverse=True)
    dir_candidates = np.arange(0.20, 0.81, 0.02)
    best = None
    for rt in res_candidates:
        for dt in dir_candidates:
            pred = pred_two_stage(res_prob, help_prob, rt, float(dt))
            m = metrics(y, pred)
            key = objective_key(m, objective, target_coverage)
            if best is None or key > best[0]:
                best = (key, rt, float(dt), m)
    assert best is not None
    return best[1], best[2], best[3]


def nested_two_stage_lr(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: int,
    inner_folds: int,
    seed: int,
    objective: str,
    target_coverage: float,
) -> tuple[np.ndarray, list[dict]]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    X = df[feature_cols]
    pred = np.full(len(df), 1, dtype=int)
    rows = []
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    c_grid = [0.1, 1.0, 10.0]
    weights: list[str | None] = [None, "balanced"]
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        for c_res in c_grid:
            for w_res in weights:
                for c_dir in c_grid:
                    for w_dir in weights:
                        oof_res = np.zeros(len(train_idx), dtype=float)
                        oof_help = np.full(len(train_idx), 0.5, dtype=float)
                        for inner_train, inner_val in inner.split(X_train, (y_train != 1).astype(int)):
                            model_res = make_lr(c_res, w_res, multi_class=False)
                            model_res.fit(X_train.iloc[inner_train], (y_train[inner_train] != 1).astype(int))
                            oof_res[inner_val] = model_res.predict_proba(X_train.iloc[inner_val])[:, 1]
                            direction_mask = np.isin(y_train[inner_train], list(RESOLVED))
                            if direction_mask.sum() > 0 and len(np.unique(y_train[inner_train][direction_mask])) == 2:
                                model_dir = make_lr(c_dir, w_dir, multi_class=False)
                                model_dir.fit(
                                    X_train.iloc[inner_train].iloc[direction_mask],
                                    (y_train[inner_train][direction_mask] == 2).astype(int),
                                )
                                oof_help[inner_val] = model_dir.predict_proba(X_train.iloc[inner_val])[:, 1]
                        rt, dt, m = threshold_search(y_train, oof_res, oof_help, objective, target_coverage)
                        key = objective_key(m, objective, target_coverage)
                        if best is None or key > best[0]:
                            best = (key, TwoStageSpec(c_res, w_res, c_dir, w_dir, rt, dt), m)
        spec = best[1]
        model_res = make_lr(spec.c_resolved, spec.w_resolved, multi_class=False)
        model_res.fit(X.iloc[train_idx], (y[train_idx] != 1).astype(int))
        direction_mask = np.isin(y[train_idx], list(RESOLVED))
        model_dir = make_lr(spec.c_direction, spec.w_direction, multi_class=False)
        model_dir.fit(X.iloc[train_idx].iloc[direction_mask], (y[train_idx][direction_mask] == 2).astype(int))
        res_prob = model_res.predict_proba(X.iloc[test_idx])[:, 1]
        help_prob = model_dir.predict_proba(X.iloc[test_idx])[:, 1]
        fold_pred = pred_two_stage(res_prob, help_prob, spec.resolved_threshold, spec.direction_threshold)
        pred[test_idx] = fold_pred
        rows.append(
            {
                "fold": fold,
                "objective": objective,
                "c_resolved": spec.c_resolved,
                "w_resolved": spec.w_resolved or "none",
                "c_direction": spec.c_direction,
                "w_direction": spec.w_direction or "none",
                "resolved_threshold": spec.resolved_threshold,
                "direction_threshold": spec.direction_threshold,
                **{f"inner_{k}": v for k, v in best[2].items()},
                **{f"test_{k}": v for k, v in metrics(y[test_idx], fold_pred).items()},
            }
        )
    return pred, rows


def majority_baselines(df: pd.DataFrame) -> dict[str, np.ndarray]:
    shares = df[["vote_not_helpful", "vote_somewhat_helpful", "vote_helpful"]].to_numpy(dtype=float)
    raw_majority = shares.argmax(axis=1)
    # raw labels: 0=NH, 1=somewhat, 2=H; map somewhat to NMR as a naive official-schema baseline.
    return {
        "always_nmr": np.full(len(df), 1, dtype=int),
        "raw_vote_majority_somewhat_as_nmr": raw_majority,
        "raw_score_thresholds_0p33_0p67": np.where(df["mean_raw_score"].to_numpy() >= 2 / 3, 2, np.where(df["mean_raw_score"].to_numpy() <= 1 / 3, 0, 1)),
        "raw_score_thresholds_0p25_0p75": np.where(df["mean_raw_score"].to_numpy() >= 0.75, 2, np.where(df["mean_raw_score"].to_numpy() <= 0.25, 0, 1)),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "officialschema_nested_cv_aggregation_20260512"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    y = df["true_label_3way"].to_numpy(dtype=int)
    predictions = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    rows = []
    fold_frames = []

    for name, pred in majority_baselines(df).items():
        predictions[name] = pred
        rows.append({"method": name, "family": "baseline", **metrics(y, pred)})

    for feature_set_name in ["summary", "full_agent", "full_agent_plus_metadata"]:
        cols = feature_sets[feature_set_name]
        for objective in ["accuracy", "balanced"]:
            name = f"multiclass_lr_{feature_set_name}_{objective}"
            pred, prob, fold_rows = nested_multiclass_lr(
                df, cols, args.folds, args.inner_folds, args.seed, objective, args.target_coverage
            )
            predictions[name] = pred
            predictions[f"{name}_prob_not_helpful"] = prob[:, 0]
            predictions[f"{name}_prob_nmr"] = prob[:, 1]
            predictions[f"{name}_prob_helpful"] = prob[:, 2]
            rows.append(
                {
                    "method": name,
                    "family": "nested_multiclass_lr",
                    "feature_set": feature_set_name,
                    "objective": objective,
                    "n_features": len(cols),
                    **metrics(y, pred),
                }
            )
            fold_frames.append(pd.DataFrame(fold_rows).assign(method=name))
        for objective in ["target_coverage", "resolved_precision"]:
            name = f"two_stage_lr_{feature_set_name}_{objective}"
            pred, fold_rows = nested_two_stage_lr(
                df, cols, args.folds, args.inner_folds, args.seed, objective, args.target_coverage
            )
            predictions[name] = pred
            rows.append(
                {
                    "method": name,
                    "family": "nested_two_stage_lr",
                    "feature_set": feature_set_name,
                    "objective": objective,
                    "n_features": len(cols),
                    **metrics(y, pred),
                }
            )
            fold_frames.append(pd.DataFrame(fold_rows).assign(method=name))

    summary = pd.DataFrame(rows)
    for col in [
        "accuracy",
        "balanced_accuracy",
        "recall_not_helpful",
        "recall_needs_more_ratings",
        "recall_helpful",
        "pred_resolved_coverage",
        "strict_accuracy_on_pred_resolved",
        "binary_accuracy_true_resolved_selected",
    ]:
        summary[f"{col}_pct"] = pd.to_numeric(summary[col], errors="coerce") * 100.0
    summary = summary.sort_values(["accuracy", "balanced_accuracy"], ascending=False)
    df.to_csv(out_dir / "officialschema_feature_table.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "officialschema_nested_cv_oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "officialschema_nested_cv_summary.csv", index=False, encoding="utf-8-sig")
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(
            out_dir / "officialschema_nested_cv_fold_metrics.csv", index=False, encoding="utf-8-sig"
        )

    metadata = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_notes": int(len(df)),
        "folds": int(args.folds),
        "inner_folds": int(args.inner_folds),
        "seed": int(args.seed),
        "target_coverage": float(args.target_coverage),
        "true_distribution": df["true_label_text"].value_counts().to_dict(),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "best_by_accuracy": summary.iloc[0].to_dict(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "method",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "pred_resolved_coverage_pct",
        "strict_accuracy_on_pred_resolved_pct",
        "binary_accuracy_true_resolved_selected_pct",
        "recall_not_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_helpful_pct",
    ]
    print(summary[cols].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
