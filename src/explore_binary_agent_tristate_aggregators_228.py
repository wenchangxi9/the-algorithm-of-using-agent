from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


def clean_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean01(s: pd.Series) -> pd.Series:
    return clean_num(s).fillna(0).clip(0, 1)


def safe_mean(g: pd.DataFrame, col: str) -> float:
    if len(g) == 0:
        return np.nan
    return float(pd.to_numeric(g[col], errors="coerce").mean())


def load_feature_table(run_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    votes = pd.read_csv(run_dir / "agent_votes.csv", low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["agent_id"] = votes["agent_id"].astype(str)
    votes["raw_binary"] = votes["parsed_rating"].map(RAW)
    votes = votes[votes["raw_binary"].isin([0, 1])].copy()
    votes["true_label_3way"] = clean_num(votes["true_label_3way"]).astype(int)
    votes["score"] = clean_num(votes["predicted_rating_score"]).fillna(votes["raw_binary"])
    for col in ["confidence", "changes_reader_understanding"]:
        if col not in votes.columns:
            votes[col] = np.nan
        votes[col] = clean_num(votes[col])
    for col in HELPFUL_REASONS + NOT_HELPFUL_REASONS:
        if col not in votes.columns:
            votes[col] = 0
        votes[col] = clean01(votes[col])
    votes["is_h"] = (votes["raw_binary"] == 1).astype(float)
    votes["is_nh"] = (votes["raw_binary"] == 0).astype(float)
    votes["helpful_reason_sum"] = votes[HELPFUL_REASONS].sum(axis=1)
    votes["not_helpful_reason_sum"] = votes[NOT_HELPFUL_REASONS].sum(axis=1)
    votes["reason_conflict_h_vote"] = ((votes["raw_binary"] == 1) & (votes["not_helpful_reason_sum"] > 0)).astype(float)
    votes["reason_conflict_nh_vote"] = ((votes["raw_binary"] == 0) & (votes["helpful_reason_sum"] > 0)).astype(float)

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
        "min_confidence": ("confidence", "min"),
        "max_confidence": ("confidence", "max"),
        "mean_understanding": ("changes_reader_understanding", "mean"),
        "std_understanding": ("changes_reader_understanding", "std"),
        "min_understanding": ("changes_reader_understanding", "min"),
        "max_understanding": ("changes_reader_understanding", "max"),
        "helpful_reason_sum": ("helpful_reason_sum", "mean"),
        "not_helpful_reason_sum": ("not_helpful_reason_sum", "mean"),
        "reason_conflict_h_vote_rate": ("reason_conflict_h_vote", "mean"),
        "reason_conflict_nh_vote_rate": ("reason_conflict_nh_vote", "mean"),
    }
    for col in HELPFUL_REASONS + NOT_HELPFUL_REASONS:
        agg[f"{col}_rate"] = (col, "mean")
    df = votes.groupby("noteId", as_index=False).agg(**agg)
    for col in [
        "std_score",
        "std_confidence",
        "min_confidence",
        "max_confidence",
        "std_understanding",
        "min_understanding",
        "max_understanding",
    ]:
        df[col] = df[col].fillna(0)
    df["share_h"] = df["vote_h"] / df["n_votes"].clip(lower=1)
    df["share_nh"] = df["vote_nh"] / df["n_votes"].clip(lower=1)
    df["h_minus_nh"] = df["share_h"] - df["share_nh"]
    df["h_nh_margin"] = df["h_minus_nh"].abs()
    df["vote_entropy_binary"] = -(
        df["share_h"].clip(1e-9, 1) * np.log(df["share_h"].clip(1e-9, 1))
        + df["share_nh"].clip(1e-9, 1) * np.log(df["share_nh"].clip(1e-9, 1))
    )
    df["near_tie"] = (df["h_nh_margin"] <= 0.125).astype(float)

    conditional_rows = []
    for note_id, group in votes.groupby("noteId", sort=False):
        h = group[group["raw_binary"] == 1]
        nh = group[group["raw_binary"] == 0]
        conditional_rows.append(
            {
                "noteId": note_id,
                "h_vote_mean_confidence": safe_mean(h, "confidence"),
                "nh_vote_mean_confidence": safe_mean(nh, "confidence"),
                "h_vote_mean_understanding": safe_mean(h, "changes_reader_understanding"),
                "nh_vote_mean_understanding": safe_mean(nh, "changes_reader_understanding"),
                "h_vote_helpful_reason_sum": safe_mean(h, "helpful_reason_sum"),
                "h_vote_not_helpful_reason_sum": safe_mean(h, "not_helpful_reason_sum"),
                "nh_vote_helpful_reason_sum": safe_mean(nh, "helpful_reason_sum"),
                "nh_vote_not_helpful_reason_sum": safe_mean(nh, "not_helpful_reason_sum"),
            }
        )
    df = df.merge(pd.DataFrame(conditional_rows), on="noteId", how="left")
    df["conf_h_minus_nh"] = df["h_vote_mean_confidence"] - df["nh_vote_mean_confidence"]
    df["understanding_h_minus_nh"] = df["h_vote_mean_understanding"] - df["nh_vote_mean_understanding"]

    agent = votes.pivot_table(index="noteId", columns="agent_id", values="raw_binary", aggfunc="first")
    agent.columns = [f"agent_{c}_is_h" for c in agent.columns]
    df = df.merge(agent.reset_index(), on="noteId", how="left")

    groups = {
        "vote": [
            "n_votes",
            "vote_h",
            "vote_nh",
            "mean_score",
            "std_score",
            "share_h",
            "share_nh",
            "h_minus_nh",
            "h_nh_margin",
            "vote_entropy_binary",
            "near_tie",
        ],
        "confidence": ["mean_confidence", "std_confidence", "min_confidence", "max_confidence", "conf_h_minus_nh"],
        "understanding": [
            "mean_understanding",
            "std_understanding",
            "min_understanding",
            "max_understanding",
            "understanding_h_minus_nh",
        ],
        "official_reasons": [f"{c}_rate" for c in HELPFUL_REASONS + NOT_HELPFUL_REASONS]
        + [
            "helpful_reason_sum",
            "not_helpful_reason_sum",
            "reason_conflict_h_vote_rate",
            "reason_conflict_nh_vote_rate",
            "h_vote_helpful_reason_sum",
            "h_vote_not_helpful_reason_sum",
            "nh_vote_helpful_reason_sum",
            "nh_vote_not_helpful_reason_sum",
        ],
        "per_agent": [c for c in df.columns if c.startswith("agent_") and c.endswith("_is_h")],
    }
    return df.sort_values("noteId").reset_index(drop=True), groups


def uniq(*items: list[str]) -> list[str]:
    out = []
    for item in items:
        for col in item:
            if col not in out:
                out.append(col)
    return out


def make_lr(c: float, weight: str | None, y: np.ndarray | None = None) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight=weight, max_iter=5000, random_state=42)),
        ]
    )


def make_svc(c: float, gamma: str) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("clf", SVC(C=c, gamma=gamma, class_weight="balanced", probability=False, random_state=42)),
        ]
    )


def make_rf(max_depth: int | None, min_leaf: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_extra(max_depth: int | None, min_leaf: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_gb(learning_rate: float, max_depth: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=seed,
                ),
            ),
        ]
    )


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
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


def obj(m: dict[str, float | int]) -> tuple[float, float, int]:
    return (float(m["balanced_accuracy"]), float(m["accuracy"]), -int(m["h_to_nh"]) - int(m["nh_to_h"]))


def nested_direct_classifier(
    df: pd.DataFrame,
    features: list[str],
    candidates: list[tuple[str, Callable[[int], Pipeline]]],
    folds: int,
    inner_folds: int,
    seed: int,
):
    y = df["true_label_3way"].to_numpy(int)
    x = df[features]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        for name, factory in candidates:
            oof = np.zeros(len(tr), dtype=int)
            xtr = x.iloc[tr].reset_index(drop=True)
            ytr = y[tr]
            for a, b in inner.split(xtr, ytr):
                model = factory(seed + fold * 100 + len(name))
                model.fit(xtr.iloc[a], ytr[a])
                oof[b] = model.predict(xtr.iloc[b])
            m = metric(ytr, oof)
            key = obj(m)
            if best is None or key > best[0]:
                best = (key, name, factory, m)
        assert best is not None
        _, name, factory, inner_metric = best
        final = factory(seed + fold * 1000)
        final.fit(x.iloc[tr], y[tr])
        pred[te] = final.predict(x.iloc[te])
        rows.append(
            {
                "fold": fold,
                "selected_candidate": name,
                **{f"inner_{k}": v for k, v in inner_metric.items()},
                **{f"test_{k}": v for k, v in metric(y[te], pred[te]).items()},
            }
        )
    return pred, pd.DataFrame(rows)


def tune_disagreement_rule(train: pd.DataFrame, y: np.ndarray):
    best = None
    for margin in np.arange(0.00, 0.55, 0.025):
        for entropy in np.arange(0.10, 0.71, 0.025):
            for conf_low in [0, 45, 55, 65, 75]:
                for understand_low in [0, 20, 30, 40, 50]:
                    pred = np.where(train["share_h"].to_numpy() >= 0.5, 2, 0)
                    nmr = (train["h_nh_margin"].to_numpy() <= margin) | (
                        train["vote_entropy_binary"].to_numpy() >= entropy
                    )
                    if conf_low:
                        nmr |= train["mean_confidence"].fillna(100).to_numpy() <= conf_low
                    if understand_low:
                        nmr |= train["mean_understanding"].fillna(100).to_numpy() <= understand_low
                    pred[nmr] = 1
                    m = metric(y, pred)
                    key = obj(m)
                    if best is None or key > best[0]:
                        best = (key, margin, entropy, conf_low, understand_low, m)
    assert best is not None
    return best[1:]


def nested_disagreement_rule(df: pd.DataFrame, folds: int, seed: int):
    y = df["true_label_3way"].to_numpy(int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(df, y), start=1):
        margin, entropy, conf_low, understand_low, inner_metric = tune_disagreement_rule(df.iloc[tr], y[tr])
        test = df.iloc[te]
        p = np.where(test["share_h"].to_numpy() >= 0.5, 2, 0)
        nmr = (test["h_nh_margin"].to_numpy() <= margin) | (test["vote_entropy_binary"].to_numpy() >= entropy)
        if conf_low:
            nmr |= test["mean_confidence"].fillna(100).to_numpy() <= conf_low
        if understand_low:
            nmr |= test["mean_understanding"].fillna(100).to_numpy() <= understand_low
        p[nmr] = 1
        pred[te] = p
        rows.append(
            {
                "fold": fold,
                "margin": margin,
                "entropy": entropy,
                "conf_low": conf_low,
                "understand_low": understand_low,
                **{f"inner_{k}": v for k, v in inner_metric.items()},
                **{f"test_{k}": v for k, v in metric(y[te], p).items()},
            }
        )
    return pred, pd.DataFrame(rows)


def prob_for_class(pipe: Pipeline, x: pd.DataFrame, cls: int) -> np.ndarray:
    classes = list(pipe.named_steps["clf"].classes_)
    prob = pipe.predict_proba(x)
    return prob[:, classes.index(cls)]


def tune_two_stage(y: np.ndarray, p_nmr: np.ndarray, p_h: np.ndarray):
    best = None
    for nmr_t in np.arange(0.15, 0.86, 0.025):
        for h_t in np.arange(0.20, 0.81, 0.025):
            pred = np.where(p_nmr >= nmr_t, 1, np.where(p_h >= h_t, 2, 0))
            m = metric(y, pred)
            key = obj(m)
            if best is None or key > best[0]:
                best = (key, float(nmr_t), float(h_t), m)
    assert best is not None
    return best[1:]


def nested_two_stage(df: pd.DataFrame, features: list[str], folds: int, inner_folds: int, seed: int):
    y = df["true_label_3way"].to_numpy(int)
    x = df[features]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    rows = []
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weights = [None, "balanced"]
    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        xtr = x.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        for c_nmr in c_grid:
            for w_nmr in weights:
                for c_h in c_grid:
                    for w_h in weights:
                        oof_nmr = np.zeros(len(ytr), dtype=float)
                        oof_h = np.zeros(len(ytr), dtype=float)
                        ok = True
                        for a, b in inner.split(xtr, ytr):
                            y_nmr = (ytr[a] == 1).astype(int)
                            resolved = np.isin(ytr[a], [0, 2])
                            if resolved.sum() < 4 or len(np.unique(ytr[a][resolved])) < 2:
                                ok = False
                                break
                            y_h = (ytr[a][resolved] == 2).astype(int)
                            m_nmr = make_lr(c_nmr, w_nmr)
                            m_h = make_lr(c_h, w_h)
                            m_nmr.fit(xtr.iloc[a], y_nmr)
                            m_h.fit(xtr.iloc[a].iloc[resolved], y_h)
                            oof_nmr[b] = prob_for_class(m_nmr, xtr.iloc[b], 1)
                            oof_h[b] = prob_for_class(m_h, xtr.iloc[b], 1)
                        if not ok:
                            continue
                        nmr_t, h_t, im = tune_two_stage(ytr, oof_nmr, oof_h)
                        key = obj(im)
                        if best is None or key > best[0]:
                            best = (key, c_nmr, w_nmr, c_h, w_h, nmr_t, h_t, im)
        assert best is not None
        _, c_nmr, w_nmr, c_h, w_h, nmr_t, h_t, im = best
        m_nmr = make_lr(c_nmr, w_nmr)
        m_h = make_lr(c_h, w_h)
        m_nmr.fit(x.iloc[tr], (y[tr] == 1).astype(int))
        resolved = np.isin(y[tr], [0, 2])
        m_h.fit(x.iloc[tr].iloc[resolved], (y[tr][resolved] == 2).astype(int))
        p_nmr = prob_for_class(m_nmr, x.iloc[te], 1)
        p_h = prob_for_class(m_h, x.iloc[te], 1)
        p = np.where(p_nmr >= nmr_t, 1, np.where(p_h >= h_t, 2, 0))
        pred[te] = p
        rows.append(
            {
                "fold": fold,
                "c_nmr": c_nmr,
                "w_nmr": w_nmr or "none",
                "c_h": c_h,
                "w_h": w_h or "none",
                "nmr_threshold": nmr_t,
                "h_threshold": h_t,
                **{f"inner_{k}": v for k, v in im.items()},
                **{f"test_{k}": v for k, v in metric(y[te], p).items()},
            }
        )
    return pred, pd.DataFrame(rows)


def save_method(out_dir: Path, name: str, df: pd.DataFrame, pred: np.ndarray, folds: pd.DataFrame) -> dict[str, float | int | str]:
    y = df["true_label_3way"].to_numpy(int)
    m = metric(y, pred)
    row = {"method": name, **m}
    pd.DataFrame({"noteId": df["noteId"], "true_label_3way": y, "true_label_text": df["true_label_text"], "pred_label_3way": pred, "pred_label_text": pd.Series(pred).map(LABEL3)}).to_csv(
        out_dir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(out_dir / f"{name}_folds.csv", index=False, encoding="utf-8-sig")
    cm = pd.DataFrame(
        confusion_matrix(y, pred, labels=[2, 1, 0]),
        index=["true_HELPFUL", "true_NMR", "true_NOT_HELPFUL"],
        columns=["pred_HELPFUL", "pred_NMR", "pred_NOT_HELPFUL"],
    )
    cm.to_csv(out_dir / f"{name}_confusion.csv", encoding="utf-8-sig")
    return row


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "tristate_aggregator_exploration_20260513"
    out_dir.mkdir(exist_ok=True)
    df, groups = load_feature_table(run_dir)
    features = {
        "vote": groups["vote"],
        "compact": uniq(groups["vote"], groups["confidence"], groups["understanding"], groups["official_reasons"]),
        "full": uniq(groups["vote"], groups["confidence"], groups["understanding"], groups["official_reasons"], groups["per_agent"]),
        "no_per_agent": uniq(groups["vote"], groups["confidence"], groups["understanding"], groups["official_reasons"]),
        "vote_reason": uniq(groups["vote"], groups["official_reasons"]),
    }
    rows = []
    pred_table = df[["noteId", "true_label_3way", "true_label_text"]].copy()

    y = df["true_label_3way"].to_numpy(int)
    raw = np.where(df["share_h"].to_numpy() >= 0.5, 2, 0)
    rows.append({"method": "raw_binary_majority_no_nmr", **metric(y, raw)})
    pred_table["raw_binary_majority_no_nmr"] = raw

    print("[1] disagreement rule", flush=True)
    pred, folds = nested_disagreement_rule(df, args.folds, args.seed + 1)
    rows.append(save_method(out_dir, "disagreement_rule", df, pred, folds))
    pred_table["disagreement_rule"] = pred

    method_idx = 2
    for feat_name, cols in features.items():
        lr_candidates = []
        for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            for w in [None, "balanced"]:
                lr_candidates.append((f"lr_C{c}_{w or 'none'}", lambda seed, c=c, w=w: make_lr(c, w)))
        print(f"[{method_idx}] direct LR {feat_name}", flush=True)
        pred, folds = nested_direct_classifier(df, cols, lr_candidates, args.folds, args.inner_folds, args.seed + method_idx)
        name = f"direct_lr_{feat_name}"
        rows.append(save_method(out_dir, name, df, pred, folds))
        pred_table[name] = pred
        method_idx += 1

    tree_candidates = []
    for depth in [2, 3, 4, None]:
        for leaf in [1, 3, 5, 8]:
            tree_candidates.append((f"rf_d{depth}_l{leaf}", lambda seed, depth=depth, leaf=leaf: make_rf(depth, leaf, seed)))
            tree_candidates.append((f"extra_d{depth}_l{leaf}", lambda seed, depth=depth, leaf=leaf: make_extra(depth, leaf, seed)))
    for lr in [0.03, 0.05, 0.08]:
        for depth in [1, 2, 3]:
            tree_candidates.append((f"gb_lr{lr}_d{depth}", lambda seed, lr=lr, depth=depth: make_gb(lr, depth, seed)))
    print(f"[{method_idx}] direct tree full", flush=True)
    pred, folds = nested_direct_classifier(df, features["full"], tree_candidates, args.folds, args.inner_folds, args.seed + method_idx)
    rows.append(save_method(out_dir, "direct_tree_full", df, pred, folds))
    pred_table["direct_tree_full"] = pred
    method_idx += 1

    svc_candidates = []
    for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
        for gamma in ["scale", "auto"]:
            svc_candidates.append((f"svc_C{c}_{gamma}", lambda seed, c=c, gamma=gamma: make_svc(c, gamma)))
    print(f"[{method_idx}] direct SVC compact", flush=True)
    pred, folds = nested_direct_classifier(df, features["compact"], svc_candidates, args.folds, args.inner_folds, args.seed + method_idx)
    rows.append(save_method(out_dir, "direct_svc_compact", df, pred, folds))
    pred_table["direct_svc_compact"] = pred
    method_idx += 1

    for feat_name in ["vote", "compact", "full", "vote_reason"]:
        print(f"[{method_idx}] two-stage {feat_name}", flush=True)
        pred, folds = nested_two_stage(df, features[feat_name], args.folds, args.inner_folds, args.seed + method_idx)
        name = f"two_stage_lr_{feat_name}"
        rows.append(save_method(out_dir, name, df, pred, folds))
        pred_table[name] = pred
        method_idx += 1

    summary = pd.DataFrame(rows)
    for col in [
        "accuracy",
        "balanced_accuracy",
        "resolved_coverage",
        "recall_not_helpful",
        "recall_needs_more_ratings",
        "recall_helpful",
    ]:
        summary[f"{col}_pct"] = summary[col] * 100
    summary = summary.sort_values(["balanced_accuracy", "accuracy"], ascending=[False, False])
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pred_table.to_csv(out_dir / "all_method_predictions.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out_dir / "feature_table.csv", index=False, encoding="utf-8-sig")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "feature_sets": {k: v for k, v in features.items()},
                "note": "All reported methods use outer-fold predictions. True labels are used only inside each outer training fold for model/threshold selection.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    show = [
        "method",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "recall_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_not_helpful_pct",
        "resolved_coverage_pct",
        "h_to_nh",
        "nh_to_h",
    ]
    print("\n=== Ranked summary ===")
    print(summary[show].to_string(index=False))
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
