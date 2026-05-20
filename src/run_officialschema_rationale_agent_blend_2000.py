from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_officialschema_nested_cv_aggregation_2000 import build_features


LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}
RATING_TO_ID = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}
RATING_NAME = {0: "nh", 1: "nmr", 2: "h"}

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("artifacts/05_agent_runs_2000"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", choices=["balanced", "accuracy", "joint"], default="balanced")
    p.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    p.add_argument("--weight-grid", nargs="+", default=["none", "balanced", "nmr_up"])
    p.add_argument("--max-word-features", type=int, default=12000)
    p.add_argument("--max-char-features", type=int, default=12000)
    return p.parse_args()


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    recs = []
    for label_id, label in LABEL.items():
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


def score_key(m: dict[str, float | int], objective: str) -> tuple[float, ...]:
    joint = 2.0 * m["accuracy"] * m["balanced_accuracy"] / max(m["accuracy"] + m["balanced_accuracy"], 1e-12)
    if objective == "joint":
        return (joint, m["balanced_accuracy"], m["accuracy"], m["min_recall"], -m["cross_error"])
    if objective == "balanced":
        return (m["balanced_accuracy"], m["accuracy"], m["min_recall"], -m["cross_error"])
    if objective == "accuracy":
        return (m["accuracy"], m["balanced_accuracy"], m["min_recall"], -m["cross_error"])
    raise ValueError(objective)


def unique_cols(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    return value.strip("_")


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


def build_structured_features(votes: pd.DataFrame) -> pd.DataFrame:
    vv = votes.copy()
    vv["noteId"] = vv["noteId"].astype(str)
    vv["parsed_rating"] = vv["parsed_rating"].fillna(vv.get("helpfulnessLevel", "SOMEWHAT_HELPFUL")).astype(str)
    vv["rating_id"] = vv["parsed_rating"].map(RATING_TO_ID)
    vv = vv[vv["rating_id"].isin([0, 1, 2])].copy()
    vv["rating_id"] = vv["rating_id"].astype(int)
    for col in ["confidence", "changes_reader_understanding", "helpfulClear", "helpfulGoodSources", "helpfulAddressesClaim", "helpfulImportantContext", "helpfulUnbiasedLanguage", "notHelpfulIncorrect", "notHelpfulSourcesMissingOrUnreliable", "notHelpfulMissingKeyPoints", "notHelpfulHardToUnderstand", "notHelpfulArgumentativeOrBiased", "notHelpfulIrrelevantSources", "notHelpfulOpinionSpeculation", "notHelpfulNoteNotNeeded"]:
        vv[col] = pd.to_numeric(vv.get(col), errors="coerce")
    vv["confidence"] = vv["confidence"].fillna(0.0)
    vv["changes_reader_understanding"] = vv["changes_reader_understanding"].fillna(0.0)
    vv["rationale_len"] = vv.get("rationale", "").fillna(0).astype(str).str.len().astype(float)

    base = vv.groupby("noteId", as_index=False).agg(
        n_votes=("agent_id", "size"),
        vote_nh=("rating_id", lambda s: int((s == 0).sum())),
        vote_nmr=("rating_id", lambda s: int((s == 1).sum())),
        vote_h=("rating_id", lambda s: int((s == 2).sum())),
        mean_confidence=("confidence", "mean"),
        std_confidence=("confidence", "std"),
        mean_changes_reader_understanding=("changes_reader_understanding", "mean"),
        std_changes_reader_understanding=("changes_reader_understanding", "std"),
        mean_rationale_len=("rationale_len", "mean"),
        std_rationale_len=("rationale_len", "std"),
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
    base["std_confidence"] = base["std_confidence"].fillna(0.0)
    base["std_changes_reader_understanding"] = base["std_changes_reader_understanding"].fillna(0.0)
    base["std_rationale_len"] = base["std_rationale_len"].fillna(0.0)
    for name in ["nh", "nmr", "h"]:
        base[f"share_{name}"] = base[f"vote_{name}"] / base["n_votes"].clip(lower=1)
    safe = np.clip(base[["share_nh", "share_nmr", "share_h"]].to_numpy(dtype=float), 1e-9, 1.0)
    base["vote_entropy"] = -(safe * np.log(safe)).sum(axis=1)
    base["vote_margin"] = np.sort(safe, axis=1)[:, -1] - np.sort(safe, axis=1)[:, -2]
    base["vote_resolved_mass"] = base["share_nh"] + base["share_h"]
    base["vote_nmr_gap"] = base["share_nmr"] - np.maximum(base["share_nh"], base["share_h"])

    def count_by_group(group_col: str, prefix: str) -> pd.DataFrame:
        tmp = vv[["noteId", group_col, "rating_id"]].copy()
        tmp[group_col] = tmp[group_col].fillna("UNKNOWN").astype(str)
        piv = tmp.assign(v=1).pivot_table(index="noteId", columns=[group_col, "rating_id"], values="v", aggfunc="sum", fill_value=0)
        if not isinstance(piv.columns, pd.MultiIndex):
            piv.columns = [f"{prefix}_{slugify(str(c))}_n{RATING_NAME.get(0, 'nh')}" for c in piv.columns]
            return piv.reset_index()
        cols = []
        for group_value, rating_id in piv.columns:
            cols.append(f"{prefix}_{slugify(str(group_value))}_{RATING_NAME.get(int(rating_id), 'x')}")
        piv.columns = cols
        return piv.reset_index()

    persona_counts = count_by_group("persona_name", "persona")
    cluster_counts = count_by_group("cluster", "cluster")
    out = base.merge(persona_counts, on="noteId", how="left").merge(cluster_counts, on="noteId", how="left")
    out = out.fillna(0.0)
    return out.sort_values("noteId").reset_index(drop=True)


def build_rationale_text(votes: pd.DataFrame) -> pd.DataFrame:
    vv = votes.copy()
    vv["noteId"] = vv["noteId"].astype(str)
    vv["parsed_rating"] = vv["parsed_rating"].fillna(vv.get("helpfulnessLevel", "SOMEWHAT_HELPFUL")).astype(str)
    vv["rating_id"] = vv["parsed_rating"].map(RATING_TO_ID)
    vv = vv[vv["rating_id"].isin([0, 1, 2])].copy()
    vv["rating_id"] = vv["rating_id"].astype(int)
    for col in ["confidence", "changes_reader_understanding"]:
        vv[col] = pd.to_numeric(vv.get(col), errors="coerce").fillna(0.0)
    vv["rationale"] = vv.get("rationale", "").fillna(0).astype(str)
    vv["persona_name"] = vv.get("persona_name", "UNKNOWN").fillna("UNKNOWN").astype(str)
    vv["cluster"] = vv.get("cluster", "UNKNOWN").fillna("UNKNOWN").astype(str)
    vv["conf_bin"] = pd.cut(vv["confidence"], bins=[-np.inf, 20, 40, 60, 80, np.inf], labels=["vlow", "low", "mid", "high", "vhigh"]).astype(str)
    vv["under_bin"] = pd.cut(vv["changes_reader_understanding"], bins=[-np.inf, 20, 40, 60, 80, np.inf], labels=["u0", "u1", "u2", "u3", "u4"]).astype(str)
    vv["piece"] = vv.apply(
        lambda r: (
            f"[{r['parsed_rating']}] [P={slugify(str(r['persona_name']))}] [C={slugify(str(r['cluster']))}] "
            f"[CF={r['conf_bin']}] [UD={r['under_bin']}] {r['rationale']}"
        ),
        axis=1,
    )
    txt = vv.groupby("noteId", as_index=False)["piece"].apply(lambda s: " [SEP] ".join(s.tolist())).rename(columns={"piece": "rationale_text"})
    return txt.sort_values("noteId").reset_index(drop=True)


def make_numeric_model(c: float, class_weight, seed: int) -> Pipeline:
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


def make_text_model(c: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=seed,
    )


def align_prob(model: Pipeline | LogisticRegression, X) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((X.shape[0], 3), dtype=float)
    for i, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, i]
    return out


def build_text_mats(train_text: list[str], test_text: list[str], max_word_features: int, max_char_features: int):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=max_word_features, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.98, max_features=max_char_features, sublinear_tf=True)
    Xw_tr = word.fit_transform(train_text)
    Xc_tr = char.fit_transform(train_text)
    Xtr = sparse.hstack([Xw_tr, Xc_tr]).tocsr()
    Xw_te = word.transform(test_text)
    Xc_te = char.transform(test_text)
    Xte = sparse.hstack([Xw_te, Xc_te]).tocsr()
    return Xtr, Xte


def choose_numeric_spec(X_train: pd.DataFrame, y_train: np.ndarray, inner_folds: int, seed: int, objective: str) -> tuple[dict, dict]:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    c_grid = [0.1, 0.3, 1.0, 3.0]
    weight_map = {
        "none": None,
        "balanced": "balanced",
        "nmr_up": {0: 1.0, 1: 1.2, 2: 1.0},
    }
    best = None
    for c in c_grid:
        for w_name in weight_map:
            w = weight_map[w_name]
            oof = np.zeros(len(y_train), dtype=int)
            for tr, va in inner.split(X_train, y_train):
                model = make_numeric_model(c, w, seed)
                model.fit(X_train.iloc[tr], y_train[tr])
                oof[va] = model.predict(X_train.iloc[va])
            m = metric(y_train, oof)
            key = score_key(m, objective)
            if best is None or key > best[0]:
                best = (key, {"c": c, "class_weight": w_name}, m)
    assert best is not None
    return best[1], best[2]


def numeric_oof_probs(X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: list[str], spec: dict, inner_folds: int, seed: int) -> np.ndarray:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y_train), 3), dtype=float)
    weight_map = {"none": None, "balanced": "balanced", "nmr_up": {0: 1.0, 1: 1.2, 2: 1.0}}
    for tr, va in inner.split(X_train, y_train):
        model = make_numeric_model(spec["c"], weight_map[spec["class_weight"]], seed)
        model.fit(X_train.iloc[tr][feature_cols], y_train[tr])
        oof[va] = align_prob(model, X_train.iloc[va][feature_cols])
    return oof


def fit_numeric_test(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, feature_cols: list[str], spec: dict, seed: int) -> np.ndarray:
    weight_map = {"none": None, "balanced": "balanced", "nmr_up": {0: 1.0, 1: 1.2, 2: 1.0}}
    model = make_numeric_model(spec["c"], weight_map[spec["class_weight"]], seed)
    model.fit(X_train[feature_cols], y_train)
    return align_prob(model, X_test[feature_cols])


def choose_text_spec(
    texts: pd.Series,
    y_train: np.ndarray,
    inner_folds: int,
    seed: int,
    objective: str,
    max_word_features: int,
    max_char_features: int,
) -> dict:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best = None
    for c in [0.3, 1.0, 3.0]:
        oof = np.zeros((len(y_train), 3), dtype=float)
        for tr, va in inner.split(texts, y_train):
            Xtr, Xva = build_text_mats(texts.iloc[tr].tolist(), texts.iloc[va].tolist(), max_word_features, max_char_features)
            model = make_text_model(c, seed)
            model.fit(Xtr, y_train[tr])
            oof[va] = align_prob(model, Xva)
        m = metric(y_train, oof.argmax(axis=1))
        key = score_key(m, objective)
        if best is None or key > best[0]:
            best = (key, {"c": c}, m)
    assert best is not None
    return best[1]


def text_oof_probs(
    texts: pd.Series,
    y_train: np.ndarray,
    spec: dict,
    inner_folds: int,
    seed: int,
    max_word_features: int,
    max_char_features: int,
) -> np.ndarray:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y_train), 3), dtype=float)
    for tr, va in inner.split(texts, y_train):
        Xtr, Xva = build_text_mats(texts.iloc[tr].tolist(), texts.iloc[va].tolist(), max_word_features, max_char_features)
        model = make_text_model(spec["c"], seed)
        model.fit(Xtr, y_train[tr])
        oof[va] = align_prob(model, Xva)
    return oof


def fit_text_test(
    train_text: pd.Series,
    y_train: np.ndarray,
    test_text: pd.Series,
    spec: dict,
    seed: int,
    max_word_features: int,
    max_char_features: int,
) -> np.ndarray:
    Xtr, Xte = build_text_mats(train_text.tolist(), test_text.tolist(), max_word_features, max_char_features)
    model = make_text_model(spec["c"], seed)
    model.fit(Xtr, y_train)
    return align_prob(model, Xte)


def blend_probs(prob_map: dict[str, np.ndarray], methods: list[str], weights: tuple[float, ...]) -> np.ndarray:
    out = np.zeros_like(next(iter(prob_map.values())))
    for method, w in zip(methods, weights):
        out += w * prob_map[method]
    return out


def choose_blend(y_train: np.ndarray, oof_probs: dict[str, np.ndarray], objective: str) -> tuple[list[str], tuple[float, ...], dict]:
    weight_grid = [0.5, 1.0, 2.0]
    methods_all = list(oof_probs.keys())
    best = None
    for r in [2, 3, 4]:
        for methods in combinations(methods_all, r):
            for weights in product(weight_grid, repeat=r):
                pred = blend_probs(oof_probs, list(methods), weights).argmax(axis=1)
                m = metric(y_train, pred)
                key = score_key(m, objective)
                if best is None or key > best[0]:
                    best = (key, list(methods), tuple(float(w) for w in weights), m)
    assert best is not None
    return best[1], best[2], best[3]


def nested_cv(
    df: pd.DataFrame,
    structured_df: pd.DataFrame,
    text_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    folds: int,
    inner_folds: int,
    seed: int,
    objective: str,
    max_word_features: int,
    max_char_features: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    y = df["true_label_3way"].to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    methods = [
        "summary_meta",
        "summary_meta_struct",
        "full_meta",
        "rationale_text",
        "blend",
    ]
    preds = {m: np.zeros(len(df), dtype=int) for m in methods}
    rows: list[dict] = []

    for fold, (tr, te) in enumerate(outer.split(df, y), start=1):
        X_tr = df.iloc[tr].reset_index(drop=True)
        X_te = df.iloc[te].reset_index(drop=True)
        y_tr = y[tr]
        y_te = y[te]

        s_cols = feature_sets["summary_plus_meta"]
        fs_cols = feature_sets["full_plus_meta"]
        ss_cols = feature_sets["summary_plus_meta_struct"]
        st_cols = [c for c in structured_df.columns if c != "noteId"]
        note_text_tr = text_df.iloc[tr]["rationale_text"].reset_index(drop=True)
        note_text_te = text_df.iloc[te]["rationale_text"].reset_index(drop=True)

        # summary numeric
        spec_sum, inner_sum_m = choose_numeric_spec(X_tr[s_cols], y_tr, inner_folds, seed + fold, objective)
        oof_sum = numeric_oof_probs(X_tr[s_cols], y_tr, s_cols, spec_sum, inner_folds, seed + fold)
        te_sum = fit_numeric_test(X_tr[s_cols], y_tr, X_te[s_cols], s_cols, spec_sum, seed + fold)
        pred_sum = te_sum.argmax(axis=1)
        preds["summary_meta"][te] = pred_sum

        # structured numeric
        X_tr_struct = pd.concat([X_tr.reset_index(drop=True), structured_df.iloc[tr].drop(columns=["noteId"]).reset_index(drop=True)], axis=1)
        X_te_struct = pd.concat([X_te.reset_index(drop=True), structured_df.iloc[te].drop(columns=["noteId"]).reset_index(drop=True)], axis=1)
        spec_struct, inner_struct_m = choose_numeric_spec(X_tr_struct[ss_cols], y_tr, inner_folds, seed + 100 + fold, objective)
        oof_struct = numeric_oof_probs(X_tr_struct[ss_cols], y_tr, ss_cols, spec_struct, inner_folds, seed + 100 + fold)
        te_struct = fit_numeric_test(X_tr_struct[ss_cols], y_tr, X_te_struct[ss_cols], ss_cols, spec_struct, seed + 100 + fold)
        pred_struct = te_struct.argmax(axis=1)
        preds["summary_meta_struct"][te] = pred_struct

        # full numeric
        spec_full, inner_full_m = choose_numeric_spec(X_tr[fs_cols], y_tr, inner_folds, seed + 200 + fold, objective)
        oof_full = numeric_oof_probs(X_tr[fs_cols], y_tr, fs_cols, spec_full, inner_folds, seed + 200 + fold)
        te_full = fit_numeric_test(X_tr[fs_cols], y_tr, X_te[fs_cols], fs_cols, spec_full, seed + 200 + fold)
        pred_full = te_full.argmax(axis=1)
        preds["full_meta"][te] = pred_full

        # rationale text
        text_spec = choose_text_spec(
            note_text_tr,
            y_tr,
            inner_folds,
            seed + 300 + fold,
            objective,
            max_word_features,
            max_char_features,
        )
        oof_text = text_oof_probs(
            note_text_tr,
            y_tr,
            text_spec,
            inner_folds,
            seed + 300 + fold,
            max_word_features,
            max_char_features,
        )
        te_text = fit_text_test(
            note_text_tr,
            y_tr,
            note_text_te,
            text_spec,
            seed + 300 + fold,
            max_word_features,
            max_char_features,
        )
        pred_text = te_text.argmax(axis=1)
        preds["rationale_text"][te] = pred_text

        # blend on outer-train OOF probabilities
        oof_map = {
            "summary_meta": oof_sum,
            "summary_meta_struct": oof_struct,
            "full_meta": oof_full,
            "rationale_text": oof_text,
        }
        blend_methods, blend_weights, blend_inner_m = choose_blend(y_tr, oof_map, objective)
        te_map = {
            "summary_meta": te_sum,
            "summary_meta_struct": te_struct,
            "full_meta": te_full,
            "rationale_text": te_text,
        }
        blend_pred = blend_probs(te_map, blend_methods, blend_weights).argmax(axis=1)
        preds["blend"][te] = blend_pred

        rows.append(
            {
                "fold": fold,
                "summary_spec": json.dumps(spec_sum),
                "structured_spec": json.dumps(spec_struct),
                "full_spec": json.dumps(spec_full),
                "text_spec": json.dumps(text_spec),
                "blend_methods": "|".join(blend_methods),
                "blend_weights": "|".join(map(str, blend_weights)),
                **{f"summary_inner_{k}": v for k, v in inner_sum_m.items()},
                **{f"structured_inner_{k}": v for k, v in inner_struct_m.items()},
                **{f"full_inner_{k}": v for k, v in inner_full_m.items()},
                **{f"blend_inner_{k}": v for k, v in blend_inner_m.items()},
                **{f"test_summary_{k}": v for k, v in metric(y_te, pred_sum).items()},
                **{f"test_structured_{k}": v for k, v in metric(y_te, pred_struct).items()},
                **{f"test_full_{k}": v for k, v in metric(y_te, pred_full).items()},
                **{f"test_text_{k}": v for k, v in metric(y_te, pred_text).items()},
                **{f"test_blend_{k}": v for k, v in metric(y_te, blend_pred).items()},
            }
        )

    return preds, rows


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "officialschema_rationale_agent_blend_20260519"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_sets = build_features(run_dir)
    fast_oof = pd.read_csv(run_dir / "officialschema_nested_cv_fast_20260512" / "officialschema_nested_cv_fast_oof_predictions.csv")
    df = add_meta_features(df, fast_oof)
    structured_df = build_structured_features(pd.read_csv(run_dir / "agent_votes.csv", low_memory=False))
    text_df = build_rationale_text(pd.read_csv(run_dir / "agent_votes.csv", low_memory=False))

    # summary + meta direct uses summary features plus fast meta columns.
    meta_cols = [c for c in df.columns if c.startswith("nested_lr_")]
    feature_sets2 = {
        "summary_plus_meta": unique_cols(feature_sets["summary"] + meta_cols),
        "full_plus_meta": unique_cols(feature_sets["full_agent_plus_metadata"] + meta_cols),
        "summary_plus_meta_struct": unique_cols(feature_sets["summary"] + meta_cols + [c for c in structured_df.columns if c != "noteId"]),
    }

    preds, fold_rows = nested_cv(
        df=df,
        structured_df=structured_df,
        text_df=text_df,
        feature_sets=feature_sets2,
        folds=args.folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        objective=args.objective,
        max_word_features=args.max_word_features,
        max_char_features=args.max_char_features,
    )

    y = df["true_label_3way"].to_numpy(dtype=int)
    out = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    rows = []
    for method, pred in preds.items():
        out[method] = pred
        rows.append({"method": method, "family": "blend_stack" if method == "blend" else "base", "n_features": 0, **metric(y, pred)})

    summary = pd.DataFrame(rows)
    for c in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful", "min_recall"]:
        summary[f"{c}_pct"] = pd.to_numeric(summary[c], errors="coerce") * 100.0
    summary["joint_pct"] = 100.0 * (
        2.0 * summary["accuracy"] * summary["balanced_accuracy"] / np.maximum(summary["accuracy"] + summary["balanced_accuracy"], 1e-12)
    )
    summary = summary.sort_values(["balanced_accuracy", "accuracy", "min_recall", "cross_error"], ascending=[False, False, False, True])

    best = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(out["true_label_text"], out[best].map(LABEL), margins=True)

    out.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
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
                "feature_sets": {k: len(v) for k, v in feature_sets2.items()},
                "best": summary.iloc[0].to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cols = [
        "method",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "joint_pct",
        "recall_not_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_helpful_pct",
        "min_recall_pct",
    ]
    print(summary[cols].to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
