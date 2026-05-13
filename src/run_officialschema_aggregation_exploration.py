from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}
RAW = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}
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
    x = pd.to_numeric(s, errors="coerce")
    return x.where(x >= 0, np.nan)


def entropy(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    arr = df[cols].fillna(0).clip(1e-9, 1).to_numpy(float)
    return pd.Series(-(arr * np.log(arr)).sum(axis=1), index=df.index)


def load_votes(run_dir: Path) -> pd.DataFrame:
    votes = pd.read_csv(run_dir / "agent_votes.csv", low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["agent_id"] = votes["agent_id"].astype(str)
    votes["raw_label"] = votes["parsed_rating"].astype(str)
    votes["raw_label_id"] = votes["raw_label"].map(RAW)
    votes = votes[votes["raw_label_id"].isin([0, 1, 2])].copy()
    for c in ["predicted_rating_score", "confidence", "changes_reader_understanding", "agree", "disagree", *HELPFUL_REASONS, *NOT_HELPFUL_REASONS]:
        if c not in votes.columns:
            votes[c] = np.nan
        votes[c] = clean01(votes[c])
    votes["true_label_3way"] = pd.to_numeric(votes["true_label_3way"], errors="coerce").astype(int)
    votes["is_h"] = (votes["raw_label_id"] == 2).astype(float)
    votes["is_sh"] = (votes["raw_label_id"] == 1).astype(float)
    votes["is_nh"] = (votes["raw_label_id"] == 0).astype(float)
    votes["helpful_reason_sum"] = votes[HELPFUL_REASONS].sum(axis=1, skipna=True)
    votes["not_helpful_reason_sum"] = votes[NOT_HELPFUL_REASONS].sum(axis=1, skipna=True)
    votes["helpful_any"] = (votes["helpful_reason_sum"] > 0).astype(float)
    votes["not_helpful_any"] = (votes["not_helpful_reason_sum"] > 0).astype(float)
    votes["h_vote_has_nh_reason"] = ((votes["raw_label_id"] == 2) & (votes["not_helpful_reason_sum"] > 0)).astype(float)
    votes["nh_vote_has_h_reason"] = ((votes["raw_label_id"] == 0) & (votes["helpful_reason_sum"] > 0)).astype(float)
    return votes


def build_features(votes: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "tweetId": ("tweetId", "first"),
        "true_label_3way": ("true_label_3way", "first"),
        "true_label_text": ("true_label_text", "first"),
        "n_votes": ("agent_id", "size"),
        "vote_h": ("is_h", "sum"),
        "vote_sh": ("is_sh", "sum"),
        "vote_nh": ("is_nh", "sum"),
        "mean_score": ("predicted_rating_score", "mean"),
        "std_score": ("predicted_rating_score", "std"),
        "mean_conf": ("confidence", "mean"),
        "std_conf": ("confidence", "std"),
        "mean_understanding": ("changes_reader_understanding", "mean"),
        "std_understanding": ("changes_reader_understanding", "std"),
        "agree_rate": ("agree", "mean"),
        "disagree_rate": ("disagree", "mean"),
        "helpful_reason_sum": ("helpful_reason_sum", "mean"),
        "not_helpful_reason_sum": ("not_helpful_reason_sum", "mean"),
        "helpful_any_rate": ("helpful_any", "mean"),
        "not_helpful_any_rate": ("not_helpful_any", "mean"),
        "h_vote_has_nh_reason_rate": ("h_vote_has_nh_reason", "mean"),
        "nh_vote_has_h_reason_rate": ("nh_vote_has_h_reason", "mean"),
    }
    for c in HELPFUL_REASONS + NOT_HELPFUL_REASONS:
        agg_spec[f"{c}_rate"] = (c, "mean")
    base = votes.groupby("noteId", as_index=False).agg(**agg_spec)
    base["true_label_3way"] = pd.to_numeric(base["true_label_3way"], errors="coerce").astype(int)
    for c in ["std_score", "std_conf", "std_understanding"]:
        base[c] = base[c].fillna(0)
    for label, count in [("h", "vote_h"), ("sh", "vote_sh"), ("nh", "vote_nh")]:
        base[f"share_{label}"] = base[count] / base["n_votes"].clip(lower=1)
    base["vote_entropy"] = entropy(base, ["share_nh", "share_sh", "share_h"])
    base["h_minus_nh"] = base["share_h"] - base["share_nh"]
    base["h_nh_margin"] = base["h_minus_nh"].abs()
    base["resolved_share"] = base["share_h"] + base["share_nh"]
    base["helpful_support"] = (
        base["share_h"]
        + 0.18 * base["helpfulImportantContext_rate"]
        + 0.18 * base["helpfulAddressesClaim_rate"]
        + 0.12 * base["helpfulGoodSources_rate"]
        + 0.08 * base["helpfulUnbiasedLanguage_rate"]
        + 0.004 * base["mean_understanding"].fillna(50)
    )
    base["not_helpful_support"] = (
        base["share_nh"]
        + 0.15 * base["notHelpfulMissingKeyPoints_rate"]
        + 0.15 * base["notHelpfulSourcesMissingOrUnreliable_rate"]
        + 0.15 * base["notHelpfulNoteNotNeeded_rate"]
        + 0.10 * base["notHelpfulIrrelevantSources_rate"]
        + 0.004 * (100 - base["mean_understanding"].fillna(50))
    )
    base["support_diff"] = base["helpful_support"] - base["not_helpful_support"]
    base["support_margin"] = base["support_diff"].abs()

    # Conditional means for each raw label group.
    for raw_id, prefix in [(2, "h"), (1, "sh"), (0, "nh")]:
        part = (
            votes[votes["raw_label_id"] == raw_id]
            .groupby("noteId")[
                [
                    "confidence",
                    "changes_reader_understanding",
                    "helpful_reason_sum",
                    "not_helpful_reason_sum",
                    "helpfulImportantContext",
                    "helpfulAddressesClaim",
                    "helpfulGoodSources",
                    "notHelpfulMissingKeyPoints",
                    "notHelpfulSourcesMissingOrUnreliable",
                    "notHelpfulNoteNotNeeded",
                ]
            ]
            .mean()
            .add_prefix(f"{prefix}_vote_")
            .reset_index()
        )
        base = base.merge(part, on="noteId", how="left")

    # Per-agent label one-hots; useful only for regularized models.
    lab = votes.pivot_table(index="noteId", columns="agent_id", values="raw_label_id", aggfunc="first")
    lab = lab.add_prefix("agent_label__").reset_index()
    base = base.merge(lab, on="noteId", how="left")
    for c in [x for x in base.columns if x.startswith("agent_label__")]:
        for raw_id, name in [(0, "nh"), (1, "sh"), (2, "h")]:
            base[f"{c}__is_{name}"] = (base[c] == raw_id).astype(float)
    base = base.drop(columns=[c for c in base.columns if c.startswith("agent_label__") and "__is_" not in c])
    return base.sort_values("noteId").reset_index(drop=True)


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
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


def obj(m: dict, mode: str) -> tuple:
    if mode == "balanced":
        return (m["balanced_accuracy"], m["accuracy"], -m["cross_error"])
    if mode == "cross_safe":
        return (-m["cross_error"], m["balanced_accuracy"], m["accuracy"])
    if mode == "helpful_nothelpful":
        return ((m["recall_helpful"] + m["recall_not_helpful"]) / 2, -m["cross_error"], m["balanced_accuracy"])
    raise ValueError(mode)


def lr_model(c: float, weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight=weight, solver="lbfgs", max_iter=5000, random_state=42)),
        ]
    )


def align_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = p[:, i]
    return out


def decision_from_prob(prob: np.ndarray, rule: str, a: float, b: float) -> np.ndarray:
    if rule == "argmax":
        return prob.argmax(axis=1)
    h, nmr, nh = prob[:, 2], prob[:, 1], prob[:, 0]
    if rule == "nmr_gate":
        return np.where(nmr >= a, 1, np.where(h >= nh, 2, 0))
    if rule == "margin_to_nmr":
        pred = prob.argmax(axis=1)
        near = np.abs(h - nh) < a
        pred[near & (nmr >= b)] = 1
        return pred
    if rule == "resolved_gap":
        resolved = np.maximum(h, nh)
        return np.where((resolved - nmr) >= a, np.where(h >= nh, 2, 0), 1)
    raise ValueError(rule)


def nested_lr_decision(df: pd.DataFrame, features: list[str], folds: int, inner_folds: int, seed: int, mode: str):
    y = df["true_label_3way"].to_numpy(int)
    X = df[features]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    fold_rows = []
    c_grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    weights = [None, "balanced"]
    rule_grid = [("argmax", 0.0, 0.0)]
    rule_grid += [("nmr_gate", x, 0.0) for x in np.arange(0.28, 0.51, 0.04)]
    rule_grid += [("margin_to_nmr", m, t) for m in [0.04, 0.08, 0.12, 0.16, 0.20] for t in [0.20, 0.25, 0.30, 0.35]]
    rule_grid += [("resolved_gap", g, 0.0) for g in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]]
    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        Xtr = X.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        for c in c_grid:
            for w in weights:
                oof = np.zeros((len(ytr), 3), dtype=float)
                for a_tr, a_va in inner.split(Xtr, ytr):
                    model = lr_model(c, w)
                    model.fit(Xtr.iloc[a_tr], ytr[a_tr])
                    oof[a_va] = align_prob(model, Xtr.iloc[a_va])
                for rule, a, b in rule_grid:
                    p = decision_from_prob(oof, rule, float(a), float(b))
                    m = metric(ytr, p)
                    key = obj(m, mode)
                    if best is None or key > best[0]:
                        best = (key, c, w, rule, float(a), float(b), m)
        _, c, w, rule, a, b, im = best
        model = lr_model(c, w)
        model.fit(X.iloc[tr], y[tr])
        pred[te] = decision_from_prob(align_prob(model, X.iloc[te]), rule, a, b)
        fm = metric(y[te], pred[te])
        fold_rows.append({"fold": fold, "c": c, "weight": w or "none", "rule": rule, "a": a, "b": b, **{f"inner_{k}": v for k, v in im.items()}, **{f"test_{k}": v for k, v in fm.items()}})
    return pred, fold_rows


def rule_scores(df: pd.DataFrame, h_extra: float, nh_extra: float) -> tuple[np.ndarray, np.ndarray]:
    h = df["share_h"].to_numpy() + h_extra * (
        0.35 * df["helpfulImportantContext_rate"].fillna(0).to_numpy()
        + 0.35 * df["helpfulAddressesClaim_rate"].fillna(0).to_numpy()
        + 0.20 * df["helpfulGoodSources_rate"].fillna(0).to_numpy()
        + 0.001 * df["mean_understanding"].fillna(50).to_numpy()
    )
    nh = df["share_nh"].to_numpy() + nh_extra * (
        0.30 * df["notHelpfulMissingKeyPoints_rate"].fillna(0).to_numpy()
        + 0.30 * df["notHelpfulSourcesMissingOrUnreliable_rate"].fillna(0).to_numpy()
        + 0.25 * df["notHelpfulNoteNotNeeded_rate"].fillna(0).to_numpy()
        + 0.001 * (100 - df["mean_understanding"].fillna(50).to_numpy())
    )
    return h, nh


def tune_rule(train: pd.DataFrame, y: np.ndarray, mode: str):
    best = None
    for h_extra in [0.0, 0.2, 0.4, 0.6]:
        for nh_extra in [0.0, 0.2, 0.4, 0.6]:
            h, nh = rule_scores(train, h_extra, nh_extra)
            sh = train["share_sh"].to_numpy()
            ent = train["vote_entropy"].to_numpy()
            for margin in np.arange(0.0, 0.36, 0.04):
                for sh_t in np.arange(0.20, 0.66, 0.08):
                    pred = np.where(h >= nh, 2, 0)
                    pred[(np.abs(h - nh) < margin) | ((sh >= sh_t) & (ent > 0.80))] = 1
                    m = metric(y, pred)
                    key = obj(m, mode)
                    if best is None or key > best[0]:
                        best = (key, h_extra, nh_extra, float(margin), float(sh_t), m)
    return best[1:]


def nested_rule(df: pd.DataFrame, folds: int, seed: int, mode: str):
    y = df["true_label_3way"].to_numpy(int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(df, y), start=1):
        h_extra, nh_extra, margin, sh_t, im = tune_rule(df.iloc[tr], y[tr], mode)
        h, nh = rule_scores(df.iloc[te], h_extra, nh_extra)
        sh = df.iloc[te]["share_sh"].to_numpy()
        ent = df.iloc[te]["vote_entropy"].to_numpy()
        p = np.where(h >= nh, 2, 0)
        p[(np.abs(h - nh) < margin) | ((sh >= sh_t) & (ent > 0.80))] = 1
        pred[te] = p
        rows.append({"fold": fold, "h_extra": h_extra, "nh_extra": nh_extra, "margin": margin, "sh_t": sh_t, **{f"inner_{k}": v for k, v in im.items()}, **{f"test_{k}": v for k, v in metric(y[te], p).items()}})
    return pred, rows


def estimate_agent_tables(train_votes: pd.DataFrame, alpha: float) -> dict[tuple[str, int], np.ndarray]:
    tables: dict[tuple[str, int], np.ndarray] = {}
    global_counts = np.bincount(train_votes["true_label_3way"].to_numpy(int), minlength=3).astype(float) + alpha
    global_prob = global_counts / global_counts.sum()
    for agent in train_votes["agent_id"].unique():
        av = train_votes[train_votes["agent_id"] == agent]
        for raw_id in [0, 1, 2]:
            g = av[av["raw_label_id"] == raw_id]
            if len(g) == 0:
                tables[(agent, raw_id)] = global_prob
            else:
                counts = np.bincount(g["true_label_3way"].to_numpy(int), minlength=3).astype(float) + alpha
                tables[(agent, raw_id)] = counts / counts.sum()
    return tables


def reliability_predict(votes_subset: pd.DataFrame, tables: dict[tuple[str, int], np.ndarray], alpha: float, conf_power: float) -> pd.DataFrame:
    global_prob = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    rows = []
    for note_id, g in votes_subset.groupby("noteId", sort=False):
        score = np.zeros(3, dtype=float)
        for _, r in g.iterrows():
            vec = tables.get((str(r["agent_id"]), int(r["raw_label_id"])), global_prob)
            conf = r.get("confidence", np.nan)
            if pd.isna(conf) or conf < 0:
                w = 1.0
            else:
                w = max(float(conf) / 100.0, 0.05) ** conf_power
            score += w * vec
        prob = score / max(score.sum(), 1e-9)
        rows.append({"noteId": note_id, "p_nh": prob[0], "p_nmr": prob[1], "p_h": prob[2]})
    return pd.DataFrame(rows)


def nested_reliability(votes: pd.DataFrame, df: pd.DataFrame, folds: int, inner_folds: int, seed: int, mode: str):
    y = df["true_label_3way"].to_numpy(int)
    note_ids = df["noteId"].astype(str).to_numpy()
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    rows = []
    alphas = [0.25, 0.5, 1.0, 2.0, 4.0]
    conf_powers = [0.0, 0.5, 1.0]
    rule_grid = [("argmax", 0.0, 0.0)] + [("margin_to_nmr", m, t) for m in [0.05, 0.1, 0.15, 0.2] for t in [0.2, 0.25, 0.3, 0.35]]
    for fold, (tr, te) in enumerate(outer.split(df, y), start=1):
        train_ids = set(note_ids[tr])
        test_ids = set(note_ids[te])
        train_votes = votes[votes["noteId"].isin(train_ids)]
        inner_df = df.iloc[tr].reset_index(drop=True)
        inner_y = y[tr]
        inner_ids = inner_df["noteId"].astype(str).to_numpy()
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best = None
        for alpha in alphas:
            for cp in conf_powers:
                oof = np.zeros((len(inner_y), 3), dtype=float)
                for a_tr, a_va in inner.split(inner_df, inner_y):
                    a_train_ids = set(inner_ids[a_tr])
                    a_val_ids = set(inner_ids[a_va])
                    tables = estimate_agent_tables(train_votes[train_votes["noteId"].isin(a_train_ids)], alpha)
                    pp = reliability_predict(train_votes[train_votes["noteId"].isin(a_val_ids)], tables, alpha, cp).set_index("noteId")
                    for pos, nid in zip(a_va, inner_ids[a_va]):
                        oof[pos] = pp.loc[nid, ["p_nh", "p_nmr", "p_h"]].to_numpy(float)
                for rule, a, b in rule_grid:
                    p = decision_from_prob(oof, rule, a, b)
                    m = metric(inner_y, p)
                    key = obj(m, mode)
                    if best is None or key > best[0]:
                        best = (key, alpha, cp, rule, a, b, m)
        _, alpha, cp, rule, a, b, im = best
        tables = estimate_agent_tables(train_votes, alpha)
        pp = reliability_predict(votes[votes["noteId"].isin(test_ids)], tables, alpha, cp).set_index("noteId")
        prob = np.vstack([pp.loc[nid, ["p_nh", "p_nmr", "p_h"]].to_numpy(float) for nid in note_ids[te]])
        pred[te] = decision_from_prob(prob, rule, a, b)
        rows.append({"fold": fold, "alpha": alpha, "conf_power": cp, "rule": rule, "a": a, "b": b, **{f"inner_{k}": v for k, v in im.items()}, **{f"test_{k}": v for k, v in metric(y[te], pred[te]).items()}})
    return pred, rows


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "aggregation_exploration_20260513"
    out_dir.mkdir(parents=True, exist_ok=True)
    votes = load_votes(run_dir)
    df = build_features(votes)
    non_features = {"noteId", "tweetId", "true_label_3way", "true_label_text"}
    compact = [
        c for c in df.columns
        if c not in non_features and not c.startswith("agent_label__")
    ]
    full = [c for c in df.columns if c not in non_features]
    y = df["true_label_3way"].to_numpy(int)
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    rows = []
    fold_tables = []

    methods = []
    methods.append(("lr_compact_balanced", lambda: nested_lr_decision(df, compact, args.folds, args.inner_folds, args.seed, "balanced")))
    methods.append(("lr_compact_cross_safe", lambda: nested_lr_decision(df, compact, args.folds, args.inner_folds, args.seed + 10, "cross_safe")))
    methods.append(("lr_full_hnh_focus", lambda: nested_lr_decision(df, full, args.folds, args.inner_folds, args.seed + 20, "helpful_nothelpful")))
    methods.append(("rule_grid_balanced", lambda: nested_rule(df, args.folds, args.seed + 30, "balanced")))
    methods.append(("rule_grid_cross_safe", lambda: nested_rule(df, args.folds, args.seed + 40, "cross_safe")))
    methods.append(("agent_reliability_balanced", lambda: nested_reliability(votes, df, args.folds, args.inner_folds, args.seed + 50, "balanced")))
    methods.append(("agent_reliability_cross_safe", lambda: nested_reliability(votes, df, args.folds, args.inner_folds, args.seed + 60, "cross_safe")))

    # Include simple reference methods.
    raw_vote = df[["vote_nh", "vote_sh", "vote_h"]].to_numpy(float).argmax(axis=1)
    raw_score = np.where(df["mean_score"].to_numpy() >= 2 / 3, 2, np.where(df["mean_score"].to_numpy() <= 1 / 3, 0, 1))
    for name, pred in {"raw_vote": raw_vote, "raw_score_033_067": raw_score}.items():
        preds[name] = pred
        rows.append({"method": name, "family": "reference", **metric(y, pred)})

    for idx, (name, fn) in enumerate(methods, start=1):
        print(f"[{idx}/{len(methods)}] running {name}", flush=True)
        pred, fold_rows = fn()
        preds[name] = pred
        rows.append({"method": name, "family": "exploration", **metric(y, pred)})
        fold_tables.append(pd.DataFrame(fold_rows).assign(method=name))
        print(f"[{idx}/{len(methods)}] done {name}: {metric(y, pred)}", flush=True)

    summary = pd.DataFrame(rows)
    for c in ["accuracy", "balanced_accuracy", "recall_not_helpful", "recall_needs_more_ratings", "recall_helpful"]:
        summary[f"{c}_pct"] = summary[c] * 100
    summary = summary.sort_values(["balanced_accuracy", "accuracy", "cross_error"], ascending=[False, False, True])
    best = str(summary.iloc[0]["method"])
    confusion = pd.crosstab(df["true_label_text"], pd.Series(preds[best]).map(LABEL), margins=True)

    df.to_csv(out_dir / "feature_table.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    if fold_tables:
        pd.concat(fold_tables, ignore_index=True).to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_notes": int(len(df)),
        "best_method": best,
        "best": summary.iloc[0].to_dict(),
        "methods": [m[0] for m in methods],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "method", "accuracy_pct", "balanced_accuracy_pct",
        "recall_not_helpful_pct", "recall_needs_more_ratings_pct", "recall_helpful_pct",
        "h_to_nh", "nh_to_h", "cross_error",
    ]
    print(summary[cols].to_string(index=False))
    print("\nBest confusion matrix:")
    print(confusion.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
