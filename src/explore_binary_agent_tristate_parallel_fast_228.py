from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260513)
    return p.parse_args()


def clean_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean01(s: pd.Series) -> pd.Series:
    return clean_num(s).fillna(0).clip(0, 1)


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
        "mean_understanding": ("changes_reader_understanding", "mean"),
        "std_understanding": ("changes_reader_understanding", "std"),
        "helpful_reason_sum": ("helpful_reason_sum", "mean"),
        "not_helpful_reason_sum": ("not_helpful_reason_sum", "mean"),
        "reason_conflict_h_vote_rate": ("reason_conflict_h_vote", "mean"),
        "reason_conflict_nh_vote_rate": ("reason_conflict_nh_vote", "mean"),
    }
    for col in HELPFUL_REASONS + NOT_HELPFUL_REASONS:
        agg[f"{col}_rate"] = (col, "mean")
    df = votes.groupby("noteId", as_index=False).agg(**agg)
    for col in ["std_score", "std_confidence", "std_understanding"]:
        df[col] = df[col].fillna(0)
    df["share_h"] = df["vote_h"] / df["n_votes"].clip(lower=1)
    df["share_nh"] = df["vote_nh"] / df["n_votes"].clip(lower=1)
    df["h_minus_nh"] = df["share_h"] - df["share_nh"]
    df["h_nh_margin"] = df["h_minus_nh"].abs()
    df["vote_entropy_binary"] = -(
        df["share_h"].clip(1e-9, 1) * np.log(df["share_h"].clip(1e-9, 1))
        + df["share_nh"].clip(1e-9, 1) * np.log(df["share_nh"].clip(1e-9, 1))
    )
    df["near_tie_0125"] = (df["h_nh_margin"] <= 0.125).astype(float)
    df["near_tie_025"] = (df["h_nh_margin"] <= 0.25).astype(float)

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
            "near_tie_0125",
            "near_tie_025",
        ],
        "compact": [],
        "official": [f"{c}_rate" for c in HELPFUL_REASONS + NOT_HELPFUL_REASONS]
        + ["helpful_reason_sum", "not_helpful_reason_sum", "reason_conflict_h_vote_rate", "reason_conflict_nh_vote_rate"],
        "confidence": ["mean_confidence", "std_confidence"],
        "understanding": ["mean_understanding", "std_understanding"],
        "per_agent": [c for c in df.columns if c.startswith("agent_") and c.endswith("_is_h")],
    }
    groups["compact"] = unique(groups["vote"], groups["official"], groups["confidence"], groups["understanding"])
    groups["full"] = unique(groups["compact"], groups["per_agent"])
    groups["vote_official"] = unique(groups["vote"], groups["official"])
    return df.sort_values("noteId").reset_index(drop=True), groups


def unique(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for col in group:
            if col not in out:
                out.append(col)
    return out


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


def make_model(spec: dict[str, Any], seed: int) -> Pipeline:
    kind = spec["kind"]
    if kind == "lr":
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("sc", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=spec["c"],
                        class_weight=spec["weight"],
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "svc":
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("sc", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=spec["c"],
                        gamma=spec["gamma"],
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "rf":
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=spec["depth"],
                        min_samples_leaf=spec["leaf"],
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "extra":
        return Pipeline(
            [
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=450,
                        max_depth=spec["depth"],
                        min_samples_leaf=spec["leaf"],
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(kind)


def direct_oof_task(payload: tuple[str, dict[str, Any], pd.DataFrame, list[str], int, int]) -> tuple[str, np.ndarray, dict[str, Any]]:
    name, spec, df, features, folds, seed = payload
    y = df["true_label_3way"].to_numpy(int)
    x = df[features]
    pred = np.zeros(len(y), dtype=int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows = []
    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        m = make_model(spec, seed + fold)
        m.fit(x.iloc[tr], y[tr])
        pred[te] = m.predict(x.iloc[te])
        fold_rows.append({"fold": fold, **metric(y[te], pred[te])})
    return name, pred, {"spec": spec, "folds": fold_rows}


def disagreement_rule_task(payload: tuple[str, dict[str, Any], pd.DataFrame, int, int]) -> tuple[str, np.ndarray, dict[str, Any]]:
    name, spec, df, folds, seed = payload
    y = df["true_label_3way"].to_numpy(int)
    pred = np.zeros(len(y), dtype=int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows = []
    for fold, (_, te) in enumerate(outer.split(df, y), start=1):
        test = df.iloc[te]
        p = np.where(test["share_h"].to_numpy() >= spec["h_cut"], 2, 0)
        nmr = (
            (test["h_nh_margin"].to_numpy() <= spec["margin"])
            | (test["vote_entropy_binary"].to_numpy() >= spec["entropy"])
            | (test["mean_confidence"].fillna(100).to_numpy() <= spec["conf_low"])
            | (test["mean_understanding"].fillna(100).to_numpy() <= spec["understanding_low"])
        )
        p[nmr] = 1
        pred[te] = p
        fold_rows.append({"fold": fold, **metric(y[te], p)})
    return name, pred, {"spec": spec, "folds": fold_rows}


def p_class(pipe: Pipeline, x: pd.DataFrame, cls: int) -> np.ndarray:
    classes = list(pipe.named_steps["clf"].classes_)
    prob = pipe.predict_proba(x)
    return prob[:, classes.index(cls)]


def two_stage_task(payload: tuple[str, dict[str, Any], pd.DataFrame, list[str], int, int]) -> tuple[str, np.ndarray, dict[str, Any]]:
    name, spec, df, features, folds, seed = payload
    y = df["true_label_3way"].to_numpy(int)
    x = df[features]
    pred = np.zeros(len(y), dtype=int)
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows = []
    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        nmr_model = make_model({"kind": "lr", "c": spec["c_nmr"], "weight": spec["w_nmr"]}, seed + fold)
        h_model = make_model({"kind": "lr", "c": spec["c_h"], "weight": spec["w_h"]}, seed + 100 + fold)
        nmr_model.fit(x.iloc[tr], (y[tr] == 1).astype(int))
        resolved = np.isin(y[tr], [0, 2])
        h_model.fit(x.iloc[tr].iloc[resolved], (y[tr][resolved] == 2).astype(int))
        p_nmr = p_class(nmr_model, x.iloc[te], 1)
        p_h = p_class(h_model, x.iloc[te], 1)
        p = np.where(p_nmr >= spec["nmr_t"], 1, np.where(p_h >= spec["h_t"], 2, 0))
        pred[te] = p
        fold_rows.append({"fold": fold, **metric(y[te], p)})
    return name, pred, {"spec": spec, "folds": fold_rows}


def add_pct(summary: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "accuracy",
        "balanced_accuracy",
        "resolved_coverage",
        "recall_not_helpful",
        "recall_needs_more_ratings",
        "recall_helpful",
    ]:
        summary[f"{col}_pct"] = summary[col] * 100
    return summary


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "tristate_aggregator_parallel_fast_20260513"
    out_dir.mkdir(exist_ok=True)
    df, groups = load_feature_table(run_dir)
    y = df["true_label_3way"].to_numpy(int)

    tasks: list[tuple[str, tuple[Any, ...]]] = []

    # Direct 3-class models.
    for feat_name in ["vote", "compact", "full", "vote_official"]:
        features = groups[feat_name]
        for c in [0.03, 0.1, 0.3, 1.0, 3.0]:
            for weight in [None, "balanced"]:
                spec = {"kind": "lr", "c": c, "weight": weight}
                name = f"direct_lr_{feat_name}_C{c}_{weight or 'none'}"
                tasks.append(("direct", (name, spec, df, features, args.folds, args.seed + len(tasks))))
        for c in [0.3, 1.0, 3.0]:
            spec = {"kind": "svc", "c": c, "gamma": "scale"}
            name = f"direct_svc_{feat_name}_C{c}"
            tasks.append(("direct", (name, spec, df, features, args.folds, args.seed + len(tasks))))

    for feat_name in ["compact", "full"]:
        features = groups[feat_name]
        for depth in [2, 3, None]:
            for leaf in [3, 6]:
                for kind in ["rf", "extra"]:
                    spec = {"kind": kind, "depth": depth, "leaf": leaf}
                    name = f"direct_{kind}_{feat_name}_d{depth}_l{leaf}"
                    tasks.append(("direct", (name, spec, df, features, args.folds, args.seed + len(tasks))))

    # Simple disagreement/NMR rules over binary votes.
    for margin in [0.0, 0.125, 0.25, 0.375]:
        for entropy in [0.55, 0.62, 0.68]:
            for h_cut in [0.5, 0.56, 0.62]:
                spec = {"margin": margin, "entropy": entropy, "h_cut": h_cut, "conf_low": -1, "understanding_low": -1}
                name = f"rule_m{margin}_e{entropy}_h{h_cut}"
                tasks.append(("rule", (name, spec, df, args.folds, args.seed + len(tasks))))

    # Two-stage fast grid: NMR detector + H/NH detector.
    for feat_name in ["vote", "compact", "vote_official", "full"]:
        features = groups[feat_name]
        for c_nmr in [0.03, 0.1, 0.3, 1.0]:
            for c_h in [0.03, 0.1, 0.3, 1.0]:
                for nmr_t in [0.25, 0.33, 0.40, 0.48]:
                    for h_t in [0.40, 0.50, 0.60]:
                        spec = {
                            "c_nmr": c_nmr,
                            "w_nmr": "balanced",
                            "c_h": c_h,
                            "w_h": "balanced",
                            "nmr_t": nmr_t,
                            "h_t": h_t,
                        }
                        name = f"twostage_{feat_name}_cn{c_nmr}_ch{c_h}_tn{nmr_t}_th{h_t}"
                        tasks.append(("two_stage", (name, spec, df, features, args.folds, args.seed + len(tasks))))

    print(f"running {len(tasks)} candidates with jobs={args.jobs}", flush=True)
    rows = []
    pred_table = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    details: dict[str, Any] = {}

    raw = np.where(df["share_h"].to_numpy() >= 0.5, 2, 0)
    rows.append({"method": "raw_binary_majority_no_nmr", **metric(y, raw)})
    pred_table["raw_binary_majority_no_nmr"] = raw

    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = []
        for kind, payload in tasks:
            if kind == "direct":
                futures.append(pool.submit(direct_oof_task, payload))
            elif kind == "rule":
                futures.append(pool.submit(disagreement_rule_task, payload))
            elif kind == "two_stage":
                futures.append(pool.submit(two_stage_task, payload))
            else:
                raise ValueError(kind)
        for future in as_completed(futures):
            name, pred, detail = future.result()
            done += 1
            rows.append({"method": name, **metric(y, pred)})
            details[name] = detail
            if done % 50 == 0 or done == len(futures):
                current = pd.DataFrame(rows)
                current = add_pct(current).sort_values(["balanced_accuracy", "accuracy"], ascending=[False, False])
                print(
                    f"[{done}/{len(futures)}] best={current.iloc[0]['method']} "
                    f"bal={current.iloc[0]['balanced_accuracy_pct']:.2f} "
                    f"acc={current.iloc[0]['accuracy_pct']:.2f}",
                    flush=True,
                )

    summary = add_pct(pd.DataFrame(rows)).sort_values(["balanced_accuracy", "accuracy"], ascending=[False, False])
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    top_methods = summary.head(20)["method"].tolist()
    pred_payload = {"raw_binary_majority_no_nmr": raw.tolist()}
    # Recompute and save predictions only for top methods to keep memory and files tidy.
    top_set = set(top_methods)
    for kind, payload in tasks:
        name = payload[0]
        if name not in top_set:
            continue
        if kind == "direct":
            _, pred, _ = direct_oof_task(payload)
        elif kind == "rule":
            _, pred, _ = disagreement_rule_task(payload)
        else:
            _, pred, _ = two_stage_task(payload)
        pred_table[name] = pred
        cm = pd.DataFrame(
            confusion_matrix(y, pred, labels=[2, 1, 0]),
            index=["true_HELPFUL", "true_NMR", "true_NOT_HELPFUL"],
            columns=["pred_HELPFUL", "pred_NMR", "pred_NOT_HELPFUL"],
        )
        cm.to_csv(out_dir / f"{name}_confusion.csv", encoding="utf-8-sig")
        pred_payload[name] = pred.tolist()
    pred_table.to_csv(out_dir / "top_method_predictions.csv", index=False, encoding="utf-8-sig")
    with (out_dir / "candidate_details.json").open("w", encoding="utf-8") as f:
        json.dump({k: details[k] for k in top_methods if k in details}, f, ensure_ascii=False, indent=2)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "n_candidates": len(tasks),
                "jobs": args.jobs,
                "note": "Parallel fast exploration. Every candidate is evaluated by outer-fold predictions; no API calls.",
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
    print("\n=== Top 20 ===")
    print(summary[show].head(20).to_string(index=False))
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
