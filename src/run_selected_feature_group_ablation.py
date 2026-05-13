from __future__ import annotations

import argparse
import json
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
RAW_LABEL = {"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2}

# Official Community Notes helpful reason checkboxes, excluding the non-diagnostic
# "Other" bucket.
HELPFUL_REASONS = [
    "helpfulClear",
    "helpfulGoodSources",
    "helpfulAddressesClaim",
    "helpfulImportantContext",
    "helpfulUnbiasedLanguage",
]

# Official Community Notes not-helpful reason checkboxes. Deprecated fields are
# included if present in a run, but missing columns are treated as zero.
NOT_HELPFUL_REASONS = [
    "notHelpfulIncorrect",
    "notHelpfulSourcesMissingOrUnreliable",
    "NotHelpfulOpinionSpeculationOrBias",
    "notHelpfulMissingKeyPoints",
    "notHelpfulOutdated",
    "notHelpfulHardToUnderstand",
    "notHelpfulArgumentativeOrBiased",
    "notHelpfulOffTopic",
    "notHelpfulSpamHarassmentOrAbuse",
    "notHelpfulIrrelevantSources",
    "notHelpfulOpinionSpeculation",
    "notHelpfulNoteNotNeeded",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/data6/wenchangxi/community_note/analysis/"
            "llm_16agent_rawrating_balanced_1to1to1_promptv3_20260513"
        ),
    )
    p.add_argument("--out-name", default="selected_feature_group_ablation_20260513")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260513)
    return p.parse_args()


def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_binary(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(0).clip(0, 1)


def entropy_3way(df: pd.DataFrame) -> pd.Series:
    arr = df[["share_nh", "share_sh", "share_h"]].fillna(0).clip(1e-9, 1).to_numpy(float)
    return pd.Series(-(arr * np.log(arr)).sum(axis=1), index=df.index)


def load_votes(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "agent_votes.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    votes = pd.read_csv(path, low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["agent_id"] = votes["agent_id"].astype(str)
    label_col = "parsed_rating" if "parsed_rating" in votes.columns else "raw_label"
    votes["raw_label"] = votes[label_col].astype(str)
    votes["raw_label_id"] = votes["raw_label"].map(RAW_LABEL)
    votes = votes[votes["raw_label_id"].isin([0, 1, 2])].copy()

    if "predicted_rating_score" not in votes.columns:
        votes["predicted_rating_score"] = votes["raw_label_id"] / 2.0
    votes["predicted_rating_score"] = clean_numeric(votes["predicted_rating_score"]).fillna(
        votes["raw_label_id"] / 2.0
    )

    for col in ["confidence", "changes_reader_understanding"]:
        if col not in votes.columns:
            votes[col] = np.nan
        votes[col] = clean_numeric(votes[col])

    for col in HELPFUL_REASONS + NOT_HELPFUL_REASONS:
        if col not in votes.columns:
            votes[col] = 0
        votes[col] = clean_binary(votes[col])

    votes["true_label_3way"] = clean_numeric(votes["true_label_3way"]).astype(int)
    if "true_label_text" not in votes.columns:
        votes["true_label_text"] = votes["true_label_3way"].map(LABEL)

    votes["is_h"] = (votes["raw_label_id"] == 2).astype(float)
    votes["is_sh"] = (votes["raw_label_id"] == 1).astype(float)
    votes["is_nh"] = (votes["raw_label_id"] == 0).astype(float)
    votes["helpful_reason_count"] = votes[HELPFUL_REASONS].sum(axis=1)
    votes["not_helpful_reason_count"] = votes[NOT_HELPFUL_REASONS].sum(axis=1)
    return votes


def build_note_features(votes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    agg = {
        "tweetId": ("tweetId", "first") if "tweetId" in votes.columns else ("noteId", "first"),
        "true_label_3way": ("true_label_3way", "first"),
        "true_label_text": ("true_label_text", "first"),
        "n_votes": ("agent_id", "size"),
        "vote_h": ("is_h", "sum"),
        "vote_sh": ("is_sh", "sum"),
        "vote_nh": ("is_nh", "sum"),
        "mean_score": ("predicted_rating_score", "mean"),
        "std_score": ("predicted_rating_score", "std"),
        "mean_confidence": ("confidence", "mean"),
        "std_confidence": ("confidence", "std"),
        "mean_understanding": ("changes_reader_understanding", "mean"),
        "std_understanding": ("changes_reader_understanding", "std"),
        "helpful_reason_count_mean": ("helpful_reason_count", "mean"),
        "not_helpful_reason_count_mean": ("not_helpful_reason_count", "mean"),
    }
    for col in HELPFUL_REASONS:
        agg[f"{col}_rate"] = (col, "mean")
    for col in NOT_HELPFUL_REASONS:
        agg[f"{col}_rate"] = (col, "mean")

    df = votes.groupby("noteId", as_index=False).agg(**agg)
    df["true_label_3way"] = clean_numeric(df["true_label_3way"]).astype(int)
    for col in ["std_score", "std_confidence", "std_understanding"]:
        df[col] = df[col].fillna(0)

    df["share_h"] = df["vote_h"] / df["n_votes"].clip(lower=1)
    df["share_sh"] = df["vote_sh"] / df["n_votes"].clip(lower=1)
    df["share_nh"] = df["vote_nh"] / df["n_votes"].clip(lower=1)
    df["vote_entropy"] = entropy_3way(df)
    df["h_minus_nh"] = df["share_h"] - df["share_nh"]
    df["h_nh_margin"] = df["h_minus_nh"].abs()
    df["resolved_vote_share"] = df["share_h"] + df["share_nh"]

    # Per-agent raw-label features: no answer-quality scores, only which raw
    # rating each persona gave. This is separated as its own ablation group.
    per_agent = votes.pivot_table(
        index="noteId", columns="agent_id", values="raw_label_id", aggfunc="first"
    )
    per_agent.columns = [f"agent_{c}_raw_label" for c in per_agent.columns]
    per_agent = per_agent.reset_index()
    df = df.merge(per_agent, on="noteId", how="left")
    per_agent_features: list[str] = []
    for col in [c for c in df.columns if c.startswith("agent_") and c.endswith("_raw_label")]:
        for raw_id, suffix in [(0, "nh"), (1, "sh"), (2, "h")]:
            out = f"{col}_is_{suffix}"
            df[out] = (df[col] == raw_id).astype(float)
            per_agent_features.append(out)
    df = df.drop(columns=[c for c in df.columns if c.startswith("agent_") and c.endswith("_raw_label")])

    feature_groups = {
        "vote": [
            "n_votes",
            "vote_h",
            "vote_sh",
            "vote_nh",
            "mean_score",
            "std_score",
            "share_h",
            "share_sh",
            "share_nh",
            "vote_entropy",
            "h_minus_nh",
            "h_nh_margin",
            "resolved_vote_share",
        ],
        "official_helpful_reasons": [f"{c}_rate" for c in HELPFUL_REASONS]
        + ["helpful_reason_count_mean"],
        "official_not_helpful_reasons": [f"{c}_rate" for c in NOT_HELPFUL_REASONS]
        + ["not_helpful_reason_count_mean"],
        "confidence": ["mean_confidence", "std_confidence"],
        "understanding": ["mean_understanding", "std_understanding"],
        "per_agent": per_agent_features,
    }
    return df.sort_values("noteId").reset_index(drop=True), feature_groups


def unique_features(groups: list[list[str]]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for col in group:
            if col not in out:
                out.append(col)
    return out


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "accuracy": float((y == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "h_to_nh": int(((y == 2) & (pred == 0)).sum()),
        "nh_to_h": int(((y == 0) & (pred == 2)).sum()),
    }
    out["cross_error"] = int(out["h_to_nh"]) + int(out["nh_to_h"])
    for label_id, label_name in LABEL.items():
        mask = y == label_id
        out[f"recall_{label_name.lower()}"] = float((pred[mask] == label_id).mean())
        out[f"n_{label_name.lower()}"] = int(mask.sum())
    return out


def objective(m: dict[str, float | int]) -> tuple[float, float, int]:
    return (
        float(m["balanced_accuracy"]),
        float(m["accuracy"]),
        -int(m["cross_error"]),
    )


def lr_pipeline(c: float, weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    class_weight=weight,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def align_proba(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    prob = model.predict_proba(x)
    out = np.zeros((len(x), 3), dtype=float)
    for i, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = prob[:, i]
    return out


def decide(prob: np.ndarray, rule: str, a: float, b: float) -> np.ndarray:
    if rule == "argmax":
        return prob.argmax(axis=1)
    nh, nmr, h = prob[:, 0], prob[:, 1], prob[:, 2]
    if rule == "nmr_gate":
        return np.where(nmr >= a, 1, np.where(h >= nh, 2, 0))
    if rule == "margin_to_nmr":
        pred = np.where(h >= nh, 2, 0)
        pred[(np.abs(h - nh) < a) & (nmr >= b)] = 1
        return pred
    if rule == "resolved_gap":
        pred = np.where(h >= nh, 2, 0)
        pred[(np.maximum(h, nh) - nmr) < a] = 1
        return pred
    raise ValueError(rule)


def nested_cv_lr(
    df: pd.DataFrame,
    features: list[str],
    folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["true_label_3way"].to_numpy(int)
    x = df[features]
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    rows: list[dict[str, float | int | str]] = []
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    weight_grid: list[str | None] = [None, "balanced"]
    rules: list[tuple[str, float, float]] = [("argmax", 0.0, 0.0)]
    rules += [("nmr_gate", float(t), 0.0) for t in np.arange(0.28, 0.57, 0.04)]
    rules += [
        ("margin_to_nmr", float(m), float(t))
        for m in [0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
        for t in [0.20, 0.25, 0.30, 0.35, 0.40]
    ]
    rules += [("resolved_gap", float(g), 0.0) for g in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]]

    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        x_train = x.iloc[tr].reset_index(drop=True)
        y_train = y[tr]
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        best: tuple[tuple[float, float, int], float, str | None, str, float, float, dict] | None = None
        for c in c_grid:
            for weight in weight_grid:
                oof = np.zeros((len(y_train), 3), dtype=float)
                for inner_tr, inner_va in inner.split(x_train, y_train):
                    model = lr_pipeline(c, weight)
                    model.fit(x_train.iloc[inner_tr], y_train[inner_tr])
                    oof[inner_va] = align_proba(model, x_train.iloc[inner_va])
                for rule, a, b in rules:
                    inner_pred = decide(oof, rule, a, b)
                    inner_metric = metric(y_train, inner_pred)
                    key = objective(inner_metric)
                    if best is None or key > best[0]:
                        best = (key, c, weight, rule, a, b, inner_metric)

        assert best is not None
        _, c, weight, rule, a, b, inner_metric = best
        model = lr_pipeline(c, weight)
        model.fit(x.iloc[tr], y[tr])
        fold_pred = decide(align_proba(model, x.iloc[te]), rule, a, b)
        pred[te] = fold_pred
        test_metric = metric(y[te], fold_pred)
        rows.append(
            {
                "fold": fold,
                "c": c,
                "class_weight": weight or "none",
                "decision_rule": rule,
                "a": a,
                "b": b,
                **{f"inner_{k}": v for k, v in inner_metric.items()},
                **{f"test_{k}": v for k, v in test_metric.items()},
            }
        )
    return pred, pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    votes = load_votes(run_dir)
    df, groups = build_note_features(votes)

    full_reference = unique_features(
        [
            groups["vote"],
            groups["official_helpful_reasons"],
            groups["official_not_helpful_reasons"],
            groups["confidence"],
            groups["understanding"],
            groups["per_agent"],
        ]
    )
    official_reasons = unique_features(
        [groups["official_helpful_reasons"], groups["official_not_helpful_reasons"]]
    )

    experiments: dict[str, list[str]] = {
        "full_reference": full_reference,
        "vote_only": groups["vote"],
        "vote_plus_official_helpful_reasons": unique_features(
            [groups["vote"], groups["official_helpful_reasons"]]
        ),
        "vote_plus_official_not_helpful_reasons": unique_features(
            [groups["vote"], groups["official_not_helpful_reasons"]]
        ),
        "vote_plus_confidence": unique_features([groups["vote"], groups["confidence"]]),
        "vote_plus_understanding": unique_features([groups["vote"], groups["understanding"]]),
        "drop_confidence": [f for f in full_reference if f not in set(groups["confidence"])],
        "drop_changes_reader_understanding": [
            f for f in full_reference if f not in set(groups["understanding"])
        ],
        "drop_per_agent_features": [f for f in full_reference if f not in set(groups["per_agent"])],
        "official_reasons_only_no_custom": official_reasons,
    }
    experiments = {k: [f for f in v if f in df.columns] for k, v in experiments.items()}

    y = df["true_label_3way"].to_numpy(int)
    preds = df[["noteId", "true_label_3way", "true_label_text"]].copy()
    summary_rows: list[dict[str, float | int | str]] = []
    fold_tables: list[pd.DataFrame] = []

    total = len(experiments)
    for idx, (name, features) in enumerate(experiments.items(), start=1):
        print(f"[{idx}/{total}] running {name} with {len(features)} features", flush=True)
        pred, fold_df = nested_cv_lr(
            df=df,
            features=features,
            folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed + idx * 101,
        )
        preds[name] = pred
        overall = metric(y, pred)
        fold_df = fold_df.assign(method=name, n_features=len(features))
        fold_tables.append(fold_df)
        row = {"method": name, "n_features": len(features), **overall}
        row["fold_accuracy_mean"] = float(fold_df["test_accuracy"].mean())
        row["fold_accuracy_se"] = float(
            fold_df["test_accuracy"].std(ddof=1) / np.sqrt(len(fold_df))
        )
        row["fold_balanced_accuracy_mean"] = float(fold_df["test_balanced_accuracy"].mean())
        row["fold_balanced_accuracy_se"] = float(
            fold_df["test_balanced_accuracy"].std(ddof=1) / np.sqrt(len(fold_df))
        )
        summary_rows.append(row)
        print(
            f"[{idx}/{total}] done {name}: "
            f"acc={100 * overall['accuracy']:.2f}, "
            f"bal={100 * overall['balanced_accuracy']:.2f}, "
            f"H={100 * overall['recall_helpful']:.2f}, "
            f"NMR={100 * overall['recall_needs_more_ratings']:.2f}, "
            f"NH={100 * overall['recall_not_helpful']:.2f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    for col in [
        "accuracy",
        "balanced_accuracy",
        "recall_not_helpful",
        "recall_needs_more_ratings",
        "recall_helpful",
        "fold_accuracy_mean",
        "fold_accuracy_se",
        "fold_balanced_accuracy_mean",
        "fold_balanced_accuracy_se",
    ]:
        summary[f"{col}_pct"] = summary[col] * 100
    summary = summary.sort_values(
        ["balanced_accuracy", "accuracy", "cross_error"], ascending=[False, False, True]
    )

    best_method = str(summary.iloc[0]["method"])
    best_confusion = pd.crosstab(
        df["true_label_text"],
        pd.Series(preds[best_method]).map(LABEL),
        margins=True,
    )

    df.to_csv(out_dir / "note_feature_table.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(out_dir / "oof_predictions_by_feature_group.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(fold_tables, ignore_index=True).to_csv(
        out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    best_confusion.to_csv(out_dir / "best_confusion.csv", encoding="utf-8-sig")
    (out_dir / "feature_groups.json").write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "n_notes": int(len(df)),
                "n_votes": int(len(votes)),
                "class_counts": df["true_label_text"].value_counts().to_dict(),
                "best_method": best_method,
                "best": summary.iloc[0].to_dict(),
                "official_helpful_reason_fields": HELPFUL_REASONS,
                "official_not_helpful_reason_fields": NOT_HELPFUL_REASONS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    show = [
        "method",
        "n_features",
        "accuracy_pct",
        "balanced_accuracy_pct",
        "recall_helpful_pct",
        "recall_needs_more_ratings_pct",
        "recall_not_helpful_pct",
        "h_to_nh",
        "nh_to_h",
        "cross_error",
        "fold_accuracy_se_pct",
    ]
    print("\n=== Ranked summary ===")
    print(summary[show].to_string(index=False))
    print("\n=== Best confusion ===")
    print(best_confusion.to_string())
    print(f"\nSaved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
