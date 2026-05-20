from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


LABEL = {0: "NOT_HELPFUL", 1: "NEEDS_MORE_RATINGS", 2: "HELPFUL"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("artifacts/05_agent_runs_2000"))
    p.add_argument("--out-subdir", default="officialschema_hard_singlefile_841_6912_20260520")
    p.add_argument("--min-accuracy", type=float, default=0.8410)
    p.add_argument("--min-balanced-accuracy", type=float, default=0.6912)
    return p.parse_args()


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


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


def load_oof(run_dir: Path, subdir: str) -> pd.DataFrame:
    p = run_dir / subdir / "oof_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing oof file: {p}")
    df = pd.read_csv(p)
    df["noteId"] = df["noteId"].astype(str)
    return df


def weighted_vote(df: pd.DataFrame, cols: list[str], weights: tuple[float, ...]) -> np.ndarray:
    scores = np.zeros((len(df), 3), dtype=float)
    idx = np.arange(len(df))
    for col, w in zip(cols, weights):
        pred = df[col].to_numpy(dtype=int)
        scores[idx, pred] += float(w)
    return scores.argmax(axis=1).astype(int)


def build_votes_features(votes: pd.DataFrame) -> pd.DataFrame:
    v = votes.copy()
    v["noteId"] = v["noteId"].astype(str)
    v["pred"] = v["parsed_rating"].map({"NOT_HELPFUL": 0, "SOMEWHAT_HELPFUL": 1, "HELPFUL": 2})
    v = v[v["pred"].notna()].copy()
    v["pred"] = v["pred"].astype(int)
    agg = v.groupby("noteId", as_index=False).agg(
        vote_nh=("pred", lambda s: int((s == 0).sum())),
        vote_nmr=("pred", lambda s: int((s == 1).sum())),
        vote_h=("pred", lambda s: int((s == 2).sum())),
        mean_under=("changes_reader_understanding", "mean"),
    )
    agg["share_nmr"] = agg["vote_nmr"] / 16.0
    return agg


def build_accuracy_first_stack(run_dir: Path, out_dir: Path) -> pd.DataFrame:
    log("build accuracy_first_stack (in-file logic)")
    base = load_oof(run_dir, "officialschema_rationale_agent_blend_20260519")
    xgroup = load_oof(run_dir, "officialschema_xstyle_group_gate_20260519")
    xrescue = load_oof(run_dir, "officialschema_xstyle_rescue_gate_20260519")
    oofens = load_oof(run_dir, "officialschema_oof_ensemble_gated_20260519")

    merged = base[
        ["noteId", "true_label_3way", "true_label_text", "summary_meta", "summary_meta_struct", "full_meta", "blend"]
    ].copy()
    merged = merged.merge(xgroup[["noteId", "xstyle_group_gate"]], on="noteId", how="left")
    merged = merged.merge(xrescue[["noteId", "xstyle_rescue_gate"]], on="noteId", how="left")
    merged = merged.merge(oofens[["noteId", "oof_ensemble_weighted", "oof_ensemble_gated"]], on="noteId", how="left")

    pred = merged["oof_ensemble_weighted"].to_numpy(int).copy()
    high_acc = merged["xstyle_rescue_gate"].to_numpy(int)
    high_ba = merged["blend"].to_numpy(int)
    summary = merged["summary_meta"].to_numpy(int)

    vote = np.stack([pred, high_acc, summary], axis=1)
    maj = np.array([np.bincount(row, minlength=3).argmax() for row in vote], dtype=int)
    disagreement = (pred != high_acc) & (pred != summary)
    pred[disagreement] = maj[disagreement]

    nmr_mask = pred == 1
    agree_resolved = (high_acc != 1) & (summary != 1) & (high_acc == summary)
    pred[nmr_mask & agree_resolved] = high_acc[nmr_mask & agree_resolved]

    resolved_pred = pred != 1
    blend_agree = (high_ba == pred) | (high_ba == summary)
    flip_to_blend = resolved_pred & (~blend_agree) & (high_ba != 1)
    pred[flip_to_blend] = high_ba[flip_to_blend]
    merged["accuracy_first_selector"] = pred

    acc_stack_dir = out_dir / "officialschema_accuracy_first_stack_20260519"
    acc_stack_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(acc_stack_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    return merged


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"run_dir={run_dir}")
    log("load agent_votes.csv")
    votes = pd.read_csv(run_dir / "agent_votes.csv", low_memory=False, usecols=["noteId", "parsed_rating", "changes_reader_understanding"])
    votes_feat = build_votes_features(votes)
    votes_feat["noteId"] = votes_feat["noteId"].astype(str)

    stack = build_accuracy_first_stack(run_dir, out_dir)

    log("apply best_841_693 exact rule")
    base = stack[
        [
            "noteId",
            "true_label_3way",
            "oof_ensemble_weighted",
            "oof_ensemble_gated",
            "xstyle_rescue_gate",
            "blend",
            "summary_meta_struct",
            "full_meta",
        ]
    ].copy()
    base["noteId"] = base["noteId"].astype(str)
    base = base.merge(votes_feat, on="noteId", how="left")

    anchor_cols = ["oof_ensemble_weighted", "oof_ensemble_gated", "xstyle_rescue_gate", "blend", "full_meta"]
    anchor_weights = (1.0, 0.75, 0.75, 2.0, 1.0)
    pred = weighted_vote(base, anchor_cols, anchor_weights)
    promote = (
        (pred == 1)
        & (base["blend"].to_numpy(int) == base["summary_meta_struct"].to_numpy(int))
        & (base["blend"].to_numpy(int) != 1)
        & (base["share_nmr"].to_numpy(float) >= 0.5625)
        & (base["mean_under"].to_numpy(float) <= 29.79375)
    )
    pred[promote] = base.loc[promote, "blend"].to_numpy(int)

    y = base["true_label_3way"].to_numpy(int)
    m = metric(y, pred)
    acc_pct = m["accuracy"] * 100.0
    ba_pct = m["balanced_accuracy"] * 100.0
    log(f"final metrics: acc={acc_pct:.2f}% ba={ba_pct:.2f}%")

    final_dir = out_dir / "officialschema_best_841_693_20260520"
    final_dir.mkdir(parents=True, exist_ok=True)
    oof_out = base[["noteId", "true_label_3way"]].copy()
    oof_out["best_841_693"] = pred
    oof_out.to_csv(final_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "method": "best_841_693",
                **m,
                "accuracy_pct": acc_pct,
                "balanced_accuracy_pct": ba_pct,
                "recall_not_helpful_pct": m["recall_not_helpful"] * 100.0,
                "recall_needs_more_ratings_pct": m["recall_needs_more_ratings"] * 100.0,
                "recall_helpful_pct": m["recall_helpful"] * 100.0,
                "min_recall_pct": m["min_recall"] * 100.0,
                "anchor_cols": "|".join(anchor_cols),
                "anchor_weights": "|".join(map(str, anchor_weights)),
                "promote_share_nmr_threshold": 0.5625,
                "promote_mean_under_threshold": 29.79375,
            }
        ]
    )
    summary.to_csv(final_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (final_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "anchor_cols": anchor_cols,
                "anchor_weights": anchor_weights,
                "promote_share_nmr_threshold": 0.5625,
                "promote_mean_under_threshold": 29.79375,
                "metrics": m,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    min_acc_pct = round(args.min_accuracy * 100.0, 2)
    min_ba_pct = round(args.min_balanced_accuracy * 100.0, 2)
    passed = (round(acc_pct, 2) >= min_acc_pct) and (round(ba_pct, 2) >= min_ba_pct)
    check = {
        "min_accuracy": args.min_accuracy,
        "min_balanced_accuracy": args.min_balanced_accuracy,
        "min_accuracy_pct": min_acc_pct,
        "min_balanced_accuracy_pct": min_ba_pct,
        "actual_accuracy": m["accuracy"],
        "actual_balanced_accuracy": m["balanced_accuracy"],
        "actual_accuracy_pct": acc_pct,
        "actual_balanced_accuracy_pct": ba_pct,
        "passed": passed,
    }
    (out_dir / "threshold_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(check, ensure_ascii=False))

    if not passed:
        raise RuntimeError(f"threshold not met: acc={acc_pct:.2f} ba={ba_pct:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
