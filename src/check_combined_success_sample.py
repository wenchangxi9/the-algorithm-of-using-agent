#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("analysis/representative_20k_sample_20260511"))
    parser.add_argument("--supp-dir", type=Path, default=Path("analysis/representative_7k_supplement_20260511"))
    parser.add_argument("--outdir", type=Path, default=Path("analysis/combined_success_sample_20260511"))
    return parser.parse_args()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", **kwargs)


def distribution_check(pop_distribution_path: Path, sample: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    pop = pd.read_csv(pop_distribution_path)
    pop = pop[pop["distribution"] == name].copy()
    pop_cols = ["population_count", "population_prop", *cols]
    pop = pop[pop_cols]
    sample_norm = sample.copy()
    for col in cols:
        if col == "year":
            pop[col] = pd.to_numeric(pop[col], errors="coerce").astype("Int64").astype(str)
            sample_norm[col] = pd.to_numeric(sample_norm[col], errors="coerce").astype("Int64").astype(str)
        else:
            pop[col] = pop[col].fillna("").astype(str)
            sample_norm[col] = sample_norm[col].fillna("").astype(str)
    sam = sample_norm.groupby(cols, dropna=False).size().rename("sample_count").reset_index()
    out = pop.merge(sam, on=cols, how="outer").fillna({"sample_count": 0})
    out["sample_count"] = out["sample_count"].astype(int)
    out["sample_prop"] = out["sample_count"] / max(1, out["sample_count"].sum())
    out["abs_prop_diff"] = (out["sample_prop"] - out["population_prop"]).abs()
    out.insert(0, "distribution", name)
    return out.sort_values(["distribution", *cols]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    base_notes = read_csv(args.base_dir / "sample_20k_notes.csv")
    supp_notes = read_csv(args.supp_dir / "sample_7k_supplement_notes.csv")
    base_notes["source_batch"] = "base_20k"
    supp_notes["source_batch"] = "supp_7k"
    notes = pd.concat([base_notes, supp_notes], ignore_index=True)

    base_success = read_csv(args.base_dir / "post_fetch_tikhub" / "posts_success.csv")
    supp_success = read_csv(args.supp_dir / "post_fetch_tikhub" / "posts_success.csv")
    success = pd.concat([base_success, supp_success], ignore_index=True)
    success["tweetId"] = success["tweetId"].astype(str)
    success = success.drop_duplicates("tweetId", keep="first")

    notes["tweetId"] = notes["tweetId"].astype(str)
    success_cols = ["tweetId", "text", "created_at", "author", "fetched_at_utc"]
    merged = notes.merge(success[success_cols], on="tweetId", how="left")
    success_notes = merged[merged["text"].fillna("").astype(str).str.strip().ne("")].copy()

    merged.to_csv(args.outdir / "combined_27k_notes_with_post_fetch_status.csv", index=False, encoding="utf-8-sig")
    success_notes.to_csv(args.outdir / "combined_success_notes_with_posts.csv", index=False, encoding="utf-8-sig")

    pop_distribution = args.base_dir / "sample_distribution_check.csv"
    reports = [
        distribution_check(pop_distribution, success_notes, ["year"], "year"),
        distribution_check(pop_distribution, success_notes, ["status_label"], "status"),
        distribution_check(pop_distribution, success_notes, ["primary_topic"], "primary_topic"),
        distribution_check(pop_distribution, success_notes, ["year", "status_label"], "year_status"),
        distribution_check(pop_distribution, success_notes, ["status_label", "primary_topic"], "status_topic"),
        distribution_check(pop_distribution, success_notes, ["year", "status_label", "primary_topic"], "year_status_topic"),
    ]
    dist = pd.concat(reports, ignore_index=True)
    dist.to_csv(args.outdir / "combined_success_distribution_check.csv", index=False)

    batch_counts = (
        success_notes.groupby("source_batch", dropna=False)
        .size()
        .rename("success_note_count")
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "candidate_notes_total": int(len(notes)),
        "candidate_unique_tweets_total": int(notes["tweetId"].nunique()),
        "success_unique_tweets_total": int(success["tweetId"].nunique()),
        "success_notes_total": int(len(success_notes)),
        "base_success_notes": int((success_notes["source_batch"] == "base_20k").sum()),
        "supp_success_notes": int((success_notes["source_batch"] == "supp_7k").sum()),
        "failed_or_missing_post_notes": int(len(merged) - len(success_notes)),
        "batch_counts": batch_counts,
        "max_abs_year_prop_diff": float(dist.loc[dist["distribution"] == "year", "abs_prop_diff"].max()),
        "max_abs_status_prop_diff": float(dist.loc[dist["distribution"] == "status", "abs_prop_diff"].max()),
        "max_abs_primary_topic_prop_diff": float(dist.loc[dist["distribution"] == "primary_topic", "abs_prop_diff"].max()),
        "max_abs_joint_prop_diff": float(dist.loc[dist["distribution"] == "year_status_topic", "abs_prop_diff"].max()),
        "outputs": [
            "combined_27k_notes_with_post_fetch_status.csv",
            "combined_success_notes_with_posts.csv",
            "combined_success_distribution_check.csv",
        ],
    }
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
