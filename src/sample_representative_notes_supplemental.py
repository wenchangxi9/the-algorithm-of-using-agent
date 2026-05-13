#!/usr/bin/env python3
"""Build a supplemental representative Community Notes sample.

This is the same year x status x primary-topic stratified design used for the
20k sample, with explicit exclusion of previously sampled noteIds/tweetIds.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STATUS_MAP = {
    "CURRENTLY_RATED_HELPFUL": "Helpful",
    "CURRENTLY_RATED_NOT_HELPFUL": "Not Helpful",
    "NEEDS_MORE_RATINGS": "Need More Ratings",
}

MISLEADING_TOPIC_ORDER = [
    ("misleadingManipulatedMedia", "Manipulated media"),
    ("misleadingFactualError", "Factual error"),
    ("misleadingOutdatedInformation", "Outdated information"),
    ("misleadingMissingImportantContext", "Missing important context"),
    ("misleadingUnverifiedClaimAsFact", "Unverified claim as fact"),
    ("misleadingSatire", "Satire"),
]

NOT_MISLEADING_COLUMNS = [
    "notMisleadingOther",
    "notMisleadingFactuallyCorrect",
    "notMisleadingOutdatedButNotWhenWritten",
    "notMisleadingClearlySatire",
    "notMisleadingPersonalOpinion",
]

NOTE_USECOLS = [
    "noteId",
    "noteAuthorParticipantId",
    "createdAtMillis",
    "tweetId",
    "classification",
    "summary",
    "isMediaNote",
    "isCollaborativeNote",
    *[col for col, _ in MISLEADING_TOPIC_ORDER],
    *NOT_MISLEADING_COLUMNS,
]

STATUS_USECOLS = [
    "noteId",
    "currentStatus",
    "currentCoreStatus",
    "currentExpansionStatus",
    "currentGroupStatus",
    "timestampMillisOfCurrentStatus",
]


class ProgressReporter:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.progress_log = outdir / "progress.log"
        self.progress_json = outdir / "progress.json"
        self.outdir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def bar(done: int, total: int, width: int = 30) -> str:
        if total <= 0:
            return "[" + "-" * width + "]"
        filled = int(round(width * min(done, total) / total))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def update(self, stage: str, done: int, total: int, detail: str = "") -> None:
        pct = float(done / total * 100) if total else 0.0
        payload = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "percent": pct,
            "bar": self.bar(done, total),
            "detail": detail,
        }
        line = (
            f"{payload['time_utc']} | {stage:<28} "
            f"{payload['bar']} {pct:6.2f}% ({done}/{total}) {detail}\n"
        )
        with open(self.progress_log, "a", encoding="utf-8") as f:
            f.write(line)
        tmp = self.progress_json.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp.replace(self.progress_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted_communitynotes_2026-04-07"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--exclude-sample", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--prefix", default="sample_7k_supplement")
    return parser.parse_args()


def read_tsv_file(file: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(file, sep="\t", usecols=lambda col: col in usecols, low_memory=False)


def read_tsv_dir(path: Path, usecols: list[str], workers: int, progress: ProgressReporter, stage: str) -> pd.DataFrame:
    files = sorted(path.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No TSV files found under {path}")
    progress.update(stage, 0, len(files), "start")
    if workers <= 1 or len(files) == 1:
        frames = []
        for idx, file in enumerate(files, start=1):
            frames.append(read_tsv_file(file, usecols))
            progress.update(stage, idx, len(files), file.name)
    else:
        frames = []
        with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
            futures = {pool.submit(read_tsv_file, file, usecols): file for file in files}
            for idx, future in enumerate(as_completed(futures), start=1):
                frames.append(future.result())
                progress.update(stage, idx, len(files), futures[future].name)
    progress.update(stage, len(files), len(files), "done")
    return pd.concat(frames, ignore_index=True)


def as_int_flag(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(float(value))
    except Exception:
        return 0


def derive_topic_columns(df: pd.DataFrame) -> pd.DataFrame:
    topics_per_row: list[list[str]] = [[] for _ in range(len(df))]

    for col, label in MISLEADING_TOPIC_ORDER:
        flag = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int).to_numpy()
        for i in np.flatnonzero(flag == 1):
            topics_per_row[int(i)].append(label)

    not_misleading = (df.get("classification", "").fillna("").astype(str) == "NOT_MISLEADING").to_numpy()
    for col in NOT_MISLEADING_COLUMNS:
        if col in df.columns:
            not_misleading |= pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    for i in np.flatnonzero(not_misleading):
        topics_per_row[int(i)].append("Not misleading")

    primary = []
    all_json = []
    topic_count = []
    for topics in topics_per_row:
        if not topics:
            topics = ["Other"]
        primary.append(topics[0])
        all_json.append(json.dumps(topics, ensure_ascii=False))
        topic_count.append(len(topics))

    return pd.DataFrame(
        {
            "primary_topic": primary,
            "all_topics_json": all_json,
            "topic_count": topic_count,
        },
        index=df.index,
    )


def allocate_largest_remainder(counts: pd.Series, sample_size: int) -> pd.Series:
    total = int(counts.sum())
    quotas = counts.astype(float) * sample_size / total
    base = np.floor(quotas).astype(int)
    remainder_needed = int(sample_size - base.sum())
    if remainder_needed > 0:
        remainders = (quotas - base).sort_values(ascending=False)
        base.loc[remainders.index[:remainder_needed]] += 1
    elif remainder_needed < 0:
        remainders = (quotas - base).sort_values(ascending=True)
        for idx in remainders.index[: -remainder_needed]:
            if base.loc[idx] > 0:
                base.loc[idx] -= 1
    assert int(base.sum()) == sample_size
    return base


def redistribute_unavailable(allocation: pd.Series, available: pd.Series) -> pd.Series:
    allocation = allocation.copy().astype(int)
    available = available.reindex(allocation.index).fillna(0).astype(int)
    over = allocation > available
    deficit = int((allocation[over] - available[over]).sum())
    allocation[over] = available[over]
    if deficit <= 0:
        return allocation

    capacity = (available - allocation).clip(lower=0)
    if int(capacity.sum()) < deficit:
        raise ValueError("Not enough remaining rows after exclusions to satisfy sample size.")
    order = capacity[capacity > 0].sort_values(ascending=False).index.tolist()
    pos = 0
    while deficit > 0:
        idx = order[pos % len(order)]
        if allocation.loc[idx] < available.loc[idx]:
            allocation.loc[idx] += 1
            deficit -= 1
        pos += 1
    return allocation


def distribution_table(population: pd.DataFrame, sample: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    pop = population.groupby(columns, dropna=False).size().rename("population_count").reset_index()
    sam = sample.groupby(columns, dropna=False).size().rename("sample_count").reset_index()
    out = pop.merge(sam, on=columns, how="outer").fillna({"population_count": 0, "sample_count": 0})
    out["population_count"] = out["population_count"].astype(int)
    out["sample_count"] = out["sample_count"].astype(int)
    out["population_prop"] = out["population_count"] / max(1, out["population_count"].sum())
    out["sample_prop"] = out["sample_count"] / max(1, out["sample_count"].sum())
    out["abs_prop_diff"] = (out["sample_prop"] - out["population_prop"]).abs()
    out.insert(0, "distribution", name)
    return out.sort_values(["distribution", *columns]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(args.outdir)
    progress.update("initialize", 0, 1, "starting supplemental representative sample")

    notes = read_tsv_dir(args.data_root / "notes", NOTE_USECOLS, args.workers, progress, "read notes shards")
    status = read_tsv_dir(args.data_root / "noteStatusHistory", STATUS_USECOLS, args.workers, progress, "read status shards")

    progress.update("preprocess ids/status", 0, 7, "normalize identifiers")
    notes["noteId"] = notes["noteId"].astype(str)
    notes["tweetId"] = pd.to_numeric(notes["tweetId"], errors="coerce").astype("Int64")
    notes["createdAtMillis"] = pd.to_numeric(notes["createdAtMillis"], errors="coerce")
    progress.update("preprocess ids/status", 1, 7, "notes normalized")
    status["noteId"] = status["noteId"].astype(str)
    status["timestampMillisOfCurrentStatus"] = pd.to_numeric(status.get("timestampMillisOfCurrentStatus"), errors="coerce")
    progress.update("preprocess ids/status", 2, 7, "status normalized")
    status = status.sort_values(["noteId", "timestampMillisOfCurrentStatus"]).drop_duplicates("noteId", keep="last")
    progress.update("preprocess ids/status", 3, 7, "status deduplicated")

    df = notes.merge(status, on="noteId", how="inner", validate="one_to_one")
    progress.update("preprocess ids/status", 4, 7, f"merged rows={len(df):,}")
    df = df[df["currentStatus"].isin(STATUS_MAP)].copy()
    df = df[df["tweetId"].notna() & (df["tweetId"].astype("int64") > 0)].copy()
    df = df[df["summary"].fillna("").astype(str).str.strip().ne("")].copy()
    progress.update("preprocess ids/status", 5, 7, f"filtered rows={len(df):,}")

    created_dt = pd.to_datetime(df["createdAtMillis"], unit="ms", utc=True, errors="coerce")
    df["createdAtUTC"] = created_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["year"] = created_dt.dt.year.astype("Int64")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df["status_label"] = df["currentStatus"].map(STATUS_MAP)
    progress.update("preprocess ids/status", 6, 7, "years/status labels")

    progress.update("derive topics", 0, 1, "vectorized topic derivation")
    topic_cols = derive_topic_columns(df)
    df["primary_topic"] = topic_cols["primary_topic"]
    df["all_topics_json"] = topic_cols["all_topics_json"]
    df["topic_count"] = topic_cols["topic_count"]
    progress.update("derive topics", 1, 1, "done")

    exclude = pd.read_csv(args.exclude_sample, dtype={"noteId": str, "tweetId": str})
    exclude_note_ids = set(exclude["noteId"].dropna().astype(str))
    exclude_tweet_ids = set(exclude["tweetId"].dropna().astype(str))
    df["tweetIdStr"] = df["tweetId"].astype("int64").astype(str)
    candidate = df[~df["noteId"].isin(exclude_note_ids) & ~df["tweetIdStr"].isin(exclude_tweet_ids)].copy()
    progress.update(
        "exclude previous sample",
        1,
        1,
        f"exclude_notes={len(exclude_note_ids):,} exclude_tweets={len(exclude_tweet_ids):,} remaining={len(candidate):,}",
    )

    strata_cols = ["year", "status_label", "primary_topic"]
    progress.update("compute allocation", 0, 3, "group by year/status/topic")
    population_counts = df.groupby(strata_cols, dropna=False).size().sort_index()
    target_allocation = allocate_largest_remainder(population_counts, args.sample_size)
    available_counts = candidate.groupby(strata_cols, dropna=False).size().sort_index()
    allocation = redistribute_unavailable(target_allocation, available_counts)
    progress.update("compute allocation", 3, 3, f"strata={len(population_counts):,}")

    rng_seed = int(args.seed)

    def sample_one_stratum(task: tuple[int, tuple[object, ...], pd.DataFrame]) -> pd.DataFrame | None:
        i, key, group = task
        n = int(allocation.loc[key]) if key in allocation.index else 0
        if n <= 0:
            return None
        return group.sample(n=n, replace=False, random_state=rng_seed + i)

    grouped = list(candidate.groupby(strata_cols, dropna=False, sort=True))
    tasks = [(i, key, group) for i, (key, group) in enumerate(grouped)]
    progress.update("sample strata", 0, len(tasks), "start")
    sampled_parts = []
    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            sampled_parts.append(sample_one_stratum(task))
            progress.update("sample strata", idx, len(tasks), f"stratum {idx}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(sample_one_stratum, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                sampled_parts.append(future.result())
                if idx == 1 or idx == len(futures) or idx % 25 == 0:
                    progress.update("sample strata", idx, len(tasks), f"completed {idx} strata")
    sampled_parts = [part for part in sampled_parts if part is not None]
    sample = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    assert len(sample) == args.sample_size
    progress.update("sample strata", len(tasks), len(tasks), f"sample rows={len(sample):,}")

    keep_cols = [
        "noteId",
        "tweetId",
        "noteAuthorParticipantId",
        "createdAtMillis",
        "createdAtUTC",
        "year",
        "currentStatus",
        "status_label",
        "currentCoreStatus",
        "currentExpansionStatus",
        "currentGroupStatus",
        "classification",
        "primary_topic",
        "all_topics_json",
        "topic_count",
        "summary",
        "isMediaNote",
        "isCollaborativeNote",
        *[col for col, _ in MISLEADING_TOPIC_ORDER],
        *NOT_MISLEADING_COLUMNS,
    ]
    sample_out = sample[keep_cols].copy()
    sample_out.insert(0, "sample_id", np.arange(1, len(sample_out) + 1, dtype=int))
    sample_out["post_text"] = ""
    sample_out["post_fetch_status"] = "pending"
    sample_out["post_fetch_source"] = ""

    notes_file = args.outdir / f"{args.prefix}_notes.csv"
    queue_file = args.outdir / f"{args.prefix}_post_fetch_queue.csv"
    sample_out.to_csv(notes_file, index=False, encoding="utf-8-sig")
    sample_out[["sample_id", "noteId", "tweetId"]].to_csv(queue_file, index=False, encoding="utf-8-sig")

    allocation_df = population_counts.rename("population_count").reset_index()
    allocation_df["target_sample_count"] = target_allocation.values
    allocation_df["final_sample_count"] = allocation.values
    allocation_df["available_after_exclusion"] = available_counts.reindex(population_counts.index).fillna(0).astype(int).values
    allocation_df["population_prop"] = allocation_df["population_count"] / allocation_df["population_count"].sum()
    allocation_df["final_sample_prop"] = allocation_df["final_sample_count"] / args.sample_size
    allocation_df["abs_prop_diff"] = (allocation_df["final_sample_prop"] - allocation_df["population_prop"]).abs()
    allocation_df.to_csv(args.outdir / f"{args.prefix}_stratum_allocation_year_status_topic.csv", index=False)

    reports = [
        distribution_table(df, sample, ["year"], "year"),
        distribution_table(df, sample, ["status_label"], "status"),
        distribution_table(df, sample, ["primary_topic"], "primary_topic"),
        distribution_table(df, sample, ["year", "status_label"], "year_status"),
        distribution_table(df, sample, ["status_label", "primary_topic"], "status_topic"),
        distribution_table(df, sample, strata_cols, "year_status_topic"),
    ]
    distribution = pd.concat(reports, ignore_index=True)
    distribution.to_csv(args.outdir / f"{args.prefix}_distribution_check.csv", index=False)

    flag_cols = [col for col, _ in MISLEADING_TOPIC_ORDER] + NOT_MISLEADING_COLUMNS
    flag_rows = []
    for col in flag_cols:
        pop_prop = df[col].map(as_int_flag).mean()
        sample_prop = sample[col].map(as_int_flag).mean()
        flag_rows.append(
            {
                "flag": col,
                "population_prop": float(pop_prop),
                "sample_prop": float(sample_prop),
                "abs_prop_diff": float(abs(sample_prop - pop_prop)),
            }
        )
    pd.DataFrame(flag_rows).to_csv(args.outdir / f"{args.prefix}_topic_multilabel_flag_check.csv", index=False)

    summary = {
        "sample_size": int(len(sample_out)),
        "seed": args.seed,
        "sampling_frame_notes": int(len(df)),
        "candidate_after_exclusion": int(len(candidate)),
        "excluded_note_ids": int(len(exclude_note_ids)),
        "excluded_tweet_ids": int(len(exclude_tweet_ids)),
        "unique_tweets_in_sample": int(sample_out["tweetId"].astype(str).nunique()),
        "max_abs_year_prop_diff": float(distribution.loc[distribution["distribution"] == "year", "abs_prop_diff"].max()),
        "max_abs_status_prop_diff": float(distribution.loc[distribution["distribution"] == "status", "abs_prop_diff"].max()),
        "max_abs_primary_topic_prop_diff": float(distribution.loc[distribution["distribution"] == "primary_topic", "abs_prop_diff"].max()),
        "max_abs_joint_prop_diff": float(distribution.loc[distribution["distribution"] == "year_status_topic", "abs_prop_diff"].max()),
        "notes_file": str(notes_file),
        "queue_file": str(queue_file),
    }
    with open(args.outdir / f"{args.prefix}_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    progress.update("complete", 1, 1, "supplemental representative sample complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
