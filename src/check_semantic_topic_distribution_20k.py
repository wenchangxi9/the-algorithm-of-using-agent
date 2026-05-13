#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TOPIC_BUCKETS = [
    "politics_elections_government",
    "war_geopolitics_international",
    "health_science_medicine",
    "economy_business_crypto",
    "crime_law_public_safety",
    "entertainment_sports_gaming",
    "social_culture_identity",
    "platform_media_internet",
]

TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "politics_elections_government": (
        r"\btrump\b",
        r"\bbiden\b",
        r"\bgop\b",
        r"\bdemocrat",
        r"\brepublican",
        r"\bgovern(or|ment)\b",
        r"\bpresident\b",
        r"\bprime minister\b",
        r"\bcongress\b",
        r"\bsenate\b",
        r"\bparliament\b",
        r"\belection",
        r"\bvote",
        r"\bcampaign",
        r"\bconstitution",
        r"\bminister\b",
        r"\bmayor\b",
        r"\blabour\b",
        r"\btories?\b",
        r"\bkamala\b",
        r"\bdesantis\b",
        r"\blula\b",
        r"\bmaduro\b",
    ),
    "war_geopolitics_international": (
        r"\brussia\b",
        r"\bukrain",
        r"\bputin\b",
        r"\bnato\b",
        r"\bcrimea\b",
        r"\bwar\b",
        r"\bmissile\b",
        r"\bmilitary\b",
        r"\binvasion\b",
        r"\bterroris",
        r"\bisrael\b",
        r"\bpalestin",
        r"\bgaza\b",
        r"\biran\b",
        r"\bhezbollah\b",
        r"\bhamas\b",
        r"\bkremlin\b",
        r"\bworld war\b",
        r"\bforeign minister\b",
    ),
    "health_science_medicine": (
        r"\bcovid\b",
        r"\bvaccine",
        r"\bfluoride\b",
        r"\bdoctor\b",
        r"\bmedical\b",
        r"\bmedicine\b",
        r"\bscient",
        r"\bjournal\b",
        r"\bdepression\b",
        r"\bmental health\b",
        r"\bcancer\b",
        r"\bvirus\b",
        r"\bflu\b",
        r"\btoxic\b",
        r"\bpublic health\b",
        r"\bhospital\b",
        r"\bpharma",
    ),
    "economy_business_crypto": (
        r"\bminimum wage\b",
        r"\bwage\b",
        r"\bbitcoin\b",
        r"\bcrypto",
        r"\bdogecoin\b",
        r"\bshiba\b",
        r"\bdollar\b",
        r"\byuan\b",
        r"\bbank",
        r"\binflation\b",
        r"\btax",
        r"\bbusiness\b",
        r"\bstock",
        r"\beconom",
        r"\bprice",
        r"\bgiveaway\b",
        r"\bfinancial\b",
        r"\bscam\b",
    ),
    "crime_law_public_safety": (
        r"\bpolice\b",
        r"\bcrime\b",
        r"\bcriminal\b",
        r"\blawyer\b",
        r"\bcourt\b",
        r"\barrest",
        r"\bprison",
        r"\bmurder",
        r"\bkill",
        r"\bassassin",
        r"\briot\b",
        r"\bviolen",
        r"\bsafety\b",
        r"\bfire\b",
        r"\bholocaust\b",
        r"\bauschwitz\b",
        r"\bdeath\b",
    ),
    "entertainment_sports_gaming": (
        r"\bminecraft\b",
        r"\bplaystation\b",
        r"\bps5\b",
        r"\bchess\b",
        r"\bmovie\b",
        r"\bfilm\b",
        r"\bcinema\b",
        r"\bconcert\b",
        r"\bmusic\b",
        r"\bfootball\b",
        r"\bboxing\b",
        r"\bworld cup\b",
        r"\bmbapp",
        r"\bgame\b",
        r"\bgaming\b",
        r"\banime\b",
        r"\bolympic",
        r"\bactor\b",
        r"\bsitcom\b",
    ),
    "social_culture_identity": (
        r"\btrans",
        r"\bwoman\b",
        r"\bwomen\b",
        r"\bman\b",
        r"\bmen\b",
        r"\brace\b",
        r"\bracist\b",
        r"\bwhite\b",
        r"\bblack\b",
        r"\bgender\b",
        r"\bsexis",
        r"\btransphob",
        r"\bidentity\b",
        r"\bqueer\b",
        r"\bimmigrant\b",
        r"\breligion\b",
        r"\bjewish\b",
        r"\bmuslim\b",
    ),
    "platform_media_internet": (
        r"\bpost\b",
        r"\btweet\b",
        r"\bheadline\b",
        r"\barticle\b",
        r"\bmedia\b",
        r"\bvideo\b",
        r"\bimage\b",
        r"\bphoto\b",
        r"\bscreenshot\b",
        r"\bdeep ?fake\b",
        r"\bai\b",
        r"\baccount\b",
        r"\bhandle\b",
        r"\bverified\b",
        r"\bengagement\b",
        r"\bbot\b",
        r"\bviral\b",
        r"\bclickbait\b",
        r"\bfake news\b",
        r"\bmisinfo\b",
        r"\bscam\b",
        r"\bspam\b",
        r"https?://",
        r"\bx\.com/",
        r"\bt\.co/",
    ),
}

COMPILED_PATTERNS = {
    bucket: tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)
    for bucket, patterns in TOPIC_PATTERNS.items()
}

TOPIC_REGEX_UNIONS = {
    bucket: "(?:" + ")|(?:".join(patterns) + ")"
    for bucket, patterns in TOPIC_PATTERNS.items()
}

STATUS_MAP = {
    "CURRENTLY_RATED_HELPFUL": "Helpful",
    "CURRENTLY_RATED_NOT_HELPFUL": "Not Helpful",
    "NEEDS_MORE_RATINGS": "Need More Ratings",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check semantic topic distribution for the representative 20k sample.")
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted_communitynotes_2026-04-07"))
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=Path("analysis/representative_20k_sample_20260511/sample_20k_notes.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/representative_20k_sample_20260511/semantic_topic_check"),
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def write_progress(outdir: Path, stage: str, done: int, total: int, detail: str = "") -> None:
    pct = 100.0 * done / total if total else 0.0
    width = 30
    filled = int(round(width * done / total)) if total else 0
    doc = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "done": done,
        "total": total,
        "percent": pct,
        "bar": "[" + "#" * filled + "-" * (width - filled) + "]",
        "detail": detail,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "semantic_topic_progress.log").open("a", encoding="utf-8") as f:
        f.write(
            f"{doc['time_utc']} | {stage:<28} {doc['bar']} {pct:6.2f}% "
            f"({done}/{total}) {detail}\n"
        )
    tmp = outdir / "semantic_topic_progress.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(outdir / "semantic_topic_progress.json")


def assign_topic(text: str) -> tuple[str, dict[str, int]]:
    scores = {
        bucket: int(sum(1 for pattern in patterns if pattern.search(text)))
        for bucket, patterns in COMPILED_PATTERNS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return "other_misc", scores
    return best, scores


def load_notes_status_frame(data_root: Path) -> pd.DataFrame:
    notes_frames: list[pd.DataFrame] = []
    usecols = ["noteId", "createdAtMillis", "tweetId", "classification", "summary"]
    for path in sorted((data_root / "notes").glob("notes-*.tsv")):
        notes_frames.append(
            pd.read_csv(
                path,
                sep="\t",
                usecols=usecols,
                dtype={
                    "noteId": "string",
                    "tweetId": "string",
                    "classification": "string",
                    "summary": "string",
                },
                low_memory=False,
            )
        )
    notes = pd.concat(notes_frames, ignore_index=True)
    status = pd.read_csv(
        data_root / "noteStatusHistory" / "noteStatusHistory-00000.tsv",
        sep="\t",
        usecols=["noteId", "currentStatus"],
        dtype={"noteId": "string", "currentStatus": "string"},
        low_memory=False,
    ).drop_duplicates("noteId", keep="last")
    df = notes.merge(status, on="noteId", how="inner", validate="one_to_one")
    df = df[df["currentStatus"].isin(STATUS_MAP)].copy()
    df = df[df["tweetId"].fillna("").astype(str).str.strip().ne("")]
    df = df[df["summary"].fillna("").astype(str).str.strip().ne("")]
    created = pd.to_datetime(pd.to_numeric(df["createdAtMillis"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    df["year"] = created.dt.year
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df["status_label"] = df["currentStatus"].map(STATUS_MAP)
    return df


def add_semantic_topics(df: pd.DataFrame, outdir: Path, stage: str) -> pd.DataFrame:
    write_progress(outdir, stage, 0, len(TOPIC_BUCKETS) + 1, f"rows={len(df):,}")
    text = (
        df["classification"].fillna("").astype(str)
        + " "
        + df["summary"].fillna("").astype(str)
    ).str.strip()
    score_df = pd.DataFrame(index=df.index)
    for idx, (bucket, pattern_union) in enumerate(TOPIC_REGEX_UNIONS.items(), start=1):
        score_df[bucket] = text.str.contains(pattern_union, case=False, regex=True, na=False).astype(np.int16)
        write_progress(outdir, stage, idx, len(TOPIC_BUCKETS) + 1, bucket)
    max_scores = score_df.max(axis=1)
    best_buckets = score_df.idxmax(axis=1)
    buckets = best_buckets.where(max_scores > 0, other="other_misc")
    scores_json = score_df.to_json(orient="records", lines=True).splitlines()
    write_progress(outdir, stage, len(TOPIC_BUCKETS) + 1, len(TOPIC_BUCKETS) + 1, "done")
    out = df.copy()
    out["semantic_topic"] = buckets.astype("string")
    out["semantic_topic_scores_json"] = scores_json
    return out


def distribution_table(population: pd.DataFrame, sample: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    pop = population.groupby(columns, dropna=False).size().rename("population_count").reset_index()
    sam = sample.groupby(columns, dropna=False).size().rename("sample_count").reset_index()
    out = pop.merge(sam, on=columns, how="outer").fillna({"population_count": 0, "sample_count": 0})
    out["population_count"] = out["population_count"].astype(int)
    out["sample_count"] = out["sample_count"].astype(int)
    pop_total = max(int(out["population_count"].sum()), 1)
    sample_total = max(int(out["sample_count"].sum()), 1)
    out["population_prop"] = out["population_count"] / pop_total
    out["sample_prop"] = out["sample_count"] / sample_total
    out["abs_prop_diff"] = (out["population_prop"] - out["sample_prop"]).abs()
    out.insert(0, "distribution", name)
    return out


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    write_progress(args.outdir, "load population", 0, 1, "start")
    population = load_notes_status_frame(args.data_root)
    write_progress(args.outdir, "load population", 1, 1, f"rows={len(population):,}")
    population = add_semantic_topics(population, args.outdir, "tag population topics")
    write_progress(args.outdir, "load sample", 0, 1, "start")
    sample = pd.read_csv(args.sample_csv, dtype={"noteId": "string"}, low_memory=False)
    write_progress(args.outdir, "load sample", 1, 1, f"rows={len(sample):,}")
    sample = add_semantic_topics(sample, args.outdir, "tag sample topics")

    write_progress(args.outdir, "compute distributions", 0, 4, "start")
    reports = [
        distribution_table(population, sample, ["semantic_topic"], "semantic_topic"),
    ]
    write_progress(args.outdir, "compute distributions", 1, 4, "semantic_topic")
    reports.extend([
        distribution_table(population, sample, ["year", "semantic_topic"], "year_semantic_topic"),
    ])
    write_progress(args.outdir, "compute distributions", 2, 4, "year_semantic_topic")
    reports.extend([
        distribution_table(population, sample, ["status_label", "semantic_topic"], "status_semantic_topic"),
    ])
    write_progress(args.outdir, "compute distributions", 3, 4, "status_semantic_topic")
    reports.extend([
        distribution_table(population, sample, ["year", "status_label", "semantic_topic"], "year_status_semantic_topic"),
    ])
    write_progress(args.outdir, "compute distributions", 4, 4, "done")
    distribution = pd.concat(reports, ignore_index=True)
    topic_summary = reports[0].sort_values("population_prop", ascending=False)

    sample[
        [
            "sample_id",
            "noteId",
            "tweetId",
            "year",
            "status_label",
            "semantic_topic",
            "semantic_topic_scores_json",
            "summary",
        ]
    ].to_csv(args.outdir / "sample_20k_with_semantic_topic.csv", index=False, encoding="utf-8-sig")
    topic_summary.to_csv(args.outdir / "semantic_topic_distribution.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(args.outdir / "semantic_topic_distribution_check.csv", index=False, encoding="utf-8-sig")

    summary = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "population_notes": int(len(population)),
        "sample_notes": int(len(sample)),
        "semantic_topic_rule": "keyword rules over classification + note summary; post text not yet included",
        "max_abs_semantic_topic_prop_diff": float(
            distribution.loc[distribution["distribution"].eq("semantic_topic"), "abs_prop_diff"].max()
        ),
        "max_abs_year_semantic_topic_prop_diff": float(
            distribution.loc[distribution["distribution"].eq("year_semantic_topic"), "abs_prop_diff"].max()
        ),
        "max_abs_status_semantic_topic_prop_diff": float(
            distribution.loc[distribution["distribution"].eq("status_semantic_topic"), "abs_prop_diff"].max()
        ),
        "max_abs_joint_prop_diff": float(
            distribution.loc[distribution["distribution"].eq("year_status_semantic_topic"), "abs_prop_diff"].max()
        ),
        "outputs": [
            "sample_20k_with_semantic_topic.csv",
            "semantic_topic_distribution.csv",
            "semantic_topic_distribution_check.csv",
            "semantic_topic_check_summary.json",
        ],
    }
    (args.outdir / "semantic_topic_check_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_progress(args.outdir, "complete", 1, 1, "semantic topic check complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
