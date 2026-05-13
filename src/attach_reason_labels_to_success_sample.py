#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HELPFUL_REASON_LABELS = [
    "helpfulAddressesClaim",
    "helpfulClear",
    "helpfulEmpathetic",
    "helpfulGoodSources",
    "helpfulImportantContext",
    "helpfulInformative",
    "helpfulUnbiasedLanguage",
    "helpfulUniqueContext",
]

NOT_HELPFUL_REASON_LABELS = [
    "notHelpfulArgumentativeOrBiased",
    "notHelpfulHardToUnderstand",
    "notHelpfulIncorrect",
    "notHelpfulIrrelevantSources",
    "notHelpfulMissingKeyPoints",
    "notHelpfulNoteNotNeeded",
    "notHelpfulOffTopic",
    "notHelpfulOpinionSpeculationOrBias",
    "notHelpfulSourcesMissingOrUnreliable",
    "notHelpfulSpamHarassmentOrAbuse",
]

REASON_LABELS = HELPFUL_REASON_LABELS + NOT_HELPFUL_REASON_LABELS

STATUS_TO_THREE_CLASS_ID = {
    "CURRENTLY_RATED_NOT_HELPFUL": 0,
    "NEEDS_MORE_RATINGS": 1,
    "CURRENTLY_RATED_HELPFUL": 2,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * min(done, total) / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class ProgressReporter:
    def __init__(self, outdir: Path) -> None:
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.progress_json = outdir / "reason_label_progress.json"
        self.progress_log = outdir / "reason_label_progress.log"

    def update(self, stage: str, done: int, total: int, detail: str = "", extra: dict | None = None) -> None:
        pct = 100.0 * done / total if total else 0.0
        payload = {
            "time_utc": utc_now(),
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "percent": pct,
            "bar": progress_bar(done, total),
            "detail": detail,
        }
        if extra:
            payload.update(extra)
        line = f"{payload['time_utc']} | {stage:<28} {payload['bar']} {pct:6.2f}% ({done}/{total}) {detail}"
        with self.progress_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        tmp = self.progress_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.progress_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=Path("analysis/combined_success_sample_20260511/combined_success_notes_with_posts.csv"),
    )
    parser.add_argument(
        "--ratings-dir",
        type=Path,
        default=Path("data/extracted_communitynotes_2026-04-07/noteRatings"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/combined_success_sample_20260511/reason_labels"),
    )
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-share", type=float, default=0.0)
    parser.add_argument("--max-reasons", type=int, default=3)
    return parser.parse_args()


def numeric_reason_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(np.int64)
    return out


def choose_reasons(
    counts: dict[str, int],
    relevant_rating_count: int,
    min_votes: int,
    min_share: float,
    max_reasons: int,
) -> list[str]:
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], -(item[1] / relevant_rating_count if relevant_rating_count else 0.0), item[0]),
    )
    selected = [
        label
        for label, count in ranked
        if count >= min_votes and (relevant_rating_count == 0 or count / relevant_rating_count >= min_share)
    ]
    if max_reasons > 0:
        selected = selected[:max_reasons]
    if selected:
        return selected
    if ranked:
        return [ranked[0][0]]
    return []


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(args.outdir)

    progress.update("load sample", 0, 1, str(args.sample_csv))
    sample = pd.read_csv(args.sample_csv, dtype={"noteId": str, "tweetId": str}, low_memory=False)
    if "currentStatus" not in sample.columns or "noteId" not in sample.columns:
        raise ValueError("sample CSV must contain noteId and currentStatus columns")
    sample["noteId"] = sample["noteId"].astype(str)
    sample["currentStatus"] = sample["currentStatus"].astype(str)
    sample = sample[sample["currentStatus"].isin(STATUS_TO_THREE_CLASS_ID)].copy()
    sample = sample.drop_duplicates("noteId", keep="first").reset_index(drop=True)
    sample["helpfulness_label_3way"] = sample["currentStatus"].map(
        {
            "CURRENTLY_RATED_HELPFUL": "Helpful",
            "NEEDS_MORE_RATINGS": "Need More Ratings",
            "CURRENTLY_RATED_NOT_HELPFUL": "Not Helpful",
        }
    )
    sample["helpfulness_class_id_3way"] = sample["currentStatus"].map(STATUS_TO_THREE_CLASS_ID).astype(int)
    sample["helpfulness_binary_label"] = sample["currentStatus"].map(
        {
            "CURRENTLY_RATED_NOT_HELPFUL": 0,
            "CURRENTLY_RATED_HELPFUL": 1,
        }
    )
    progress.update("load sample", 1, 1, f"notes={len(sample):,}")

    note_ids = set(sample["noteId"])
    note_to_idx = {note_id: i for i, note_id in enumerate(sample["noteId"].tolist())}
    n = len(sample)
    helpful_counts = np.zeros((n, len(HELPFUL_REASON_LABELS)), dtype=np.int64)
    not_helpful_counts = np.zeros((n, len(NOT_HELPFUL_REASON_LABELS)), dtype=np.int64)
    helpful_vote_counts = np.zeros(n, dtype=np.int64)
    not_helpful_vote_counts = np.zeros(n, dtype=np.int64)

    usecols = ["noteId", "helpful", "notHelpful", "helpfulnessLevel"] + REASON_LABELS
    ratings_files = sorted(args.ratings_dir.glob("ratings-*.tsv"))
    if not ratings_files:
        raise FileNotFoundError(f"No ratings-*.tsv found under {args.ratings_dir}")

    total_matched_rows = 0
    total_chunks = 0
    progress.update("scan ratings", 0, len(ratings_files), "start")
    for file_idx, ratings_file in enumerate(ratings_files, start=1):
        file_matched_rows = 0
        file_chunks = 0
        for chunk in pd.read_csv(
            ratings_file,
            sep="\t",
            usecols=lambda col: col in usecols,
            dtype={"noteId": str},
            chunksize=args.chunk_size,
            low_memory=False,
        ):
            file_chunks += 1
            total_chunks += 1
            chunk["noteId"] = chunk["noteId"].astype(str)
            chunk = chunk[chunk["noteId"].isin(note_ids)]
            matched = len(chunk)
            if matched == 0:
                if file_chunks == 1 or file_chunks % 25 == 0:
                    progress.update(
                        "scan ratings",
                        file_idx - 1,
                        len(ratings_files),
                        f"{ratings_file.name} chunk={file_chunks} matched=0",
                        {"total_chunks": total_chunks, "total_matched_rows": int(total_matched_rows)},
                    )
                continue
            file_matched_rows += matched
            total_matched_rows += matched

            level = chunk["helpfulnessLevel"].fillna("").astype(str)
            helpful_flag = pd.to_numeric(chunk.get("helpful", 0), errors="coerce").fillna(0).astype(int)
            not_helpful_flag = pd.to_numeric(chunk.get("notHelpful", 0), errors="coerce").fillna(0).astype(int)

            helpful_mask = level.isin(["HELPFUL", "SOMEWHAT_HELPFUL"]) | (helpful_flag == 1)
            helpful_rows = chunk.loc[helpful_mask, ["noteId"] + HELPFUL_REASON_LABELS]
            if not helpful_rows.empty:
                helpful_numeric = numeric_reason_frame(helpful_rows, HELPFUL_REASON_LABELS)
                helpful_sums = helpful_numeric.groupby(helpful_rows["noteId"]).sum()
                helpful_sizes = helpful_rows.groupby("noteId").size()
                for note_id, row in helpful_sums.iterrows():
                    idx = note_to_idx.get(str(note_id))
                    if idx is not None:
                        helpful_counts[idx, :] += row.to_numpy(dtype=np.int64)
                        helpful_vote_counts[idx] += int(helpful_sizes.loc[note_id])

            not_helpful_mask = (level == "NOT_HELPFUL") | (not_helpful_flag == 1)
            not_helpful_rows = chunk.loc[not_helpful_mask, ["noteId"] + NOT_HELPFUL_REASON_LABELS]
            if not not_helpful_rows.empty:
                not_helpful_numeric = numeric_reason_frame(not_helpful_rows, NOT_HELPFUL_REASON_LABELS)
                not_helpful_sums = not_helpful_numeric.groupby(not_helpful_rows["noteId"]).sum()
                not_helpful_sizes = not_helpful_rows.groupby("noteId").size()
                for note_id, row in not_helpful_sums.iterrows():
                    idx = note_to_idx.get(str(note_id))
                    if idx is not None:
                        not_helpful_counts[idx, :] += row.to_numpy(dtype=np.int64)
                        not_helpful_vote_counts[idx] += int(not_helpful_sizes.loc[note_id])

            if file_chunks == 1 or file_chunks % 10 == 0:
                progress.update(
                    "scan ratings",
                    file_idx - 1,
                    len(ratings_files),
                    f"{ratings_file.name} chunk={file_chunks} matched_file={file_matched_rows:,}",
                    {"total_chunks": total_chunks, "total_matched_rows": int(total_matched_rows)},
                )

        progress.update(
            "scan ratings",
            file_idx,
            len(ratings_files),
            f"{ratings_file.name} done matched_file={file_matched_rows:,}",
            {"total_chunks": total_chunks, "total_matched_rows": int(total_matched_rows)},
        )

    progress.update("assemble labels", 0, n, "start")
    reason_rows: list[dict[str, object]] = []
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        note_id = str(row.noteId)
        status = str(row.currentStatus)
        helpful_dict = {
            label: int(value)
            for label, value in zip(HELPFUL_REASON_LABELS, helpful_counts[i - 1, :])
            if int(value) > 0
        }
        not_helpful_dict = {
            label: int(value)
            for label, value in zip(NOT_HELPFUL_REASON_LABELS, not_helpful_counts[i - 1, :])
            if int(value) > 0
        }
        if status == "CURRENTLY_RATED_HELPFUL":
            selected_counts = helpful_dict
            selected_rating_count = int(helpful_vote_counts[i - 1])
            reason_source = "helpful_side_status_aligned"
        elif status == "CURRENTLY_RATED_NOT_HELPFUL":
            selected_counts = not_helpful_dict
            selected_rating_count = int(not_helpful_vote_counts[i - 1])
            reason_source = "not_helpful_side_status_aligned"
        else:
            selected_counts = {**helpful_dict, **not_helpful_dict}
            selected_rating_count = int(helpful_vote_counts[i - 1] + not_helpful_vote_counts[i - 1])
            reason_source = "both_sides_need_more_ratings"

        reasons = choose_reasons(
            selected_counts,
            relevant_rating_count=selected_rating_count,
            min_votes=args.min_votes,
            min_share=args.min_share,
            max_reasons=args.max_reasons,
        )
        reason_set = set(reasons)
        out: dict[str, object] = {
            "noteId": note_id,
            "reasons": ";".join(reasons),
            "reason_count": len(reasons),
            "reason_source": reason_source,
            "relevant_reason_rating_count": selected_rating_count,
            "helpful_reason_rating_count": int(helpful_vote_counts[i - 1]),
            "not_helpful_reason_rating_count": int(not_helpful_vote_counts[i - 1]),
            "raw_helpful_reason_counts_json": json.dumps(helpful_dict, ensure_ascii=False, sort_keys=True),
            "raw_not_helpful_reason_counts_json": json.dumps(not_helpful_dict, ensure_ascii=False, sort_keys=True),
            "raw_selected_reason_counts_json": json.dumps(selected_counts, ensure_ascii=False, sort_keys=True),
        }
        for label in REASON_LABELS:
            out[f"reason__{label}"] = 1 if label in reason_set else 0
        for label, value in zip(HELPFUL_REASON_LABELS, helpful_counts[i - 1, :]):
            out[f"helpful_reason_count__{label}"] = int(value)
        for label, value in zip(NOT_HELPFUL_REASON_LABELS, not_helpful_counts[i - 1, :]):
            out[f"not_helpful_reason_count__{label}"] = int(value)
        reason_rows.append(out)
        if i == 1 or i == n or i % 5000 == 0:
            progress.update("assemble labels", i, n, f"rows={i:,}")

    reason_df = pd.DataFrame(reason_rows)
    merged = sample.merge(reason_df, on="noteId", how="left", validate="one_to_one")

    reason_cols = [f"reason__{label}" for label in REASON_LABELS]
    summary_rows = []
    for label in REASON_LABELS:
        summary_rows.append(
            {
                "reason_label": label,
                "positive_notes": int(merged[f"reason__{label}"].sum()),
                "positive_share": float(merged[f"reason__{label}"].mean()),
            }
        )
    reason_summary = pd.DataFrame(summary_rows).sort_values("positive_notes", ascending=False)

    output_csv = args.outdir / "combined_success_notes_with_posts_and_reasons.csv"
    reasons_only_csv = args.outdir / "reason_labels_for_combined_success_sample.csv"
    reason_summary_csv = args.outdir / "reason_label_summary.csv"
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    reason_df.to_csv(reasons_only_csv, index=False, encoding="utf-8-sig")
    reason_summary.to_csv(reason_summary_csv, index=False, encoding="utf-8-sig")

    coverage_by_status = (
        merged.groupby("helpfulness_label_3way", dropna=False)
        .agg(
            notes=("noteId", "size"),
            notes_with_any_reason=("reason_count", lambda s: int((s > 0).sum())),
            mean_reason_count=("reason_count", "mean"),
            mean_relevant_reason_rating_count=("relevant_reason_rating_count", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "time_utc": utc_now(),
        "sample_csv": str(args.sample_csv),
        "ratings_dir": str(args.ratings_dir),
        "output_csv": str(output_csv),
        "reasons_only_csv": str(reasons_only_csv),
        "reason_summary_csv": str(reason_summary_csv),
        "notes": int(len(merged)),
        "notes_with_any_reason": int((merged["reason_count"] > 0).sum()),
        "notes_without_reason": int((merged["reason_count"] == 0).sum()),
        "mean_reason_count": float(merged["reason_count"].mean()),
        "mean_relevant_reason_rating_count": float(merged["relevant_reason_rating_count"].mean()),
        "helpfulness_label_distribution": merged["helpfulness_label_3way"].value_counts(dropna=False).to_dict(),
        "coverage_by_status": coverage_by_status,
        "reason_labels": REASON_LABELS,
        "min_votes": args.min_votes,
        "min_share": args.min_share,
        "max_reasons": args.max_reasons,
        "total_matched_rating_rows": int(total_matched_rows),
        "total_rating_chunks": int(total_chunks),
        "aggregation_rule": (
            "Helpful notes use helpful-side reason checkboxes; Not Helpful notes use not-helpful-side "
            "reason checkboxes; Need More Ratings notes use both sides because there is no resolved side."
        ),
    }
    (args.outdir / "reason_label_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    progress.update("complete", 1, 1, "reason labels attached")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
