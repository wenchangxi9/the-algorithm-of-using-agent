#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


STATUS_TO_LABEL = {
    "CURRENTLY_RATED_NOT_HELPFUL": 0,
    "NEEDS_MORE_RATINGS": 1,
    "CURRENTLY_RATED_HELPFUL": 2,
}

NOTE_STATUS_TO_TEXT = {
    0: "NOT_HELPFUL",
    1: "NEEDS_MORE_RATINGS",
    2: "HELPFUL",
}

RAW_RATING_TO_SCORE = {
    "NOT_HELPFUL": 0.0,
    "SOMEWHAT_HELPFUL": 0.5,
    "HELPFUL": 1.0,
}

OFFICIAL_HELPFUL_REASON_KEYS = [
    "helpfulClear",
    "helpfulGoodSources",
    "helpfulAddressesClaim",
    "helpfulImportantContext",
    "helpfulUnbiasedLanguage",
]

OFFICIAL_NOT_HELPFUL_REASON_KEYS = [
    "notHelpfulIncorrect",
    "notHelpfulSourcesMissingOrUnreliable",
    "notHelpfulMissingKeyPoints",
    "notHelpfulHardToUnderstand",
    "notHelpfulArgumentativeOrBiased",
    "notHelpfulIrrelevantSources",
    "notHelpfulOpinionSpeculation",
    "notHelpfulNoteNotNeeded",
]

REQUIRED_OUTPUT_KEYS = [
    "helpfulnessLevel",
    "agree",
    "disagree",
    *OFFICIAL_HELPFUL_REASON_KEYS,
    *OFFICIAL_NOT_HELPFUL_REASON_KEYS,
    "confidence",
    "changes_reader_understanding",
    "rationale",
]


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
        self.progress_json = outdir / "progress.json"
        self.progress_log = outdir / "progress.log"
        self.lock = threading.Lock()

    def update(self, stage: str, done: int, total: int, detail: str = "", extra: dict[str, Any] | None = None) -> None:
        pct = 100.0 * done / total if total else 0.0
        doc: dict[str, Any] = {
            "time_utc": utc_now(),
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "percent": pct,
            "bar": progress_bar(done, total),
            "detail": detail,
        }
        if extra:
            doc.update(extra)
        line = f"{doc['time_utc']} | {stage:<28} {doc['bar']} {pct:6.2f}% ({done}/{total}) {detail}"
        with self.lock:
            with self.progress_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            tmp = self.progress_json.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.progress_json)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, item: dict[str, Any]) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 16-agent official raw-rating Community Notes pilot.")
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=Path("analysis/combined_success_sample_20260511/combined_success_notes_with_posts.csv"),
    )
    parser.add_argument(
        "--cluster-summary-csv",
        type=Path,
        default=Path(
            "analysis/official_mfcore_rater_clustering_20260510_201855/"
            "k_search_2_32_step1/cluster_summary_k16.csv"
        ),
    )
    parser.add_argument("--outdir", type=Path, default=Path("analysis/llm_16agent_rawrating_pilot_20_20260512"))
    parser.add_argument("--api-key-file", type=Path, default=Path("secrets/openai_api_key.txt"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.gpt.ge/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"))
    parser.add_argument("--max-notes", type=int, default=20)
    parser.add_argument(
        "--selection-mode",
        choices=["balanced_status", "sample_order", "representative"],
        default="balanced_status",
    )
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    return parser.parse_args()


def read_api_key(args: argparse.Namespace) -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if args.api_key_file.exists():
        key = args.api_key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError(
        "No API key found. Set OPENAI_API_KEY or create secrets/openai_api_key.txt on the server."
    )


def coerce_score(value: object) -> int:
    try:
        parsed = int(round(float(value)))
    except Exception:
        return -1
    return max(0, min(parsed, 100))


def coerce_yes_no(value: object) -> int:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"YES", "Y", "TRUE", "1"}:
        return 1
    if text in {"NO", "N", "FALSE", "0"}:
        return 0
    return -1


def extract_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        trial_candidates = [candidate]
        if candidate.startswith('"') and candidate.endswith('"'):
            trial_candidates.append(candidate[1:-1])
        if '""' in candidate and '\\"' not in candidate:
            trial_candidates.append(candidate.replace('""', '"'))
        for trial in trial_candidates:
            try:
                payload = json.loads(trial)
                if isinstance(payload, dict):
                    return payload
                if isinstance(payload, str):
                    nested = json.loads(payload)
                    if isinstance(nested, dict):
                        return nested
            except Exception:
                continue
    raise ValueError(f"No JSON object found in response: {text[:300]}")


def normalize_rating(value: object, raw_text: str) -> tuple[str, float]:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    raw_upper = raw_text.upper()
    if text in {"HELPFUL", "CURRENTLY_RATED_HELPFUL"}:
        return "HELPFUL", 1.0
    if text in {"NOT_HELPFUL", "NOTHELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"}:
        return "NOT_HELPFUL", 0.0
    if text in {"SOMEWHAT_HELPFUL", "SOMEWHATHELPFUL", "PARTLY_HELPFUL", "PARTIALLY_HELPFUL"}:
        return "SOMEWHAT_HELPFUL", 0.5
    if "SOMEWHAT_HELPFUL" in raw_upper or "SOMEWHAT HELPFUL" in raw_upper:
        return "SOMEWHAT_HELPFUL", 0.5
    if "NOT_HELPFUL" in raw_upper or "NOT HELPFUL" in raw_upper:
        return "NOT_HELPFUL", 0.0
    if "HELPFUL" in raw_upper:
        return "HELPFUL", 1.0
    return "UNKNOWN", -1.0


def parse_response(text: str) -> dict[str, Any]:
    payload = extract_json_blob(text)
    rating, score = normalize_rating(payload.get("helpfulnessLevel", payload.get("rating")), text)
    parsed = {
        "helpfulnessLevel": rating,
        "parsed_rating": rating,
        "predicted_rating_score": score,
        "agree": coerce_yes_no(payload.get("agree")),
        "disagree": coerce_yes_no(payload.get("disagree")),
        "helpfulClear": coerce_yes_no(payload.get("helpfulClear")),
        "helpfulGoodSources": coerce_yes_no(payload.get("helpfulGoodSources")),
        "helpfulAddressesClaim": coerce_yes_no(payload.get("helpfulAddressesClaim")),
        "helpfulImportantContext": coerce_yes_no(payload.get("helpfulImportantContext")),
        "helpfulUnbiasedLanguage": coerce_yes_no(payload.get("helpfulUnbiasedLanguage")),
        "notHelpfulIncorrect": coerce_yes_no(payload.get("notHelpfulIncorrect")),
        "notHelpfulSourcesMissingOrUnreliable": coerce_yes_no(payload.get("notHelpfulSourcesMissingOrUnreliable")),
        "notHelpfulMissingKeyPoints": coerce_yes_no(payload.get("notHelpfulMissingKeyPoints")),
        "notHelpfulHardToUnderstand": coerce_yes_no(payload.get("notHelpfulHardToUnderstand")),
        "notHelpfulArgumentativeOrBiased": coerce_yes_no(payload.get("notHelpfulArgumentativeOrBiased")),
        "notHelpfulIrrelevantSources": coerce_yes_no(payload.get("notHelpfulIrrelevantSources")),
        "notHelpfulOpinionSpeculation": coerce_yes_no(payload.get("notHelpfulOpinionSpeculation")),
        "notHelpfulNoteNotNeeded": coerce_yes_no(payload.get("notHelpfulNoteNotNeeded")),
        "confidence": coerce_score(payload.get("confidence")),
        "changes_reader_understanding": coerce_score(payload.get("changes_reader_understanding")),
        "rationale": " ".join(str(payload.get("rationale", "")).split())[:500],
    }
    return parsed


def band(value: float, series: pd.Series, higher: str = "high", lower: str = "low") -> str:
    if series.empty:
        return "middle"
    rank = float((series <= value).mean())
    if rank >= 0.80:
        return f"very {higher}"
    if rank >= 0.60:
        return f"moderately {higher}"
    if rank <= 0.20:
        return f"very {lower}"
    if rank <= 0.40:
        return f"moderately {lower}"
    return "middle"


def get_numeric_value(row: pd.Series, names: list[str], default: float = np.nan) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                continue
    return float(default)


def get_numeric_series(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            series = pd.to_numeric(df[name], errors="coerce")
            if series.notna().any():
                return series
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def describe_band(value: float, series: pd.Series, higher: str, lower: str) -> str:
    if pd.isna(value):
        return "middle"
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return "middle"
    return band(float(value), valid, higher=higher, lower=lower)


def helpful_tendency_label(share_helpful: float, share_not_helpful: float) -> str:
    if pd.isna(share_helpful) or pd.isna(share_not_helpful):
        return "is broadly balanced in positive versus negative judgments"
    margin = share_helpful - share_not_helpful
    if margin >= 0.35:
        return "is strongly tilted toward Helpful judgments"
    if margin >= 0.15:
        return "is mildly tilted toward Helpful judgments"
    if margin <= -0.25:
        return "is strongly tilted toward Not Helpful judgments"
    if margin <= -0.10:
        return "is mildly tilted toward Not Helpful judgments"
    return "is broadly balanced in positive versus negative judgments"


def recent_shift_label(help_shift: float, not_shift: float) -> str:
    if pd.isna(help_shift) or pd.isna(not_shift):
        return "its recent behavior is broadly stable"
    if help_shift >= 0.08:
        return "it has recently become more willing to endorse notes"
    if help_shift <= -0.08:
        return "it has recently become less willing to endorse notes"
    if not_shift >= 0.08:
        return "it has recently become more likely to reject notes"
    if not_shift <= -0.08:
        return "its recent rejection tendency has softened"
    return "its recent behavior is broadly stable"


def authoring_label(crh: float, crnh: float, notes_authored: float) -> str:
    if pd.isna(notes_authored):
        return "author-side behavior is not explicitly measured here"
    if notes_authored < 1:
        return "these users almost never write notes themselves and behave more like pure raters"
    if not pd.isna(crh) and crh >= 0.20:
        return "when they write notes, those notes succeed relatively often"
    if not pd.isna(crnh) and crnh >= 0.15:
        return "they do write notes, but their authored notes are rejected fairly often"
    return "they write some notes, but are not especially strong or especially weak as note authors"


def persona_bias_instruction(
    share_helpful: float,
    share_not_helpful: float,
    strict_band: str,
    evidence_band: str,
) -> str:
    parts: list[str] = []
    margin = 0.0 if pd.isna(share_helpful) or pd.isna(share_not_helpful) else share_helpful - share_not_helpful

    if strict_band in {"very strict", "moderately strict"}:
        parts.append(
            "This cluster uses a fairly high bar for a full HELPFUL rating, but it should not reject a note merely because the note is imperfect."
        )
    else:
        parts.append(
            "This cluster is willing to reward a note that is not perfect if it still materially improves a reader's understanding."
        )

    if margin >= 0.15:
        parts.append("In ambiguous cases, it leans somewhat more toward giving credit to genuinely useful notes.")
    elif margin <= -0.10:
        parts.append("In ambiguous cases, it leans somewhat more skeptical and asks notes to prove their value.")
    else:
        parts.append("In ambiguous cases, it does not strongly lean positive or negative.")

    if evidence_band in {"very evidence-sensitive", "moderately evidence-sensitive"}:
        parts.append("Source quality, direct support, and whether the note truly changes understanding strongly affect its rating.")
    else:
        parts.append("It does not require perfect sourcing, but still expects the note to deliver real contextual value.")

    return " ".join(parts)


def persona_name(row: pd.Series) -> str:
    agree = get_numeric_value(row, ["mean_raterAgreeRatio", "bw_rater_agree_ratio"], 0.5)
    helpfulness = get_numeric_value(row, ["mean_aboveHelpfulnessThreshold", "bw_helpfulness_pass"], 0.5)
    note_score = get_numeric_value(row, ["mean_meanNoteScore", "bw_mean_note_score"], 0.1)
    diff = get_numeric_value(row, ["mean_crhCrnhRatioDifference", "bw_crh_crnh_ratio_difference"], 0.0)
    factor = get_numeric_value(row, ["mean_internalRaterFactor1", "bw_final_rater_factor_1"], 0.0)
    if agree < 0.2:
        return "low-agreement idiosyncratic rater"
    if agree < 0.65:
        return "mixed-agreement boundary-case rater"
    if diff < -1.0 or note_score < 0.02:
        return "skeptical low-score rater"
    if diff > 0.15 or note_score > 0.25:
        return "positive high-context rater"
    if helpfulness >= 0.75 and factor > 0.30:
        return "high-helpfulness positive-axis rater"
    if helpfulness >= 0.75 and factor < -0.30:
        return "high-helpfulness negative-axis rater"
    if helpfulness < 0.25:
        return "strict unresolved-prone rater"
    return "mainstream consensus rater"


def build_personas(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.sort_values("cluster").reset_index(drop=True).copy()
    df["persona_name"] = df.apply(persona_name, axis=1)
    helpful_share_series = get_numeric_series(df, ["share_helpful"])
    not_helpful_share_series = get_numeric_series(df, ["share_not_helpful"])
    somewhat_share_series = get_numeric_series(df, ["share_somewhat_helpful"])
    evidence_series = get_numeric_series(df, ["evidence_focus_rate", "mean_meanNoteScore", "bw_mean_note_score"])
    strict_series = get_numeric_series(df, ["strict_rejection_rate"])
    civility_series = get_numeric_series(df, ["civility_rejection_rate"])
    redundancy_series = get_numeric_series(df, ["redundancy_rejection_rate"])
    burst_series = get_numeric_series(df, ["activity_burstiness"])
    recent_activity_series = get_numeric_series(df, ["recent_90d_share"])
    note_len_series = get_numeric_series(df, ["avg_summary_char_len"])
    prompts = []
    for row in df.itertuples(index=False):
        series = pd.Series(row._asdict())
        cluster = int(series["cluster"])
        if "share" in df.columns and pd.notna(series.get("share")):
            share = float(series["share"])
        elif "users" in df.columns and df["users"].sum() > 0 and pd.notna(series.get("users")):
            share = float(series["users"]) / float(df["users"].sum())
        elif "n_raters" in df.columns and df["n_raters"].sum() > 0 and pd.notna(series.get("n_raters")):
            share = float(series["n_raters"]) / float(df["n_raters"].sum())
        else:
            share = 1.0 / max(len(df), 1)

        agree = get_numeric_value(series, ["mean_raterAgreeRatio", "bw_rater_agree_ratio"], 0.5)
        helpfulness = get_numeric_value(series, ["mean_aboveHelpfulnessThreshold", "bw_helpfulness_pass"], 0.5)
        note_score = get_numeric_value(series, ["mean_meanNoteScore", "bw_mean_note_score"], 0.1)
        diff = get_numeric_value(series, ["mean_crhCrnhRatioDifference", "bw_crh_crnh_ratio_difference"], 0.0)
        factor = get_numeric_value(series, ["mean_internalRaterFactor1", "bw_final_rater_factor_1"], 0.0)
        intercept = get_numeric_value(series, ["mean_internalRaterIntercept", "bw_final_rater_intercept"], 0.0)
        first_factor = get_numeric_value(series, ["mean_internalFirstRoundRaterFactor1", "bw_pre_rater_factor_1"], 0.0)

        share_helpful = get_numeric_value(series, ["share_helpful"], np.nan)
        share_not_helpful = get_numeric_value(series, ["share_not_helpful"], np.nan)
        share_somewhat_helpful = get_numeric_value(series, ["share_somewhat_helpful"], np.nan)
        recent_helpful_shift = get_numeric_value(series, ["recent_helpful_shift"], np.nan)
        recent_not_helpful_shift = get_numeric_value(series, ["recent_not_helpful_shift"], np.nan)
        notes_authored = get_numeric_value(series, ["notes_authored", "avg_notes_authored"], np.nan)
        share_authored_crh = get_numeric_value(series, ["share_authored_crh"], np.nan)
        share_authored_crnh = get_numeric_value(series, ["share_authored_crnh"], np.nan)
        avg_ratings_given = get_numeric_value(series, ["avg_ratings_given", "ratings_given"], np.nan)
        recent_90d_share = get_numeric_value(series, ["recent_90d_share"], np.nan)
        activity_burstiness = get_numeric_value(series, ["activity_burstiness"], np.nan)
        evidence_focus_rate = get_numeric_value(series, ["evidence_focus_rate", "mean_meanNoteScore", "bw_mean_note_score"], note_score)
        strict_rejection_rate = get_numeric_value(series, ["strict_rejection_rate"], 1.0 - helpfulness)
        civility_rejection_rate = get_numeric_value(series, ["civility_rejection_rate"], np.nan)
        redundancy_rejection_rate = get_numeric_value(series, ["redundancy_rejection_rate"], np.nan)
        avg_summary_char_len = get_numeric_value(series, ["avg_summary_char_len"], np.nan)

        evidence_band = describe_band(
            evidence_focus_rate,
            evidence_series,
            higher="evidence-sensitive",
            lower="evidence-tolerant",
        )
        strict_band = describe_band(
            strict_rejection_rate,
            strict_series if strict_series.notna().any() else 1.0 - get_numeric_series(df, ["mean_aboveHelpfulnessThreshold", "bw_helpfulness_pass"]).fillna(0.5),
            higher="strict",
            lower="lenient",
        )
        civility_band = describe_band(
            civility_rejection_rate,
            civility_series,
            higher="civility-sensitive",
            lower="less civility-sensitive",
        )
        redundancy_band = describe_band(
            redundancy_rejection_rate,
            redundancy_series,
            higher="redundancy-sensitive",
            lower="less redundancy-sensitive",
        )
        burst_band = describe_band(
            activity_burstiness,
            burst_series,
            higher="bursty",
            lower="steady",
        )
        recent_band = describe_band(
            recent_90d_share,
            recent_activity_series,
            higher="recently active",
            lower="less recently active",
        )
        note_len_band = describe_band(
            avg_summary_char_len,
            note_len_series,
            higher="long-note-oriented",
            lower="short-note-oriented",
        )

        helpful_label = helpful_tendency_label(share_helpful, share_not_helpful)
        recent_label = recent_shift_label(recent_helpful_shift, recent_not_helpful_shift)
        author_label = authoring_label(share_authored_crh, share_authored_crnh, notes_authored)
        bias_instruction = persona_bias_instruction(
            share_helpful,
            share_not_helpful,
            strict_band,
            evidence_band,
        )
        score_hint = (
            "This cluster tends to reward notes that directly fix the central misleading implication and add substantial context."
            if note_score > 0.20 or diff > 0.10
            else "This cluster is cautious about giving too much credit to notes that are partial, speculative, redundant, or weakly sourced."
            if note_score < 0.05 or diff < -0.50
            else "This cluster is fairly balanced: it rewards useful context, but does not over-credit weak or tangential notes."
        )
        latent_hint = (
            "Its MF position lies on the positive side of the contributor space, so it may be somewhat more receptive to notes aligned with that behavioral region."
            if factor > 0.25
            else "Its MF position lies on the negative side of the contributor space, so it may be somewhat more skeptical of notes aligned with that behavioral region."
            if factor < -0.25
            else "Its MF position lies near the center of the contributor space, so it should avoid extreme judgments unless the case is clear."
        )

        profile_lines = [
            f"- Population share: {share * 100:.2f}% of clustered raters.",
            f"- Persona type: {series['persona_name']}.",
            f"- Agreement profile: mean rater-agreement ratio {agree:.3f}.",
            f"- MF profile: intercept {intercept:.3f}, factor1 {factor:.3f}, first-round factor1 {first_factor:.3f}.",
            f"- Note-score tendency: {note_score:.3f}; CRH-minus-CRNH tendency: {diff:.3f}.",
            f"- Helpfulness-pass tendency: {helpfulness:.3f}.",
        ]

        if not pd.isna(share_helpful) and not pd.isna(share_not_helpful):
            share_text = f"- Historical HELPFUL / NOT_HELPFUL shares: {share_helpful * 100:.1f}% / {share_not_helpful * 100:.1f}%."
            if not pd.isna(share_somewhat_helpful):
                share_text = (
                    f"- Historical HELPFUL / SOMEWHAT_HELPFUL / NOT_HELPFUL shares: "
                    f"{share_helpful * 100:.1f}% / {share_somewhat_helpful * 100:.1f}% / {share_not_helpful * 100:.1f}%. "
                    f"Overall, this cluster {helpful_label}."
                )
            else:
                share_text += f" Overall, this cluster {helpful_label}."
            profile_lines.append(share_text)

        if not pd.isna(avg_ratings_given):
            activity_line = f"- Activity level: on average these users give {avg_ratings_given:.1f} ratings"
            if not pd.isna(recent_90d_share):
                activity_line += f"; recent-90-day share {recent_90d_share * 100:.1f}% ({recent_band})"
            if not pd.isna(activity_burstiness):
                activity_line += f"; temporal pattern is {burst_band}"
            activity_line += "."
            profile_lines.append(activity_line)

        if not pd.isna(evidence_focus_rate):
            profile_lines.append(
                f"- Evidence sensitivity: {evidence_band}. This reflects how much this cluster cares about sourcing, direct support, and whether the note truly improves understanding."
            )

        if not pd.isna(strict_rejection_rate):
            profile_lines.append(
                f"- Rejection strictness: {strict_band}. This reflects how easily this cluster rejects notes for weak evidence, logical gaps, lack of necessity, or overstated claims."
            )

        if not pd.isna(civility_rejection_rate):
            profile_lines.append(f"- Civility sensitivity: {civility_band}.")

        if not pd.isna(redundancy_rejection_rate):
            profile_lines.append(f"- Redundancy sensitivity: {redundancy_band}.")

        if not pd.isna(notes_authored):
            author_line = f"- Author-side profile: they write {notes_authored:.1f} notes on average; {author_label}."
            if not pd.isna(share_authored_crh) and not pd.isna(share_authored_crnh):
                author_line += (
                    f" Their authored-note CRH / CRNH shares are "
                    f"{share_authored_crh * 100:.1f}% / {share_authored_crnh * 100:.1f}%."
                )
            profile_lines.append(author_line)

        if not pd.isna(avg_summary_char_len):
            profile_lines.append(f"- Typical authored-note length preference: {note_len_band}.")

        if not pd.isna(recent_helpful_shift) and not pd.isna(recent_not_helpful_shift):
            profile_lines.append(f"- Recent drift: {recent_label}.")

        profile_lines.append(f"- Edge-case tendency: {bias_instruction}")
        profile_lines.append(f"- Additional interpretation: {score_hint}")
        profile_lines.append(f"- Latent-position interpretation: {latent_hint}")
        profile_block = "\n".join(profile_lines)

        agent_id = f"C{cluster:02d}"
        prompt = f"""
You are simulating a real X Community Notes rater, not a generic assistant and not an average neutral judge.

Your fixed identity is contributor cluster {agent_id}. Stay consistent with this cluster's historical behavior rather than collapsing toward a generic average user.

This cluster's historical behavior profile:
{profile_block}

Task:
Act like one raw Community Notes rater and output one official-style raw helpfulness rating for the NOTE relative to the POST.
Your allowed ratings are exactly:
- HELPFUL
- SOMEWHAT_HELPFUL
- NOT_HELPFUL

Decision guide:
- HELPFUL: the note directly addresses the post's central claim or implication and gives accurate, relevant context that would materially improve a reader's understanding. It does not need to be perfect; if the main correction/context is sound and important, rate it HELPFUL.
- SOMEWHAT_HELPFUL: use this only for genuinely mixed or borderline cases: the note contains some useful context, but a major gap prevents a clear HELPFUL rating, or the helpful and not-helpful evidence is closely balanced. Do not use SOMEWHAT_HELPFUL as a safe default when the note is clearly useful or clearly unhelpful.
- NOT_HELPFUL: the note is incorrect, unsupported, biased, off-topic, too minor, tangential, unnecessary, or fails to address the central claim. If the note's main contribution would not meaningfully change reader understanding, rate it NOT_HELPFUL rather than SOMEWHAT_HELPFUL.

Official Community Notes rating criteria to apply:
Helpful-positive reasons:
- Clear and/or well-written: the note is understandable, specific, and not confusing.
- Cites high-quality sources: the note relies on credible, relevant sources when factual support is needed.
- Directly addresses the post's claim: the note responds to the central claim or implication, not a side issue.
- Provides important context: the note adds context that would change how readers interpret the post.
- Neutral or unbiased language: the note is factual and non-argumentative.

Not-helpful reasons:
- Incorrect information: the note itself makes a false or misleading claim.
- Sources missing or unreliable: important factual claims lack credible support.
- Sources do not support the note: cited sources are irrelevant, weak, or do not actually prove the note's claim.
- Misses key points or irrelevant: the note does not address the central issue in the post.
- Hard to understand: the note is unclear enough that readers would not benefit from it.
- Argumentative or biased language: the note reads as opinion, attack, or persuasion rather than context.
- Opinion or speculation: the note relies on interpretation or speculation rather than verifiable context.
- Note not needed on this post: the post is not materially misleading or the note adds no necessary context.
- Spam, harassment, or abuse: the note is abusive, promotional, or otherwise inappropriate.

Practical calibration:
- A note can be HELPFUL without satisfying every helpful-positive reason; the decisive question is whether it accurately and materially improves understanding of the central claim.
- A note should be NOT_HELPFUL if it has a fatal flaw: incorrect content, unsupported central claim, irrelevant evidence, missing the central issue, or unnecessary/tangential context.
- Use SOMEWHAT_HELPFUL only when the note has real partial value and no fatal flaw, but still has a substantial limitation that prevents a full HELPFUL rating.

Failure-avoidance guardrails learned from Community Notes edge cases:
- Do not require a note to address every side detail. If it directly corrects one central, material claim or implication in the post, and that correction is well supported, it can be HELPFUL.
- For manipulated media, old media, misidentified locations, people, objects, signs, or screenshots, a note that identifies the real source/context of the media is usually HELPFUL when the identification is specific and supported.
- For short, sarcastic, or elliptical posts, infer the central implication from the available post text and media description. Do not reject a note merely because the post is short if the note addresses the implicit claim.
- Do not over-credit a related statistic, semantic nitpick, partisan counterpoint, or whataboutism. If the note is true but does not change the reader's understanding of the post's central claim, rate it NOT_HELPFUL.
- Be careful with source quality. A bare social-media link, an interested party's denial, or a disputed official statement is not enough for HELPFUL if the note uses it to conclusively disprove a contested claim.
- If a post says that something was reported, alleged, or planned, a denial alone does not necessarily make the post false. The note must show that the report itself is wrong or materially misleading.
- If a note overstates its correction beyond what its source proves, treat that as a not-helpful flaw even when the note sounds like a fact-check.
- If the note only attacks the author, changes the subject, or supplies background that is mainly reputational rather than explanatory for the post's claim, rate it NOT_HELPFUL.

Decision procedure:
1. First identify the post's central claim, misleading implication, or reason a note may be needed.
2. Then identify the note's main contribution.
3. Check whether the note directly targets the central claim, including implicit claims from attached media, sarcasm, or short posts.
4. Check for fatal not-helpful flaws: incorrect information, unsupported central factual claim, irrelevant sources, missing the central claim, biased/speculative framing, note not needed, or overclaiming beyond the cited source.
5. Ask whether the note's contribution clearly improves reader understanding of the central claim:
   - If yes, choose HELPFUL.
   - If no, choose NOT_HELPFUL.
   - Choose SOMEWHAT_HELPFUL only when the answer is genuinely partial or balanced.
6. Avoid excessive middle ratings. Community Notes raters can be decisive: a useful but imperfect note is often HELPFUL, and a weak or unnecessary note is often NOT_HELPFUL.

How this cluster should judge:
1. Rate the NOTE itself, not whether you politically agree with the post or the note.
2. Read both the POST and the NOTE, and judge whether the note meaningfully improves a typical reader's understanding of the post.
3. Focus especially on whether the note addresses the core misleading implication, whether it would change reader understanding, and whether it is necessary enough to deserve note-level attention.
4. Use HELPFUL when the note materially improves understanding, even if it is concise, imperfectly worded, or not exhaustive.
5. Use HELPFUL when the note provides a direct factual correction, necessary context, or source-backed clarification of the central claim, unless a clear defect makes it unreliable.
6. Use NOT_HELPFUL when the note misses the core claim, adds only minor or tangential detail, is poorly supported, is itself inaccurate, is argumentative, or fails to provide meaningful contextual value.
7. Use NOT_HELPFUL when the note merely states a related fact but does not explain why the post is misleading or why the reader's interpretation should change.
8. Use SOMEWHAT_HELPFUL sparingly for partial but real value: for example, a note with relevant context but incomplete sourcing, incomplete claim coverage, or a correction that is useful but not enough to resolve the post.
9. In ambiguous cases, do not revert to an average-user judgment; stay faithful to this cluster's own rating habits.

Rating calibration:
- Do not penalize a note into SOMEWHAT_HELPFUL just because it is not comprehensive. If it addresses the central issue and would materially help readers, choose HELPFUL.
- Do not reward a note with SOMEWHAT_HELPFUL just because it sounds plausible. If it lacks support, misses the central issue, or is unnecessary, choose NOT_HELPFUL.
- Treat SOMEWHAT_HELPFUL as a narrow middle category, not as uncertainty. If uncertainty comes from your lack of external knowledge but the note itself clearly provides or lacks useful context, still choose HELPFUL or NOT_HELPFUL accordingly.

When producing your JSON:
- Set helpfulnessLevel to exactly one of HELPFUL, SOMEWHAT_HELPFUL, or NOT_HELPFUL.
- Set agree=1 and disagree=0 if you agree with the note's conclusion; set agree=0 and disagree=1 if you disagree; if genuinely unclear, you may set both to 0.
- For helpful reason fields, mark 1 only when that reason positively supports the note.
- For not-helpful reason fields, mark 1 only when that problem clearly applies.
- Multiple reason fields may be 1 at the same time.
- confidence is our extra research field from 0 to 100.
- changes_reader_understanding is our extra research field from 0 to 100.

Return exactly one JSON object and no other text:
{{"helpfulnessLevel":"HELPFUL or SOMEWHAT_HELPFUL or NOT_HELPFUL","agree":0 or 1,"disagree":0 or 1,"helpfulClear":0 or 1,"helpfulGoodSources":0 or 1,"helpfulAddressesClaim":0 or 1,"helpfulImportantContext":0 or 1,"helpfulUnbiasedLanguage":0 or 1,"notHelpfulIncorrect":0 or 1,"notHelpfulSourcesMissingOrUnreliable":0 or 1,"notHelpfulMissingKeyPoints":0 or 1,"notHelpfulHardToUnderstand":0 or 1,"notHelpfulArgumentativeOrBiased":0 or 1,"notHelpfulIrrelevantSources":0 or 1,"notHelpfulOpinionSpeculation":0 or 1,"notHelpfulNoteNotNeeded":0 or 1,"confidence":0-100 integer,"changes_reader_understanding":0-100 integer,"rationale":"brief reason, max 35 words"}}
""".strip()
        prompts.append(prompt)
    df["agent_id"] = df["cluster"].map(lambda x: f"C{int(x):02d}")
    df["system_prompt"] = prompts
    return df


def load_notes(path: Path, max_notes: int, selection_mode: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"noteId": str, "tweetId": str}, low_memory=False)
    required = {"noteId", "tweetId", "summary", "text", "currentStatus"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sample CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["post_text"] = df["text"].fillna("").astype(str)
    df["note_text"] = df["summary"].fillna("").astype(str)
    df["currentStatus"] = df["currentStatus"].astype(str)
    df = df[df["currentStatus"].isin(STATUS_TO_LABEL)].copy()
    df = df[df["post_text"].str.strip().ne("") & df["note_text"].str.strip().ne("")].copy()
    df = df.drop_duplicates("noteId", keep="first").reset_index(drop=True)
    df["true_label_3way"] = df["currentStatus"].map(STATUS_TO_LABEL).astype(int)
    df["true_label_text"] = df["true_label_3way"].map(NOTE_STATUS_TO_TEXT)

    if max_notes <= 0 or max_notes >= len(df):
        return df.reset_index(drop=True)

    if selection_mode == "sample_order":
        return df.head(max_notes).reset_index(drop=True)

    if selection_mode == "representative":
        strata_cols = ["year", "currentStatus", "primary_topic"]
        for col in strata_cols:
            if col not in df.columns:
                raise ValueError(f"representative selection requires column: {col}")
        counts = df.groupby(strata_cols, dropna=False).size().sort_index()
        quotas = counts.astype(float) * max_notes / float(counts.sum())
        allocation = np.floor(quotas).astype(int)
        remaining = int(max_notes - allocation.sum())
        if remaining > 0:
            remainders = (quotas - allocation).sort_values(ascending=False)
            allocation.loc[remainders.index[:remaining]] += 1
        parts = []
        for i, (key, group) in enumerate(df.groupby(strata_cols, dropna=False, sort=True)):
            take = int(allocation.loc[key]) if key in allocation.index else 0
            if take <= 0:
                continue
            parts.append(group.sample(n=take, replace=False, random_state=seed + i))
        return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    statuses = ["CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL", "NEEDS_MORE_RATINGS"]
    base = max_notes // len(statuses)
    remainder = max_notes % len(statuses)
    parts = []
    selected_idx: set[int] = set()
    for i, status in enumerate(statuses):
        subset = df[df["currentStatus"] == status]
        take = min(len(subset), base + (1 if i < remainder else 0))
        if take > 0:
            idx = rng.choice(subset.index.to_numpy(), size=take, replace=False)
            selected_idx.update(int(x) for x in idx)
            parts.append(df.loc[idx])
    remaining = max_notes - sum(len(p) for p in parts)
    if remaining > 0:
        pool = df.drop(index=list(selected_idx))
        parts.append(pool.sample(n=remaining, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_user_prompt(row: pd.Series) -> str:
    created = str(row.get("createdAtUTC", "")).strip()
    topic = str(row.get("primary_topic", "")).strip()
    post = str(row["post_text"]).strip()
    note = str(row["note_text"]).strip()
    return f"""
Post creation context: {created}
Author-selected note topic: {topic}

POST:
{post}

COMMUNITY NOTE:
{note}

Evaluate only the note's usefulness for the post. Do not infer the official status from metadata.
""".strip()


def call_chat(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    doc = json.loads(body)
    return str(doc["choices"][0]["message"]["content"])


def run_one(note_row: pd.Series, persona_row: pd.Series, args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    user_prompt = build_user_prompt(note_row)
    raw = ""
    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            raw = call_chat(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                system_prompt=str(persona_row["system_prompt"]),
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            break
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(args.retry_base_sleep * attempt + random.random())
    try:
        parsed = parse_response(raw)
    except Exception as exc:
        last_error = f"{last_error}; parse={type(exc).__name__}: {exc}".strip("; ")
        parsed = {
            "helpfulnessLevel": "UNKNOWN",
            "parsed_rating": "UNKNOWN",
            "predicted_rating_score": -1.0,
            "agree": -1,
            "disagree": -1,
            "helpfulClear": -1,
            "helpfulGoodSources": -1,
            "helpfulAddressesClaim": -1,
            "helpfulImportantContext": -1,
            "helpfulUnbiasedLanguage": -1,
            "notHelpfulIncorrect": -1,
            "notHelpfulSourcesMissingOrUnreliable": -1,
            "notHelpfulMissingKeyPoints": -1,
            "notHelpfulHardToUnderstand": -1,
            "notHelpfulArgumentativeOrBiased": -1,
            "notHelpfulIrrelevantSources": -1,
            "notHelpfulOpinionSpeculation": -1,
            "notHelpfulNoteNotNeeded": -1,
            "confidence": -1,
            "changes_reader_understanding": -1,
            "rationale": "",
        }
    return {
        "noteId": str(note_row["noteId"]),
        "tweetId": str(note_row["tweetId"]),
        "currentStatus": str(note_row["currentStatus"]),
        "true_label_3way": int(note_row["true_label_3way"]),
        "true_label_text": str(note_row["true_label_text"]),
        "agent_id": str(persona_row["agent_id"]),
        "cluster": int(persona_row["cluster"]),
        "persona_name": str(persona_row["persona_name"]),
        "helpfulnessLevel": parsed["helpfulnessLevel"],
        "parsed_rating": parsed["parsed_rating"],
        "predicted_rating_score": float(parsed["predicted_rating_score"]),
        "agree": int(parsed["agree"]),
        "disagree": int(parsed["disagree"]),
        "helpfulClear": int(parsed["helpfulClear"]),
        "helpfulGoodSources": int(parsed["helpfulGoodSources"]),
        "helpfulAddressesClaim": int(parsed["helpfulAddressesClaim"]),
        "helpfulImportantContext": int(parsed["helpfulImportantContext"]),
        "helpfulUnbiasedLanguage": int(parsed["helpfulUnbiasedLanguage"]),
        "notHelpfulIncorrect": int(parsed["notHelpfulIncorrect"]),
        "notHelpfulSourcesMissingOrUnreliable": int(parsed["notHelpfulSourcesMissingOrUnreliable"]),
        "notHelpfulMissingKeyPoints": int(parsed["notHelpfulMissingKeyPoints"]),
        "notHelpfulHardToUnderstand": int(parsed["notHelpfulHardToUnderstand"]),
        "notHelpfulArgumentativeOrBiased": int(parsed["notHelpfulArgumentativeOrBiased"]),
        "notHelpfulIrrelevantSources": int(parsed["notHelpfulIrrelevantSources"]),
        "notHelpfulOpinionSpeculation": int(parsed["notHelpfulOpinionSpeculation"]),
        "notHelpfulNoteNotNeeded": int(parsed["notHelpfulNoteNotNeeded"]),
        "confidence": int(parsed["confidence"]),
        "changes_reader_understanding": int(parsed["changes_reader_understanding"]),
        "rationale": parsed["rationale"],
        "raw_completion": raw,
        "api_error": last_error,
    }


def save_votes(votes: list[dict[str, Any]], outdir: Path) -> None:
    if not votes:
        return
    df = pd.DataFrame(votes).sort_values(["noteId", "agent_id"]).reset_index(drop=True)
    df.to_csv(outdir / "agent_votes.csv", index=False, encoding="utf-8-sig")
    note_rows = []
    for note_id, group in df.groupby("noteId", sort=False):
        valid = group[group["predicted_rating_score"].isin([0.0, 0.5, 1.0])]
        counts = valid["parsed_rating"].value_counts().to_dict()
        note_rows.append(
            {
                "noteId": note_id,
                "tweetId": group["tweetId"].iloc[0],
                "currentStatus": group["currentStatus"].iloc[0],
                "true_label_3way": int(group["true_label_3way"].iloc[0]),
                "true_label_text": group["true_label_text"].iloc[0],
                "n_votes": int(len(group)),
                "valid_votes": int(len(valid)),
                "mean_rating_score": float(pd.to_numeric(valid["predicted_rating_score"], errors="coerce").mean()) if not valid.empty else np.nan,
                "vote_helpful": int(counts.get("HELPFUL", 0)),
                "vote_somewhat_helpful": int(counts.get("SOMEWHAT_HELPFUL", 0)),
                "vote_not_helpful": int(counts.get("NOT_HELPFUL", 0)),
                "agree_rate": float(pd.to_numeric(valid["agree"], errors="coerce").mean()) if not valid.empty else np.nan,
                "disagree_rate": float(pd.to_numeric(valid["disagree"], errors="coerce").mean()) if not valid.empty else np.nan,
                "helpful_clear_rate": float(pd.to_numeric(valid["helpfulClear"], errors="coerce").mean()) if not valid.empty else np.nan,
                "helpful_good_sources_rate": float(pd.to_numeric(valid["helpfulGoodSources"], errors="coerce").mean()) if not valid.empty else np.nan,
                "helpful_addresses_claim_rate": float(pd.to_numeric(valid["helpfulAddressesClaim"], errors="coerce").mean()) if not valid.empty else np.nan,
                "helpful_important_context_rate": float(pd.to_numeric(valid["helpfulImportantContext"], errors="coerce").mean()) if not valid.empty else np.nan,
                "helpful_unbiased_language_rate": float(pd.to_numeric(valid["helpfulUnbiasedLanguage"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_incorrect_rate": float(pd.to_numeric(valid["notHelpfulIncorrect"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_sources_missing_or_unreliable_rate": float(pd.to_numeric(valid["notHelpfulSourcesMissingOrUnreliable"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_missing_key_points_rate": float(pd.to_numeric(valid["notHelpfulMissingKeyPoints"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_hard_to_understand_rate": float(pd.to_numeric(valid["notHelpfulHardToUnderstand"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_argumentative_or_biased_rate": float(pd.to_numeric(valid["notHelpfulArgumentativeOrBiased"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_irrelevant_sources_rate": float(pd.to_numeric(valid["notHelpfulIrrelevantSources"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_opinion_speculation_rate": float(pd.to_numeric(valid["notHelpfulOpinionSpeculation"], errors="coerce").mean()) if not valid.empty else np.nan,
                "not_helpful_note_not_needed_rate": float(pd.to_numeric(valid["notHelpfulNoteNotNeeded"], errors="coerce").mean()) if not valid.empty else np.nan,
                "mean_confidence": float(pd.to_numeric(valid["confidence"], errors="coerce").mean()) if not valid.empty else np.nan,
                "mean_changes_reader_understanding": float(pd.to_numeric(valid["changes_reader_understanding"], errors="coerce").mean()) if not valid.empty else np.nan,
            }
        )
    pd.DataFrame(note_rows).to_csv(outdir / "note_vote_summary.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(args.outdir)
    api_key = read_api_key(args)

    summary = pd.read_csv(args.cluster_summary_csv)
    personas = build_personas(summary)
    notes = load_notes(args.sample_csv, args.max_notes, args.selection_mode, args.seed)

    personas.to_csv(args.outdir / "persona_prompts.csv", index=False, encoding="utf-8-sig")
    notes.to_csv(args.outdir / "pilot_notes.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "time_utc": utc_now(),
        "sample_csv": str(args.sample_csv),
        "cluster_summary_csv": str(args.cluster_summary_csv),
        "model": args.model,
        "base_url": args.base_url,
        "n_notes": int(len(notes)),
        "n_agents": int(len(personas)),
        "tasks": int(len(notes) * len(personas)),
        "selection_mode": args.selection_mode,
        "output_schema": REQUIRED_OUTPUT_KEYS,
        "label_space": RAW_RATING_TO_SCORE,
    }
    (args.outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = [(note_row, persona_row) for _, note_row in notes.iterrows() for _, persona_row in personas.iterrows()]
    writer = JsonlWriter(args.outdir / "agent_votes.jsonl")
    votes: list[dict[str, Any]] = []
    completed = 0
    progress.update("llm calls", 0, len(tasks), "start")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_map = {
            pool.submit(run_one, note_row, persona_row, args, api_key): (str(note_row["noteId"]), str(persona_row["agent_id"]))
            for note_row, persona_row in tasks
        }
        for future in as_completed(future_map):
            completed += 1
            note_id, agent_id = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "noteId": note_id,
                    "tweetId": "",
                    "currentStatus": "",
                    "true_label_3way": -1,
                    "true_label_text": "",
                    "agent_id": agent_id,
                    "cluster": -1,
                    "persona_name": "",
                    "parsed_rating": "UNKNOWN",
                    "predicted_rating_score": -1.0,
                    "confidence": -1,
                    "addresses_core_claim": -1,
                    "changes_reader_understanding": -1,
                    "note_needed": -1,
                    "evidence_strength": -1,
                    "misses_key_points": -1,
                    "too_minor_or_tangential": -1,
                    "rationale": "",
                    "raw_completion": "",
                    "api_error": f"{type(exc).__name__}: {exc}",
                }
            writer.write(row)
            votes.append(row)
            if completed == 1 or completed % args.progress_every == 0 or completed == len(tasks):
                valid = sum(1 for v in votes if float(v.get("predicted_rating_score", -1.0)) in {0.0, 0.5, 1.0})
                progress.update("llm calls", completed, len(tasks), f"last={note_id}/{agent_id}", {"valid_votes": valid})
            if completed == 1 or completed % args.save_every == 0 or completed == len(tasks):
                save_votes(votes, args.outdir)

    save_votes(votes, args.outdir)
    vote_df = pd.DataFrame(votes)
    final_summary = {
        **metadata,
        "completed_votes": int(len(vote_df)),
        "valid_votes": int(vote_df["predicted_rating_score"].isin([0.0, 0.5, 1.0]).sum()),
        "rating_distribution": vote_df["parsed_rating"].value_counts(dropna=False).to_dict(),
        "api_errors": int(vote_df["api_error"].fillna("").astype(str).str.strip().ne("").sum()),
    }
    (args.outdir / "summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    progress.update("complete", 1, 1, "16-agent raw-rating pilot complete")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
