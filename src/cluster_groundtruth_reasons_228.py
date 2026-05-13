from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE = Path("/data6/wenchangxi/community_note")
RUN_DIR = BASE / "analysis/llm_16agent_binaryrating_balanced_228_20260513"
DATA_DIR = BASE / "data/extracted_communitynotes_2026-04-07"
OUT_DIR = RUN_DIR / "groundtruth_reason_clusters_20260513"

HELPFUL_REASON_FIELDS = [
    "helpfulOther",
    "helpfulClear",
    "helpfulGoodSources",
    "helpfulAddressesClaim",
    "helpfulImportantContext",
    "helpfulUnbiasedLanguage",
]
NOT_HELPFUL_REASON_FIELDS = [
    "notHelpfulOther",
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
NOTE_META_FIELDS = [
    "misleadingManipulatedMedia",
    "misleadingFactualError",
    "misleadingOutdatedInformation",
    "misleadingMissingImportantContext",
    "misleadingUnverifiedClaimAsFact",
    "misleadingSatire",
    "notMisleadingOther",
    "notMisleadingFactuallyCorrect",
    "notMisleadingOutdatedButNotWhenWritten",
    "notMisleadingClearlySatire",
    "notMisleadingPersonalOpinion",
]


def clean01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1)


def read_target_notes() -> pd.DataFrame:
    notes = pd.read_csv(RUN_DIR / "pilot_notes.csv", dtype={"noteId": str, "tweetId": str}, low_memory=False)
    notes["noteId"] = notes["noteId"].astype(str)
    notes["true_label_3way"] = pd.to_numeric(notes["true_label_3way"], errors="coerce").astype(int)
    notes["true_label_text"] = notes["true_label_text"].astype(str)
    for col in NOTE_META_FIELDS:
        if col in notes.columns:
            notes[col] = clean01(notes[col])
        else:
            notes[col] = 0
    return notes


def extract_ratings_for_notes(note_ids: set[str]) -> pd.DataFrame:
    parts = []
    rating_files = sorted((DATA_DIR / "noteRatings").glob("ratings-*.tsv"))
    for idx, path in enumerate(rating_files, start=1):
        print(f"[ratings {idx}/{len(rating_files)}] {path.name}", flush=True)
        matched = 0
        for chunk in pd.read_csv(path, sep="\t", dtype={"noteId": str}, low_memory=False, chunksize=750_000):
            chunk = chunk[chunk["noteId"].astype(str).isin(note_ids)].copy()
            if len(chunk):
                matched += len(chunk)
                parts.append(chunk)
        if matched:
            print(f"  matched {matched} rows", flush=True)
    if not parts:
        return pd.DataFrame()
    ratings = pd.concat(parts, ignore_index=True)
    ratings["noteId"] = ratings["noteId"].astype(str)
    for col in HELPFUL_REASON_FIELDS + NOT_HELPFUL_REASON_FIELDS:
        if col not in ratings.columns:
            ratings[col] = 0
        ratings[col] = clean01(ratings[col])
    return ratings


def aggregate_reasons(notes: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        agg = notes[["noteId"]].copy()
        for col in HELPFUL_REASON_FIELDS + NOT_HELPFUL_REASON_FIELDS:
            agg[f"{col}_rate"] = np.nan
        agg["n_ratings"] = 0
        agg["n_helpful_raw"] = 0
        agg["n_somewhat_raw"] = 0
        agg["n_not_helpful_raw"] = 0
    else:
        ratings["helpfulnessLevel"] = ratings.get("helpfulnessLevel", "").astype(str)
        spec = {
            "n_ratings": ("participantId", "size") if "participantId" in ratings.columns else ("noteId", "size"),
            "n_helpful_raw": ("helpfulnessLevel", lambda s: int((s == "HELPFUL").sum())),
            "n_somewhat_raw": ("helpfulnessLevel", lambda s: int((s == "SOMEWHAT_HELPFUL").sum())),
            "n_not_helpful_raw": ("helpfulnessLevel", lambda s: int((s == "NOT_HELPFUL").sum())),
        }
        for col in HELPFUL_REASON_FIELDS + NOT_HELPFUL_REASON_FIELDS:
            spec[f"{col}_rate"] = (col, "mean")
        agg = ratings.groupby("noteId", as_index=False).agg(**spec)
    merged = notes.merge(agg, on="noteId", how="left")
    for col in ["n_ratings", "n_helpful_raw", "n_somewhat_raw", "n_not_helpful_raw"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    for col in [f"{c}_rate" for c in HELPFUL_REASON_FIELDS + NOT_HELPFUL_REASON_FIELDS]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    return merged


def text_features(df: pd.DataFrame):
    text = (
        "POST: "
        + df.get("post_text", df.get("text", "")).fillna("").astype(str)
        + "\nNOTE: "
        + df.get("note_text", df.get("summary", "")).fillna("").astype(str)
    )
    vec = TfidfVectorizer(max_features=80, ngram_range=(1, 2), stop_words="english", min_df=2)
    try:
        x = vec.fit_transform(text)
        return pd.DataFrame(x.toarray(), columns=[f"tfidf_{t}" for t in vec.get_feature_names_out()], index=df.index)
    except ValueError:
        return pd.DataFrame(index=df.index)


def choose_k(x: np.ndarray, n: int) -> int:
    if n < 8:
        return min(2, n)
    best_k, best_score = 2, -1.0
    for k in range(2, min(8, n - 1) + 1):
        labels = KMeans(n_clusters=k, random_state=20260513, n_init=30).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(x, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def top_terms(row: pd.Series, fields: list[str], n: int = 5) -> list[str]:
    vals = [(f, float(row.get(f, 0))) for f in fields]
    vals = sorted(vals, key=lambda x: x[1], reverse=True)
    return [f"{name.replace('_rate','')}={value:.2f}" for name, value in vals[:n] if value > 0]


def cluster_subset(df: pd.DataFrame, label_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df["true_label_text"] == label_name].copy().reset_index(drop=True)
    reason_cols = [f"{c}_rate" for c in HELPFUL_REASON_FIELDS + NOT_HELPFUL_REASON_FIELDS]
    meta_cols = NOTE_META_FIELDS + ["n_ratings", "n_helpful_raw", "n_somewhat_raw", "n_not_helpful_raw"]
    tfidf = text_features(subset)
    feature_cols = reason_cols + meta_cols
    feature_df = pd.concat([subset[feature_cols], tfidf], axis=1).fillna(0)
    scaler = StandardScaler()
    x = scaler.fit_transform(feature_df)
    k = choose_k(x, len(subset))
    labels = KMeans(n_clusters=k, random_state=20260513, n_init=50).fit_predict(x)
    subset["reason_cluster"] = labels

    summary_rows = []
    for cid, group in subset.groupby("reason_cluster"):
        center = group[reason_cols + meta_cols].mean(numeric_only=True)
        helpful_terms = top_terms(center, [f"{c}_rate" for c in HELPFUL_REASON_FIELDS], 6)
        not_helpful_terms = top_terms(center, [f"{c}_rate" for c in NOT_HELPFUL_REASON_FIELDS], 8)
        meta_terms = top_terms(center, NOTE_META_FIELDS, 6)
        examples = []
        for _, row in group.head(5).iterrows():
            post = " ".join(str(row.get("post_text", row.get("text", ""))).split())[:180]
            note = " ".join(str(row.get("note_text", row.get("summary", ""))).split())[:220]
            examples.append({"noteId": row["noteId"], "post": post, "note": note})
        summary_rows.append(
            {
                "label": label_name,
                "cluster": int(cid),
                "n": int(len(group)),
                "share": float(len(group) / max(len(subset), 1)),
                "avg_ratings": float(group["n_ratings"].mean()),
                "avg_raw_helpful": float(group["n_helpful_raw"].mean()),
                "avg_raw_somewhat": float(group["n_somewhat_raw"].mean()),
                "avg_raw_not_helpful": float(group["n_not_helpful_raw"].mean()),
                "top_helpful_reasons": "; ".join(helpful_terms),
                "top_not_helpful_reasons": "; ".join(not_helpful_terms),
                "top_note_meta_tags": "; ".join(meta_terms),
                "example_json": json.dumps(examples, ensure_ascii=False),
            }
        )
    return subset, pd.DataFrame(summary_rows).sort_values(["label", "cluster"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notes = read_target_notes()
    ratings = extract_ratings_for_notes(set(notes["noteId"].astype(str)))
    ratings.to_csv(OUT_DIR / "matched_official_ratings_for_228.csv", index=False, encoding="utf-8-sig")
    data = aggregate_reasons(notes, ratings)
    data.to_csv(OUT_DIR / "notes_with_official_reason_rates.csv", index=False, encoding="utf-8-sig")

    all_labeled = []
    all_summary = []
    for label in ["HELPFUL", "NOT_HELPFUL"]:
        labeled, summary = cluster_subset(data, label)
        all_labeled.append(labeled)
        all_summary.append(summary)
    labeled_df = pd.concat(all_labeled, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    labeled_df.to_csv(OUT_DIR / "helpful_nh_notes_with_reason_clusters.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_DIR / "reason_cluster_summary.csv", index=False, encoding="utf-8-sig")
    print("matched ratings:", len(ratings), "notes:", ratings["noteId"].nunique() if not ratings.empty else 0)
    print(summary_df[["label", "cluster", "n", "share", "avg_ratings", "top_helpful_reasons", "top_not_helpful_reasons", "top_note_meta_tags"]].to_string(index=False))
    print(f"Saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
