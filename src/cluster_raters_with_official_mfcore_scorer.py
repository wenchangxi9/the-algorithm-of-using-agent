#!/usr/bin/env python3
"""Cluster Community Notes raters using X's official MFCoreScorer outputs.

This script deliberately separates two steps:
1. Use the official open-source X Community Notes scorer code to learn
   rater-level behavioral representations.
2. Run our downstream clustering on those scorer-derived rater features.

The clustering step is not part of the official scorer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("official_mfcore_rater_clustering")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use X official MFCoreScorer rater outputs for downstream rater clustering."
    )
    parser.add_argument("--official-src", required=True, help="Path to official scoring/src.")
    parser.add_argument("--notes", required=True, help="Path to notes TSV or directory.")
    parser.add_argument("--ratings", required=True, help="Path to ratings TSV or directory.")
    parser.add_argument("--status", required=True, help="Path to noteStatusHistory TSV or directory.")
    parser.add_argument("--enrollment", required=True, help="Path to userEnrollment TSV or directory.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--k-list",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12, 16],
        help="Cluster counts to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=["prescore", "score"],
        default="prescore",
        help=(
            "prescore uses official MFCoreScorer.prescore rater outputs; "
            "score additionally runs final MFCoreScorer.score output."
        ),
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10000,
        help="Sample size used for silhouette/CH/DB metrics.",
    )
    parser.add_argument(
        "--min-complete-features",
        type=int,
        default=2,
        help="Minimum non-missing clustering features required per rater.",
    )
    return parser.parse_args()


def configure_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def add_official_src_to_path(official_src: str) -> None:
    official_src_path = Path(official_src).resolve()
    if not official_src_path.exists():
        raise FileNotFoundError(f"Official scoring src path not found: {official_src_path}")
    sys.path.insert(0, str(official_src_path))


def memory_note(label: str) -> None:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss_gb = process.memory_info().rss / (1024**3)
        LOGGER.info("%s RSS memory: %.2f GiB", label, rss_gb)
    except Exception:
        LOGGER.info("%s RSS memory: unavailable", label)


def load_official_data(args: argparse.Namespace):
    from scoring.process_data import LocalDataLoader

    LOGGER.info("Loading and official-preprocessing Community Notes data.")
    loader = LocalDataLoader(
        args.notes,
        args.ratings,
        args.status,
        args.enrollment,
        headers=True,
        shouldFilterNotMisleadingNotes=True,
        log=True,
    )
    notes, ratings, note_status_history, user_enrollment = loader.get_data()
    LOGGER.info(
        "Official preprocessed data: notes=%s ratings=%s raters=%s status_rows=%s enrollment_rows=%s",
        f"{len(notes):,}",
        f"{len(ratings):,}",
        f"{ratings['raterParticipantId'].nunique():,}",
        f"{len(note_status_history):,}",
        f"{len(user_enrollment):,}",
    )
    memory_note("After official data load")
    return notes, ratings, note_status_history, user_enrollment


def make_empty_note_topics(notes: pd.DataFrame):
    from scoring import constants as c

    # MFCoreScorer has excludeTopics=True. An empty noteTopics table means no notes are
    # excluded as topic-specific notes in this standalone MFCore run.
    return pd.DataFrame(
        {
            c.noteIdKey: pd.Series([], dtype=notes[c.noteIdKey].dtype),
            c.noteTopicKey: pd.Series([], dtype="int64"),
        }
    )


def run_official_mfcore(args: argparse.Namespace, outdir: Path):
    from scoring import constants as c
    from scoring.mf_core_scorer import MFCoreScorer
    from scoring.process_data import write_tsv_local

    notes, ratings, note_status_history, user_enrollment = load_official_data(args)
    note_topics = make_empty_note_topics(notes)
    scorer = MFCoreScorer(
        seed=args.seed,
        pseudoraters=False,
        useStableInitialization=True,
        saveIntermediateState=False,
        threads=args.threads,
    )

    LOGGER.info("Running official MFCoreScorer in %s mode.", args.mode)
    start = time.time()
    if args.mode == "score":
        scored_notes, helpfulness_scores, aux_note_info = scorer.score(
            note_topics,
            ratings,
            note_status_history,
            user_enrollment,
        )
        rater_output = helpfulness_scores.copy()
        note_output = scored_notes.copy()
        aux_path = outdir / "official_mfcore_aux_note_info.tsv"
        write_tsv_local(aux_note_info, str(aux_path))
    else:
        result = scorer.prescore(
            c.PrescoringArgs(
                noteTopics=note_topics,
                ratings=ratings,
                noteStatusHistory=note_status_history,
                userEnrollment=user_enrollment,
            )
        )
        rater_output = result.helpfulnessScores.copy()
        note_output = result.scoredNotes.copy()
        if result.scorerName is not None and c.scorerNameKey not in rater_output.columns:
            rater_output[c.scorerNameKey] = result.scorerName
        if result.scorerName is not None and c.scorerNameKey not in note_output.columns:
            note_output[c.scorerNameKey] = result.scorerName

    LOGGER.info("Official MFCoreScorer finished in %.2f minutes.", (time.time() - start) / 60.0)
    LOGGER.info("Official rater output rows=%s cols=%s", f"{len(rater_output):,}", list(rater_output.columns))
    memory_note("After official MFCoreScorer")

    rater_path = outdir / "official_mfcore_rater_output.tsv"
    note_path = outdir / "official_mfcore_note_output.tsv"
    write_tsv_local(rater_output, str(rater_path))
    write_tsv_local(note_output, str(note_path))
    LOGGER.info("Saved official MFCore rater output: %s", rater_path)
    LOGGER.info("Saved official MFCore note output: %s", note_path)
    return rater_output


def first_existing(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def prepare_feature_frame(rater_output: pd.DataFrame, min_complete_features: int) -> tuple[pd.DataFrame, list[str]]:
    # Handle both prescoring internal column names and final MFCore external names.
    id_candidates = ["raterParticipantId", "participantId"]
    id_cols = first_existing(id_candidates, rater_output)
    if not id_cols:
        raise ValueError(f"No rater id column found. Columns: {list(rater_output.columns)}")
    rater_id_col = id_cols[0]

    feature_candidates = [
        "coreRaterIntercept",
        "coreRaterFactor1",
        "coreFirstRoundRaterIntercept",
        "coreFirstRoundRaterFactor1",
        "internalRaterIntercept",
        "internalRaterFactor1",
        "internalFirstRoundRaterIntercept",
        "internalFirstRoundRaterFactor1",
        "crhCrnhRatioDifference",
        "meanNoteScore",
        "raterAgreeRatio",
        "aboveHelpfulnessThreshold",
    ]
    feature_cols = first_existing(feature_candidates, rater_output)
    if not feature_cols:
        raise ValueError(f"No expected official MFCore rater feature columns found.")

    features = rater_output[[rater_id_col] + feature_cols].copy()
    if rater_id_col != "raterParticipantId":
        features = features.rename(columns={rater_id_col: "raterParticipantId"})

    for col in feature_cols:
        if str(features[col].dtype) in {"boolean", "bool"}:
            features[col] = features[col].astype("float64")
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    non_missing = features[feature_cols].notna().sum(axis=1)
    features = features[non_missing >= min_complete_features].copy()
    features[feature_cols] = features[feature_cols].fillna(features[feature_cols].median(numeric_only=True))
    features = features.drop_duplicates(subset=["raterParticipantId"])
    LOGGER.info(
        "Prepared clustering features: raters=%s feature_cols=%s",
        f"{len(features):,}",
        feature_cols,
    )
    return features, feature_cols


def evaluate_and_cluster(
    features: pd.DataFrame,
    feature_cols: list[str],
    k_list: list[int],
    sample_size: int,
    seed: int,
):
    x = features[feature_cols].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    rng = np.random.default_rng(seed)
    metric_n = min(sample_size, len(features))
    metric_idx = rng.choice(len(features), size=metric_n, replace=False)
    x_metric = x_scaled[metric_idx]

    metrics = []
    fitted_models = {}
    for k in k_list:
        LOGGER.info("Fitting MiniBatchKMeans K=%s on %s raters.", k, f"{len(features):,}")
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=8192,
            n_init=20,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(x_scaled)
        fitted_models[k] = (model, labels)
        metric_labels = labels[metric_idx]
        row = {
            "k": k,
            "n_raters": len(features),
            "metric_sample_size": metric_n,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(x_metric, metric_labels)) if len(set(metric_labels)) > 1 else np.nan,
            "calinski_harabasz": float(calinski_harabasz_score(x_metric, metric_labels)) if len(set(metric_labels)) > 1 else np.nan,
            "davies_bouldin": float(davies_bouldin_score(x_metric, metric_labels)) if len(set(metric_labels)) > 1 else np.nan,
        }
        LOGGER.info("K=%s metrics: %s", k, row)
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics)
    best_row = metrics_df.sort_values(["silhouette", "calinski_harabasz"], ascending=[False, False]).iloc[0]
    best_k = int(best_row["k"])
    LOGGER.info("Selected best K by silhouette: K=%s", best_k)
    return metrics_df, best_k, fitted_models[best_k][1]


def summarize_clusters(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    summary = (
        features.groupby("cluster", dropna=False)
        .agg(
            n_raters=("raterParticipantId", "size"),
            **{f"mean_{col}": (col, "mean") for col in feature_cols},
            **{f"median_{col}": (col, "median") for col in feature_cols},
        )
        .reset_index()
        .sort_values("cluster")
    )
    total = summary["n_raters"].sum()
    summary["share"] = summary["n_raters"] / total
    return summary


def write_metadata(args: argparse.Namespace, outdir: Path, metrics_df: pd.DataFrame, best_k: int, feature_cols: list[str]) -> None:
    metadata = {
        "description": "Official X Community Notes MFCoreScorer rater representations followed by downstream clustering.",
        "important_note": "The official scorer provides rater-level representations; clustering is our downstream agent-construction step.",
        "mode": args.mode,
        "seed": args.seed,
        "threads": args.threads,
        "k_list": args.k_list,
        "selected_k_by_silhouette": best_k,
        "feature_columns": feature_cols,
        "inputs": {
            "official_src": args.official_src,
            "notes": args.notes,
            "ratings": args.ratings,
            "status": args.status,
            "enrollment": args.enrollment,
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    configure_logging(outdir)
    add_official_src_to_path(args.official_src)

    LOGGER.info("Starting official MFCore scorer-based rater clustering.")
    LOGGER.info("Arguments: %s", vars(args))
    start = time.time()

    rater_output = run_official_mfcore(args, outdir)
    features, feature_cols = prepare_feature_frame(rater_output, args.min_complete_features)
    metrics_df, best_k, labels = evaluate_and_cluster(
        features,
        feature_cols,
        args.k_list,
        args.silhouette_sample_size,
        args.seed,
    )
    features["cluster"] = labels
    summary = summarize_clusters(features, feature_cols)

    metrics_path = outdir / "k_selection_metrics.csv"
    features_path = outdir / "rater_features_with_official_mfcore_clusters.csv"
    summary_path = outdir / "cluster_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    features.to_csv(features_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_metadata(args, outdir, metrics_df, best_k, feature_cols)

    LOGGER.info("Saved K metrics: %s", metrics_path)
    LOGGER.info("Saved rater features with clusters: %s", features_path)
    LOGGER.info("Saved cluster summary: %s", summary_path)
    LOGGER.info("Done in %.2f minutes.", (time.time() - start) / 60.0)


if __name__ == "__main__":
    main()
