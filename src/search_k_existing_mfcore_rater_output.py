#!/usr/bin/env python3
"""Search K for clustering existing official MFCoreScorer rater outputs."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("search_k_existing_mfcore")


def setup_logging(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(outdir / "run.log", encoding="utf-8"),
        ],
    )


def prepare_feature_frame(rater_output: pd.DataFrame, min_complete_features: int) -> tuple[pd.DataFrame, list[str]]:
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
    feature_cols = [col for col in feature_candidates if col in rater_output.columns]
    if "raterParticipantId" not in rater_output.columns:
        raise ValueError("Missing raterParticipantId")
    if not feature_cols:
        raise ValueError("No expected feature columns")

    features = rater_output[["raterParticipantId"] + feature_cols].copy()
    for col in feature_cols:
        if str(features[col].dtype) in {"boolean", "bool"}:
            features[col] = features[col].astype("float64")
        else:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    non_missing = features[feature_cols].notna().sum(axis=1)
    features = features[non_missing >= min_complete_features].copy()
    features[feature_cols] = features[feature_cols].fillna(features[feature_cols].median(numeric_only=True))
    features = features.drop_duplicates(subset=["raterParticipantId"])
    return features, feature_cols


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
    summary["share"] = summary["n_raters"] / summary["n_raters"].sum()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater-output", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=32)
    parser.add_argument("--silhouette-sample-size", type=int, default=10000)
    parser.add_argument("--min-complete-features", type=int, default=2)
    parser.add_argument("--save-best-features", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    setup_logging(outdir)
    start = time.time()

    LOGGER.info("Loading existing official MFCore rater output: %s", args.rater_output)
    rater_output = pd.read_csv(args.rater_output, sep="\t", low_memory=False)
    LOGGER.info("Loaded rows=%s cols=%s", f"{len(rater_output):,}", len(rater_output.columns))

    features, feature_cols = prepare_feature_frame(rater_output, args.min_complete_features)
    LOGGER.info("Prepared features: raters=%s feature_cols=%s", f"{len(features):,}", feature_cols)

    x = features[feature_cols].to_numpy(dtype=np.float64)
    x_scaled = StandardScaler().fit_transform(x)

    rng = np.random.default_rng(args.seed)
    metric_n = min(args.silhouette_sample_size, len(features))
    metric_idx = rng.choice(len(features), size=metric_n, replace=False)
    x_metric = x_scaled[metric_idx]

    metrics: list[dict[str, float | int]] = []
    best_labels = None
    best_k = None
    best_sil = -np.inf

    for k in range(args.k_min, args.k_max + 1):
        LOGGER.info("Fitting MiniBatchKMeans K=%s on %s raters", k, f"{len(features):,}")
        t0 = time.time()
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=args.seed,
            batch_size=8192,
            n_init=20,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(x_scaled)
        metric_labels = labels[metric_idx]
        if len(set(metric_labels)) > 1:
            sil = float(silhouette_score(x_metric, metric_labels))
            ch = float(calinski_harabasz_score(x_metric, metric_labels))
            db = float(davies_bouldin_score(x_metric, metric_labels))
        else:
            sil = float("nan")
            ch = float("nan")
            db = float("nan")

        row = {
            "k": k,
            "n_raters": len(features),
            "metric_sample_size": metric_n,
            "inertia": float(model.inertia_),
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "seconds": time.time() - t0,
        }
        metrics.append(row)
        pd.DataFrame(metrics).to_csv(outdir / "k_selection_metrics_k2_to_k32.csv", index=False)
        LOGGER.info("K=%s metrics=%s", k, row)

        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels.copy()

    metrics_df = pd.DataFrame(metrics)
    best_by_sil = int(metrics_df.sort_values(["silhouette", "calinski_harabasz"], ascending=[False, False]).iloc[0]["k"])
    best_by_db = int(metrics_df.sort_values(["davies_bouldin", "silhouette"], ascending=[True, False]).iloc[0]["k"])
    LOGGER.info("Best K by silhouette=%s; best K by Davies-Bouldin=%s", best_by_sil, best_by_db)

    if args.save_best_features and best_labels is not None and best_k is not None:
        features = features.copy()
        features["cluster"] = best_labels
        features.to_csv(outdir / f"rater_features_with_k{best_k}_clusters.csv", index=False)
        summarize_clusters(features, feature_cols).to_csv(outdir / f"cluster_summary_k{best_k}.csv", index=False)

    metadata = {
        "description": "K=2..32 interval-1 search using existing official MFCoreScorer rater outputs.",
        "rater_output": args.rater_output,
        "seed": args.seed,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "feature_columns": feature_cols,
        "selected_k_by_silhouette": best_by_sil,
        "selected_k_by_davies_bouldin": best_by_db,
        "elapsed_minutes": (time.time() - start) / 60.0,
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    LOGGER.info("Done in %.2f minutes", (time.time() - start) / 60.0)


if __name__ == "__main__":
    main()
