from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


MISLEADING_CLASSIFICATION = "MISINFORMED_OR_POTENTIALLY_MISLEADING"
PUBLIC_TSV_DELAY_MS = 48 * 60 * 60 * 1000
DELETED_NOTE_TOMBSTONES_LAUNCH_MS = 1652918400000
MAX_HISTORICAL_VALID_RATINGS = 5


@dataclass
class RatingDataset:
    user_ids: np.ndarray
    note_ids: np.ndarray
    user_idx: np.ndarray
    note_idx: np.ndarray
    helpful_num: np.ndarray
    created_at_millis: np.ndarray

    @property
    def n_users(self) -> int:
        return int(len(self.user_ids))

    @property
    def n_notes(self) -> int:
        return int(len(self.note_ids))

    @property
    def n_ratings(self) -> int:
        return int(len(self.helpful_num))


@dataclass
class MFResult:
    global_intercept: float
    user_intercepts: np.ndarray
    user_factors: np.ndarray
    note_intercepts: np.ndarray
    note_factors: np.ndarray
    rmse_history: list[float]


@dataclass
class HelpfulnessResult:
    participant_ids: np.ndarray
    rater_agree_ratio: np.ndarray
    rating_count: np.ndarray
    successful_rating_count: np.ndarray
    unsuccessful_rating_count: np.ndarray
    mean_note_score: np.ndarray
    crh_ratio: np.ndarray
    crnh_ratio: np.ndarray
    crh_crnh_ratio_difference: np.ndarray
    above_helpfulness_threshold: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster Community Notes users with a Birdwatch-style pipeline: "
            "two-stage biased matrix factorization, contributor helpfulness filtering, "
            "and KMeans over paper-derived user parameters."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("extracted_communitynotes_2026-04-07"),
        help="Directory containing extracted Community Notes TSV files.",
    )
    parser.add_argument(
        "--base-features",
        type=Path,
        default=Path("data/base_user_features/user_features_with_behavior_features.csv"),
        help="Existing user feature table used to define the user universe and legacy summaries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mf_clustering"),
        help="Directory where matrix-factorized clustering outputs will be written.",
    )
    parser.add_argument(
        "--ratings-chunksize",
        type=int,
        default=2_000_000,
        help="Chunk size for scanning rating TSVs.",
    )
    parser.add_argument(
        "--max-base-users",
        type=int,
        default=None,
        help="Optional limit on the number of base users kept for debugging.",
    )
    parser.add_argument(
        "--max-ratings-files",
        type=int,
        default=None,
        help="Optional limit on the number of ratings shards to scan.",
    )
    parser.add_argument(
        "--min-ratings-per-rater",
        type=int,
        default=10,
        help="Minimum number of ratings a user needs before MF filtering.",
    )
    parser.add_argument(
        "--min-raters-per-note",
        type=int,
        default=5,
        help="Minimum number of ratings a note needs before MF filtering.",
    )
    parser.add_argument(
        "--als-iterations",
        type=int,
        default=12,
        help="Maximum ALS iterations for each MF pass.",
    )
    parser.add_argument(
        "--als-tol",
        type=float,
        default=1e-4,
        help="Relative RMSE improvement tolerance for ALS early stopping.",
    )
    parser.add_argument(
        "--global-intercept-lambda",
        type=float,
        default=0.15,
        help="L2 penalty on the global intercept.",
    )
    parser.add_argument(
        "--user-intercept-lambda",
        type=float,
        default=0.15,
        help="L2 penalty on user intercepts.",
    )
    parser.add_argument(
        "--note-intercept-lambda",
        type=float,
        default=0.15,
        help="L2 penalty on note intercepts.",
    )
    parser.add_argument(
        "--user-factor-lambda",
        type=float,
        default=0.03,
        help="L2 penalty on user factors.",
    )
    parser.add_argument(
        "--note-factor-lambda",
        type=float,
        default=0.03,
        help="L2 penalty on note factors.",
    )
    parser.add_argument(
        "--crh-threshold",
        type=float,
        default=0.40,
        help="Preliminary note-intercept threshold for CRH.",
    )
    parser.add_argument(
        "--crnh-intercept-threshold",
        type=float,
        default=-0.05,
        help="Base preliminary note-intercept threshold for CRNH.",
    )
    parser.add_argument(
        "--crnh-note-factor-multiplier",
        type=float,
        default=-0.80,
        help="Multiplier applied to abs(note_factor) when thresholding CRNH notes.",
    )
    parser.add_argument(
        "--min-mean-note-score",
        type=float,
        default=0.05,
        help="Minimum mean preliminary note score for contributor helpfulness.",
    )
    parser.add_argument(
        "--min-crh-vs-crnh-ratio",
        type=float,
        default=0.0,
        help="Minimum CRH ratio minus 5 * CRNH ratio for contributor helpfulness.",
    )
    parser.add_argument(
        "--min-rater-agree-ratio",
        type=float,
        default=0.66,
        help="Minimum valid-rating agreement ratio for contributor helpfulness.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="4,6,8,10,12,16",
        help="Comma-separated K candidates for downstream user clustering.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=20_000,
        help="Sample size used for silhouette scoring during K selection.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for MF initialization, sampling, and clustering.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def load_base_features(path: Path, max_base_users: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "participantId" not in df.columns:
        raise ValueError(f"{path} is missing participantId")
    if "cluster" in df.columns:
        df = df.rename(columns={"cluster": "old_cluster"})
    df["participantId"] = df["participantId"].astype("string")
    if max_base_users is not None:
        df = df.head(max_base_users).copy()
    return df.reset_index(drop=True)


def load_misleading_notes(notes_files: list[Path]) -> set[int]:
    misleading_note_ids: set[int] = set()
    for notes_file in notes_files:
        log(f"[notes] reading misleading-note ids from {notes_file.name}")
        reader = pd.read_csv(
            notes_file,
            sep="\t",
            usecols=["noteId", "classification"],
            dtype={"noteId": "int64", "classification": "string"},
            chunksize=500_000,
            low_memory=False,
        )
        for chunk in reader:
            mask = chunk["classification"] == MISLEADING_CLASSIFICATION
            if mask.any():
                misleading_note_ids.update(chunk.loc[mask, "noteId"].astype("int64").tolist())
    if not misleading_note_ids:
        raise RuntimeError("No misleading notes were found in the notes TSVs.")
    return misleading_note_ids


def compute_helpful_num(chunk: pd.DataFrame) -> pd.Series:
    helpful_num = pd.Series(np.nan, index=chunk.index, dtype="float32")
    level = chunk["helpfulnessLevel"].fillna("")
    helpful_num.loc[level == "NOT_HELPFUL"] = 0.0
    helpful_num.loc[level == "SOMEWHAT_HELPFUL"] = 0.5
    helpful_num.loc[level == "HELPFUL"] = 1.0

    helpful_flag = chunk["helpful"].fillna(0).astype("int8")
    not_helpful_flag = chunk["notHelpful"].fillna(0).astype("int8")
    helpful_num.loc[helpful_flag == 1] = 1.0
    helpful_num.loc[not_helpful_flag == 1] = 0.0
    return helpful_num.astype("float32")


def collect_rating_dataset(
    ratings_files: list[Path],
    participant_ids: np.ndarray,
    misleading_note_ids: set[int],
    chunksize: int,
) -> RatingDataset:
    participant_to_idx = {participant_id: idx for idx, participant_id in enumerate(participant_ids.tolist())}
    note_to_idx: dict[int, int] = {}
    note_ids: list[int] = []

    user_parts: list[np.ndarray] = []
    note_parts: list[np.ndarray] = []
    helpful_parts: list[np.ndarray] = []
    created_parts: list[np.ndarray] = []

    participant_set = set(participant_ids.tolist())
    dtype_map = {
        "noteId": "int64",
        "raterParticipantId": "string",
        "createdAtMillis": "int64",
        "helpfulnessLevel": "string",
        "helpful": "float32",
        "notHelpful": "float32",
    }

    for ratings_file in ratings_files:
        log(f"[ratings] reading {ratings_file.name}")
        reader = pd.read_csv(
            ratings_file,
            sep="\t",
            usecols=[
                "noteId",
                "raterParticipantId",
                "createdAtMillis",
                "helpfulnessLevel",
                "helpful",
                "notHelpful",
            ],
            dtype=dtype_map,
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk_idx, chunk in enumerate(reader, start=1):
            chunk = chunk[chunk["raterParticipantId"].isin(participant_set)]
            chunk = chunk[chunk["noteId"].isin(misleading_note_ids)]
            if chunk.empty:
                continue

            chunk = chunk.copy()
            chunk["helpfulNum"] = compute_helpful_num(chunk)
            chunk = chunk.dropna(subset=["helpfulNum"])
            if chunk.empty:
                continue

            unique_notes = pd.Index(chunk["noteId"].astype("int64").unique())
            missing_notes = [note_id for note_id in unique_notes.tolist() if note_id not in note_to_idx]
            if missing_notes:
                start = len(note_ids)
                note_ids.extend(missing_notes)
                note_to_idx.update(
                    {note_id: start + offset for offset, note_id in enumerate(missing_notes)}
                )

            chunk["user_idx"] = chunk["raterParticipantId"].map(participant_to_idx).astype("int32")
            chunk["note_idx"] = chunk["noteId"].map(note_to_idx).astype("int32")

            user_parts.append(chunk["user_idx"].to_numpy(dtype=np.int32, copy=False))
            note_parts.append(chunk["note_idx"].to_numpy(dtype=np.int32, copy=False))
            helpful_parts.append(chunk["helpfulNum"].to_numpy(dtype=np.float32, copy=False))
            created_parts.append(chunk["createdAtMillis"].to_numpy(dtype=np.int64, copy=False))

            if chunk_idx == 1 or chunk_idx % 10 == 0:
                log(f"[ratings] {ratings_file.name} chunk {chunk_idx} done")

    if not user_parts:
        raise RuntimeError("No ratings remained after filtering to base users and misleading notes.")

    return RatingDataset(
        user_ids=np.asarray(participant_ids, dtype=object),
        note_ids=np.asarray(note_ids, dtype=np.int64),
        user_idx=np.concatenate(user_parts),
        note_idx=np.concatenate(note_parts),
        helpful_num=np.concatenate(helpful_parts),
        created_at_millis=np.concatenate(created_parts),
    )


def apply_min_count_filter(
    dataset: RatingDataset,
    min_ratings_per_rater: int,
    min_raters_per_note: int,
    initial_active_mask: np.ndarray | None = None,
    max_passes: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if initial_active_mask is None:
        active = np.ones(dataset.n_ratings, dtype=bool)
    else:
        active = initial_active_mask.astype(bool, copy=True)
    user_keep = np.ones(dataset.n_users, dtype=bool)
    note_keep = np.ones(dataset.n_notes, dtype=bool)

    for iteration in range(1, max_passes + 1):
        prev_active = active.copy()
        user_counts = np.bincount(dataset.user_idx[active], minlength=dataset.n_users)
        user_keep = user_counts >= min_ratings_per_rater
        active = prev_active & user_keep[dataset.user_idx]

        note_counts = np.bincount(dataset.note_idx[active], minlength=dataset.n_notes)
        note_keep = note_counts >= min_raters_per_note
        active = active & note_keep[dataset.note_idx]

        if np.array_equal(active, prev_active):
            log(f"[filter] min-count filtering converged in {iteration} passes")
            break
    else:
        log(f"[filter] min-count filtering hit max passes={max_passes}")

    final_user_counts = np.bincount(dataset.user_idx[active], minlength=dataset.n_users)
    final_note_counts = np.bincount(dataset.note_idx[active], minlength=dataset.n_notes)
    final_user_keep = final_user_counts > 0
    final_note_keep = final_note_counts > 0
    return active, final_user_keep, final_note_keep, final_user_counts, final_note_counts


def compress_dataset(
    dataset: RatingDataset,
    active_mask: np.ndarray,
    user_keep: np.ndarray,
    note_keep: np.ndarray,
) -> tuple[RatingDataset, np.ndarray, np.ndarray]:
    user_old_idx = np.flatnonzero(user_keep)
    note_old_idx = np.flatnonzero(note_keep)

    user_new_idx = np.full(dataset.n_users, -1, dtype=np.int32)
    note_new_idx = np.full(dataset.n_notes, -1, dtype=np.int32)
    user_new_idx[user_old_idx] = np.arange(len(user_old_idx), dtype=np.int32)
    note_new_idx[note_old_idx] = np.arange(len(note_old_idx), dtype=np.int32)

    active_mask = active_mask & user_keep[dataset.user_idx] & note_keep[dataset.note_idx]

    compressed = RatingDataset(
        user_ids=dataset.user_ids[user_old_idx],
        note_ids=dataset.note_ids[note_old_idx],
        user_idx=user_new_idx[dataset.user_idx[active_mask]],
        note_idx=note_new_idx[dataset.note_idx[active_mask]],
        helpful_num=dataset.helpful_num[active_mask].astype(np.float32, copy=False),
        created_at_millis=dataset.created_at_millis[active_mask].astype(np.int64, copy=False),
    )
    return compressed, user_old_idx, note_old_idx


def load_note_metadata(
    notes_files: list[Path],
    status_file: Path,
    note_ids: np.ndarray,
    participant_ids: np.ndarray,
) -> pd.DataFrame:
    participant_to_idx = {participant_id: idx for idx, participant_id in enumerate(participant_ids.tolist())}
    target_note_ids = set(note_ids.astype(np.int64).tolist())

    notes_frames: list[pd.DataFrame] = []
    for notes_file in notes_files:
        reader = pd.read_csv(
            notes_file,
            sep="\t",
            usecols=["noteId", "noteAuthorParticipantId", "createdAtMillis"],
            dtype={
                "noteId": "int64",
                "noteAuthorParticipantId": "string",
                "createdAtMillis": "int64",
            },
            chunksize=500_000,
            low_memory=False,
        )
        for chunk in reader:
            chunk = chunk[chunk["noteId"].isin(target_note_ids)]
            if not chunk.empty:
                notes_frames.append(chunk)

    if not notes_frames:
        raise RuntimeError("Could not recover note metadata from notes TSVs.")

    notes_df = (
        pd.concat(notes_frames, ignore_index=True)
        .drop_duplicates(subset=["noteId"], keep="first")
        .reset_index(drop=True)
    )

    status_df = pd.read_csv(
        status_file,
        sep="\t",
        usecols=["noteId", "timestampMillisOfLatestNonNMRStatus"],
        dtype={"noteId": "int64", "timestampMillisOfLatestNonNMRStatus": "float64"},
        low_memory=False,
    )
    status_df = (
        status_df[status_df["noteId"].isin(target_note_ids)]
        .drop_duplicates(subset=["noteId"], keep="first")
        .reset_index(drop=True)
    )

    metadata = pd.DataFrame({"noteId": note_ids.astype(np.int64)})
    metadata = metadata.merge(notes_df, on="noteId", how="left")
    metadata = metadata.merge(status_df, on="noteId", how="left")
    if metadata["createdAtMillis"].isna().any():
        missing = int(metadata["createdAtMillis"].isna().sum())
        raise RuntimeError(f"Missing note metadata for {missing} kept notes.")
    metadata["noteAuthorParticipantId"] = metadata["noteAuthorParticipantId"].astype("string")
    metadata["author_user_idx"] = (
        metadata["noteAuthorParticipantId"].map(participant_to_idx).fillna(-1).astype("int32")
    )
    return metadata


def build_sparse_matrix(dataset: RatingDataset) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(
        (dataset.helpful_num, (dataset.user_idx, dataset.note_idx)),
        shape=(dataset.n_users, dataset.n_notes),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


def make_init_vector(size: int, rng: np.random.Generator, init: np.ndarray | None) -> np.ndarray:
    if init is not None:
        return init.astype(np.float64, copy=True)
    return rng.normal(loc=0.0, scale=0.1, size=size).astype(np.float64)


def fit_biased_rank1_als(
    dataset: RatingDataset,
    user_intercept_lambda: float,
    note_intercept_lambda: float,
    user_factor_lambda: float,
    note_factor_lambda: float,
    global_intercept_lambda: float,
    max_iterations: int,
    tol: float,
    random_state: int,
    init_user_intercepts: np.ndarray | None = None,
    init_user_factors: np.ndarray | None = None,
    init_note_intercepts: np.ndarray | None = None,
    init_note_factors: np.ndarray | None = None,
) -> MFResult:
    matrix_csr = build_sparse_matrix(dataset)
    matrix_csc = matrix_csr.tocsc()
    row_idx = dataset.user_idx.astype(np.int64, copy=False)
    col_idx = dataset.note_idx.astype(np.int64, copy=False)
    values = dataset.helpful_num.astype(np.float64, copy=False)

    rng = np.random.default_rng(random_state)
    global_intercept = float(values.mean())
    user_intercepts = make_init_vector(dataset.n_users, rng, init_user_intercepts)
    user_factors = make_init_vector(dataset.n_users, rng, init_user_factors)
    note_intercepts = make_init_vector(dataset.n_notes, rng, init_note_intercepts)
    note_factors = make_init_vector(dataset.n_notes, rng, init_note_factors)
    rmse_history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        pred_without_global = (
            user_intercepts[row_idx]
            + note_intercepts[col_idx]
            + user_factors[row_idx] * note_factors[col_idx]
        )
        global_intercept = float(
            np.sum(values - pred_without_global) / (len(values) + global_intercept_lambda)
        )

        for user_idx_value in range(dataset.n_users):
            start = matrix_csr.indptr[user_idx_value]
            end = matrix_csr.indptr[user_idx_value + 1]
            if start == end:
                user_intercepts[user_idx_value] = 0.0
                user_factors[user_idx_value] = 0.0
                continue

            note_ids = matrix_csr.indices[start:end]
            targets = matrix_csr.data[start:end].astype(np.float64, copy=False)
            note_f = note_factors[note_ids]
            adjusted = targets - global_intercept - note_intercepts[note_ids]

            count = float(len(note_ids))
            sum_q = float(note_f.sum())
            sum_q2 = float(np.dot(note_f, note_f))
            sum_t = float(adjusted.sum())
            sum_qt = float(np.dot(note_f, adjusted))

            system = np.array(
                [
                    [count + user_intercept_lambda, sum_q],
                    [sum_q, sum_q2 + user_factor_lambda],
                ],
                dtype=np.float64,
            )
            target = np.array([sum_t, sum_qt], dtype=np.float64)
            solution = np.linalg.solve(system, target)
            user_intercepts[user_idx_value], user_factors[user_idx_value] = solution

        for note_idx_value in range(dataset.n_notes):
            start = matrix_csc.indptr[note_idx_value]
            end = matrix_csc.indptr[note_idx_value + 1]
            if start == end:
                note_intercepts[note_idx_value] = 0.0
                note_factors[note_idx_value] = 0.0
                continue

            user_ids = matrix_csc.indices[start:end]
            targets = matrix_csc.data[start:end].astype(np.float64, copy=False)
            user_f = user_factors[user_ids]
            adjusted = targets - global_intercept - user_intercepts[user_ids]

            count = float(len(user_ids))
            sum_p = float(user_f.sum())
            sum_p2 = float(np.dot(user_f, user_f))
            sum_t = float(adjusted.sum())
            sum_pt = float(np.dot(user_f, adjusted))

            system = np.array(
                [
                    [count + note_intercept_lambda, sum_p],
                    [sum_p, sum_p2 + note_factor_lambda],
                ],
                dtype=np.float64,
            )
            target = np.array([sum_t, sum_pt], dtype=np.float64)
            solution = np.linalg.solve(system, target)
            note_intercepts[note_idx_value], note_factors[note_idx_value] = solution

        pred = (
            global_intercept
            + user_intercepts[row_idx]
            + note_intercepts[col_idx]
            + user_factors[row_idx] * note_factors[col_idx]
        )
        rmse = float(np.sqrt(np.mean((values - pred) ** 2)))
        rmse_history.append(rmse)
        log(f"[als] iteration={iteration} rmse={rmse:.6f}")

        if len(rmse_history) >= 2:
            prev_rmse = rmse_history[-2]
            improvement = (prev_rmse - rmse) / max(prev_rmse, 1e-12)
            if improvement >= 0 and improvement < tol:
                log(f"[als] converged with relative improvement {improvement:.6e}")
                break

    if np.nanmean(user_factors) > 0:
        user_factors *= -1.0
        note_factors *= -1.0

    return MFResult(
        global_intercept=global_intercept,
        user_intercepts=user_intercepts.astype(np.float32),
        user_factors=user_factors.astype(np.float32),
        note_intercepts=note_intercepts.astype(np.float32),
        note_factors=note_factors.astype(np.float32),
        rmse_history=rmse_history,
    )


def determine_preliminary_note_labels(
    note_intercepts: np.ndarray,
    note_factors: np.ndarray,
    note_rating_counts: np.ndarray,
    min_raters_per_note: int,
    crh_threshold: float,
    crnh_intercept_threshold: float,
    crnh_note_factor_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enough_ratings = note_rating_counts >= min_raters_per_note
    crh = enough_ratings & (note_intercepts >= crh_threshold)
    crnh = enough_ratings & (
        note_intercepts <= (crnh_intercept_threshold + crnh_note_factor_multiplier * np.abs(note_factors))
    )
    return crh, crnh, enough_ratings


def first_k_mask_per_note(note_idx: np.ndarray, created_at_millis: np.ndarray, k: int) -> np.ndarray:
    if len(note_idx) == 0:
        return np.zeros(0, dtype=bool)

    order = np.lexsort((created_at_millis, note_idx))
    sorted_notes = note_idx[order]
    group_change = np.ones(len(sorted_notes), dtype=bool)
    group_change[1:] = sorted_notes[1:] != sorted_notes[:-1]
    group_starts = np.flatnonzero(group_change)
    group_ends = np.append(group_starts[1:], len(sorted_notes))

    ranks = np.empty(len(sorted_notes), dtype=np.int32)
    for start, end in zip(group_starts, group_ends):
        ranks[start:end] = np.arange(end - start, dtype=np.int32)

    keep_sorted = ranks < k
    keep = np.zeros(len(note_idx), dtype=bool)
    keep[order] = keep_sorted
    return keep


def compute_valid_ratings(
    dataset: RatingDataset,
    note_metadata: pd.DataFrame,
) -> np.ndarray:
    note_created = note_metadata["createdAtMillis"].to_numpy(dtype=np.int64)
    note_latest_non_nmr = note_metadata["timestampMillisOfLatestNonNMRStatus"].to_numpy(dtype=np.float64)

    rating_note_created = note_created[dataset.note_idx]
    rating_latest_non_nmr = note_latest_non_nmr[dataset.note_idx]

    within_public_delay = dataset.created_at_millis <= (rating_note_created + PUBLIC_TSV_DELAY_MS)
    before_latest_non_nmr = np.isnan(rating_latest_non_nmr) | (
        dataset.created_at_millis < rating_latest_non_nmr
    )

    historical_note_mask = rating_note_created < DELETED_NOTE_TOMBSTONES_LAUNCH_MS
    valid_historical_mask = first_k_mask_per_note(
        dataset.note_idx[historical_note_mask & within_public_delay],
        dataset.created_at_millis[historical_note_mask & within_public_delay],
        MAX_HISTORICAL_VALID_RATINGS,
    )

    valid = np.zeros(dataset.n_ratings, dtype=bool)
    modern_mask = (~historical_note_mask) & within_public_delay & before_latest_non_nmr
    valid[modern_mask] = True

    historical_subset_idx = np.flatnonzero(historical_note_mask & within_public_delay)
    valid[historical_subset_idx] = valid_historical_mask
    return valid


def compute_helpfulness_scores(
    dataset: RatingDataset,
    note_metadata: pd.DataFrame,
    prelim_result: MFResult,
    note_rating_counts: np.ndarray,
    args: argparse.Namespace,
) -> HelpfulnessResult:
    crh, crnh, _ = determine_preliminary_note_labels(
        prelim_result.note_intercepts,
        prelim_result.note_factors,
        note_rating_counts,
        args.min_raters_per_note,
        args.crh_threshold,
        args.crnh_intercept_threshold,
        args.crnh_note_factor_multiplier,
    )

    valid = compute_valid_ratings(dataset, note_metadata)
    binary_mask = valid & np.isin(dataset.helpful_num, np.array([0.0, 1.0], dtype=np.float32))
    label_mask = crh[dataset.note_idx] | crnh[dataset.note_idx]
    binary_mask &= label_mask

    helpful_mask = dataset.helpful_num == 1.0
    not_helpful_mask = dataset.helpful_num == 0.0
    successful = (crh[dataset.note_idx] & helpful_mask) | (crnh[dataset.note_idx] & not_helpful_mask)
    unsuccessful = (crh[dataset.note_idx] & not_helpful_mask) | (crnh[dataset.note_idx] & helpful_mask)

    rating_count = np.bincount(dataset.user_idx[binary_mask], minlength=dataset.n_users)
    successful_rating_count = np.bincount(
        dataset.user_idx[binary_mask],
        weights=successful[binary_mask].astype(np.float32),
        minlength=dataset.n_users,
    ).astype(np.int64)
    unsuccessful_rating_count = np.bincount(
        dataset.user_idx[binary_mask],
        weights=unsuccessful[binary_mask].astype(np.float32),
        minlength=dataset.n_users,
    ).astype(np.int64)
    rater_agree_ratio = np.divide(
        successful_rating_count,
        rating_count,
        out=np.full(dataset.n_users, np.nan, dtype=np.float32),
        where=rating_count > 0,
    )

    author_idx = note_metadata["author_user_idx"].to_numpy(dtype=np.int32)
    authored_mask = author_idx >= 0
    author_note_counts = np.bincount(author_idx[authored_mask], minlength=dataset.n_users)
    author_crh_counts = np.bincount(
        author_idx[authored_mask],
        weights=crh[authored_mask].astype(np.float32),
        minlength=dataset.n_users,
    )
    author_crnh_counts = np.bincount(
        author_idx[authored_mask],
        weights=crnh[authored_mask].astype(np.float32),
        minlength=dataset.n_users,
    )
    author_note_score_sum = np.bincount(
        author_idx[authored_mask],
        weights=prelim_result.note_intercepts[authored_mask].astype(np.float64),
        minlength=dataset.n_users,
    )

    mean_note_score = np.divide(
        author_note_score_sum,
        author_note_counts,
        out=np.full(dataset.n_users, np.nan, dtype=np.float64),
        where=author_note_counts > 0,
    ).astype(np.float32)
    crh_ratio = np.divide(
        author_crh_counts,
        author_note_counts,
        out=np.full(dataset.n_users, np.nan, dtype=np.float64),
        where=author_note_counts > 0,
    ).astype(np.float32)
    crnh_ratio = np.divide(
        author_crnh_counts,
        author_note_counts,
        out=np.full(dataset.n_users, np.nan, dtype=np.float64),
        where=author_note_counts > 0,
    ).astype(np.float32)
    crh_crnh_ratio_difference = (crh_ratio - 5.0 * crnh_ratio).astype(np.float32)

    author_ok = (
        ((crh_crnh_ratio_difference >= args.min_crh_vs_crnh_ratio) & (mean_note_score >= args.min_mean_note_score))
        | (np.isnan(crh_crnh_ratio_difference) & np.isnan(mean_note_score))
        | (np.isnan(crh_crnh_ratio_difference) & (mean_note_score >= args.min_mean_note_score))
    )
    rater_ok = rater_agree_ratio >= args.min_rater_agree_ratio
    above_threshold = author_ok & rater_ok

    return HelpfulnessResult(
        participant_ids=dataset.user_ids.astype(object),
        rater_agree_ratio=rater_agree_ratio.astype(np.float32),
        rating_count=rating_count.astype(np.int64),
        successful_rating_count=successful_rating_count.astype(np.int64),
        unsuccessful_rating_count=unsuccessful_rating_count.astype(np.int64),
        mean_note_score=mean_note_score.astype(np.float32),
        crh_ratio=crh_ratio.astype(np.float32),
        crnh_ratio=crnh_ratio.astype(np.float32),
        crh_crnh_ratio_difference=crh_crnh_ratio_difference.astype(np.float32),
        above_helpfulness_threshold=above_threshold.astype(bool),
    )


def helpfulness_to_frame(result: HelpfulnessResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participantId": result.participant_ids,
            "bw_rater_agree_ratio": result.rater_agree_ratio,
            "bw_valid_rating_count": result.rating_count,
            "bw_successful_rating_count": result.successful_rating_count,
            "bw_unsuccessful_rating_count": result.unsuccessful_rating_count,
            "bw_mean_note_score": result.mean_note_score,
            "bw_crh_ratio": result.crh_ratio,
            "bw_crnh_ratio": result.crnh_ratio,
            "bw_crh_crnh_ratio_difference": result.crh_crnh_ratio_difference,
            "bw_helpfulness_pass": result.above_helpfulness_threshold.astype(int),
        }
    )


def build_note_params_frame(
    dataset: RatingDataset,
    note_metadata: pd.DataFrame,
    result: MFResult,
    note_rating_counts: np.ndarray,
    args: argparse.Namespace,
    prefix: str,
) -> pd.DataFrame:
    crh, crnh, enough_ratings = determine_preliminary_note_labels(
        result.note_intercepts,
        result.note_factors,
        note_rating_counts,
        args.min_raters_per_note,
        args.crh_threshold,
        args.crnh_intercept_threshold,
        args.crnh_note_factor_multiplier,
    )
    note_status = np.where(crh, "CURRENTLY_RATED_HELPFUL", np.where(crnh, "CURRENTLY_RATED_NOT_HELPFUL", "NEEDS_MORE_RATINGS"))
    return pd.DataFrame(
        {
            "noteId": dataset.note_ids.astype(np.int64),
            "noteAuthorParticipantId": note_metadata["noteAuthorParticipantId"].astype("string"),
            "createdAtMillis": note_metadata["createdAtMillis"].astype("int64"),
            f"{prefix}_note_intercept": result.note_intercepts.astype(np.float32),
            f"{prefix}_note_factor_1": result.note_factors.astype(np.float32),
            f"{prefix}_rating_count": note_rating_counts.astype(np.int64),
            f"{prefix}_enough_ratings": enough_ratings.astype(int),
            f"{prefix}_status": note_status,
        }
    )


def build_user_params_frame(
    participant_ids: np.ndarray,
    result: MFResult,
    prefix: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participantId": participant_ids.astype(object),
            f"{prefix}_rater_intercept": result.user_intercepts.astype(np.float32),
            f"{prefix}_rater_factor_1": result.user_factors.astype(np.float32),
        }
    )


def standardize(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    matrix = df[feature_columns].astype("float64").to_numpy()
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0
    return (matrix - means) / stds


def evaluate_k_values(
    matrix: np.ndarray,
    k_values: list[int],
    random_state: int,
    silhouette_sample_size: int,
) -> tuple[pd.DataFrame, int, dict[int, np.ndarray]]:
    if len(matrix) < 2:
        raise RuntimeError("At least two users are required for clustering.")

    valid_k_values = sorted({int(k) for k in k_values if 2 <= int(k) < len(matrix)})
    if not valid_k_values:
        raise RuntimeError("No valid K values remain after accounting for the number of clustered users.")

    rng = np.random.default_rng(random_state)
    sample_size = min(silhouette_sample_size, len(matrix))
    sample_idx = rng.choice(len(matrix), size=sample_size, replace=False)
    sample_matrix = matrix[sample_idx]

    rows: list[dict[str, float | int]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in valid_k_values:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=4096,
            n_init=10,
            max_iter=200,
        )
        labels = model.fit_predict(matrix)
        labels_by_k[k] = labels
        sample_labels = labels[sample_idx]

        silhouette = float(silhouette_score(sample_matrix, sample_labels))
        calinski = float(calinski_harabasz_score(sample_matrix, sample_labels))
        davies = float(davies_bouldin_score(sample_matrix, sample_labels))
        rows.append(
            {
                "k": int(k),
                "silhouette": silhouette,
                "calinski_harabasz": calinski,
                "davies_bouldin": davies,
                "inertia": float(model.inertia_),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_k = int(metrics_df.sort_values(["silhouette", "calinski_harabasz"], ascending=[False, False]).iloc[0]["k"])
    return metrics_df, best_k, labels_by_k


def summarize_clusters(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    summary = df.groupby("cluster", sort=True).agg(
        users=("participantId", "size"),
        avg_bw_final_rater_intercept=("bw_final_rater_intercept", "mean"),
        avg_bw_final_rater_factor_1=("bw_final_rater_factor_1", "mean"),
        avg_bw_rater_agree_ratio=("bw_rater_agree_ratio", "mean"),
        avg_bw_mean_note_score=("bw_mean_note_score", "mean"),
        avg_bw_helpfulness_pass=("bw_helpfulness_pass", "mean"),
        avg_share_helpful=("share_helpful", "mean"),
        avg_share_not_helpful=("share_not_helpful", "mean"),
        avg_notes_authored=("notes_authored", "mean"),
    )
    centers = df.groupby("cluster", sort=True)[feature_columns].mean()
    summary = summary.join(centers)

    if "old_cluster" in df.columns:
        dominant_old_cluster = (
            df.groupby("cluster", sort=True)["old_cluster"]
            .agg(lambda values: int(pd.Series(values).mode(dropna=True).iloc[0]))
            .rename("dominant_old_cluster")
        )
        summary = summary.join(dominant_old_cluster)
    return summary.reset_index()


def build_old_cluster_crosstab(df: pd.DataFrame) -> pd.DataFrame | None:
    if "old_cluster" not in df.columns:
        return None
    return pd.crosstab(df["cluster"], df["old_cluster"], dropna=False).reset_index()


def save_outputs(
    enriched: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    k_metrics: pd.DataFrame,
    helpfulness_df: pd.DataFrame,
    prelim_note_df: pd.DataFrame,
    final_note_df: pd.DataFrame,
    prelim_user_df: pd.DataFrame,
    final_user_df: pd.DataFrame,
    final_user_counts_df: pd.DataFrame,
    final_note_counts_df: pd.DataFrame,
    crosstab: pd.DataFrame | None,
    feature_columns: list[str],
    prelim_result: MFResult,
    final_result: MFResult,
    prelim_dataset: RatingDataset,
    final_dataset: RatingDataset,
    best_k: int,
    base_user_count: int,
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    enriched.to_csv(args.output_dir / "user_features_with_mf_clusters.csv", index=False)
    cluster_summary.to_csv(args.output_dir / "cluster_summary.csv", index=False)
    k_metrics.to_csv(args.output_dir / "k_selection_metrics.csv", index=False)
    helpfulness_df.to_csv(args.output_dir / "rater_helpfulness_scores.csv", index=False)
    prelim_note_df.to_csv(args.output_dir / "note_params_preliminary.csv", index=False)
    final_note_df.to_csv(args.output_dir / "note_params_final.csv", index=False)
    prelim_user_df.to_csv(args.output_dir / "rater_params_preliminary.csv", index=False)
    final_user_df.to_csv(args.output_dir / "rater_params_final.csv", index=False)
    final_user_counts_df.to_csv(args.output_dir / "kept_user_rating_counts.csv", index=False)
    final_note_counts_df.to_csv(args.output_dir / "kept_note_rating_counts.csv", index=False)
    if crosstab is not None:
        crosstab.to_csv(args.output_dir / "old_vs_mf_cluster_crosstab.csv", index=False)

    metadata = {
        "data_root": str(args.data_root.resolve()),
        "base_features": str(args.base_features.resolve()),
        "users_in_base_universe": int(base_user_count),
        "users_in_preliminary_universe": int(len(prelim_user_df)),
        "users_after_helpfulness_filter": int(len(final_user_df)),
        "ratings_before_helpfulness_filter": int(prelim_dataset.n_ratings),
        "ratings_after_helpfulness_filter": int(final_dataset.n_ratings),
        "notes_before_helpfulness_filter": int(prelim_dataset.n_notes),
        "notes_after_helpfulness_filter": int(final_dataset.n_notes),
        "feature_columns": feature_columns,
        "k_values": [int(value) for value in args.k_values.split(",") if value.strip()],
        "best_k": int(best_k),
        "min_ratings_per_rater": int(args.min_ratings_per_rater),
        "min_raters_per_note": int(args.min_raters_per_note),
        "als_iterations": int(args.als_iterations),
        "als_tolerance": float(args.als_tol),
        "l2_penalties": {
            "global_intercept": float(args.global_intercept_lambda),
            "user_intercept": float(args.user_intercept_lambda),
            "note_intercept": float(args.note_intercept_lambda),
            "user_factor": float(args.user_factor_lambda),
            "note_factor": float(args.note_factor_lambda),
        },
        "thresholds": {
            "crh_threshold": float(args.crh_threshold),
            "crnh_intercept_threshold": float(args.crnh_intercept_threshold),
            "crnh_note_factor_multiplier": float(args.crnh_note_factor_multiplier),
            "min_mean_note_score": float(args.min_mean_note_score),
            "min_crh_vs_crnh_ratio": float(args.min_crh_vs_crnh_ratio),
            "min_rater_agree_ratio": float(args.min_rater_agree_ratio),
        },
        "preliminary_global_intercept": float(prelim_result.global_intercept),
        "final_global_intercept": float(final_result.global_intercept),
        "preliminary_rmse_history": [float(value) for value in prelim_result.rmse_history],
        "final_rmse_history": [float(value) for value in final_result.rmse_history],
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_features = load_base_features(args.base_features, args.max_base_users)
    participant_ids = base_features["participantId"].astype("string").to_numpy()

    notes_files = sorted((args.data_root / "notes").glob("notes-*.tsv"))
    ratings_files = sorted((args.data_root / "noteRatings").glob("ratings-*.tsv"))
    if args.max_ratings_files is not None:
        ratings_files = ratings_files[: args.max_ratings_files]
    status_file = args.data_root / "noteStatusHistory" / "noteStatusHistory-00000.tsv"
    if not notes_files or not ratings_files or not status_file.exists():
        raise RuntimeError("Expected notes, noteRatings, and noteStatusHistory TSVs under the data root.")

    misleading_note_ids = load_misleading_notes(notes_files)
    raw_dataset = collect_rating_dataset(
        ratings_files,
        participant_ids,
        misleading_note_ids,
        args.ratings_chunksize,
    )
    log(
        "[raw] users in universe="
        f"{raw_dataset.n_users} | notes rated={raw_dataset.n_notes} | ratings kept={raw_dataset.n_ratings}"
    )

    prelim_active, prelim_user_keep, prelim_note_keep, prelim_user_counts, prelim_note_counts = apply_min_count_filter(
        raw_dataset,
        args.min_ratings_per_rater,
        args.min_raters_per_note,
    )
    prelim_dataset, prelim_user_old_idx, prelim_note_old_idx = compress_dataset(
        raw_dataset,
        prelim_active,
        prelim_user_keep,
        prelim_note_keep,
    )
    if prelim_dataset.n_ratings == 0:
        raise RuntimeError("No ratings remained after the preliminary min-count filter.")

    prelim_base_features = (
        base_features.iloc[prelim_user_old_idx].copy().reset_index(drop=True)
    )
    prelim_note_metadata = load_note_metadata(
        notes_files,
        status_file,
        prelim_dataset.note_ids,
        prelim_dataset.user_ids,
    )
    prelim_note_rating_counts = np.bincount(prelim_dataset.note_idx, minlength=prelim_dataset.n_notes).astype(np.int64)

    log(
        "[prelim] users="
        f"{prelim_dataset.n_users} | notes={prelim_dataset.n_notes} | ratings={prelim_dataset.n_ratings}"
    )
    prelim_result = fit_biased_rank1_als(
        prelim_dataset,
        user_intercept_lambda=args.user_intercept_lambda,
        note_intercept_lambda=args.note_intercept_lambda,
        user_factor_lambda=args.user_factor_lambda,
        note_factor_lambda=args.note_factor_lambda,
        global_intercept_lambda=args.global_intercept_lambda,
        max_iterations=args.als_iterations,
        tol=args.als_tol,
        random_state=args.random_state,
    )

    helpfulness_result = compute_helpfulness_scores(
        prelim_dataset,
        prelim_note_metadata,
        prelim_result,
        prelim_note_rating_counts,
        args,
    )
    helpfulness_df = helpfulness_to_frame(helpfulness_result)

    helpful_active = helpfulness_result.above_helpfulness_threshold[prelim_dataset.user_idx]
    if not helpful_active.any():
        raise RuntimeError("No ratings remained after contributor helpfulness filtering.")

    final_active, final_user_keep, final_note_keep, final_user_counts, final_note_counts = apply_min_count_filter(
        prelim_dataset,
        args.min_ratings_per_rater,
        args.min_raters_per_note,
        initial_active_mask=helpful_active,
        max_passes=10,
    )

    final_dataset, final_user_old_idx, final_note_old_idx = compress_dataset(
        prelim_dataset,
        final_active,
        final_user_keep,
        final_note_keep,
    )
    if final_dataset.n_ratings == 0:
        raise RuntimeError("No ratings remained after the final helpfulness filter.")

    final_note_metadata = prelim_note_metadata.iloc[final_note_old_idx].reset_index(drop=True).copy()
    final_base_features = prelim_base_features.iloc[final_user_old_idx].copy().reset_index(drop=True)

    init_user_intercepts = prelim_result.user_intercepts[final_user_old_idx]
    init_user_factors = prelim_result.user_factors[final_user_old_idx]
    init_note_intercepts = prelim_result.note_intercepts[final_note_old_idx]
    init_note_factors = prelim_result.note_factors[final_note_old_idx]

    log(
        "[final] users="
        f"{final_dataset.n_users} | notes={final_dataset.n_notes} | ratings={final_dataset.n_ratings}"
    )
    final_result = fit_biased_rank1_als(
        final_dataset,
        user_intercept_lambda=args.user_intercept_lambda,
        note_intercept_lambda=args.note_intercept_lambda,
        user_factor_lambda=args.user_factor_lambda,
        note_factor_lambda=args.note_factor_lambda,
        global_intercept_lambda=args.global_intercept_lambda,
        max_iterations=args.als_iterations,
        tol=args.als_tol,
        random_state=args.random_state,
        init_user_intercepts=init_user_intercepts,
        init_user_factors=init_user_factors,
        init_note_intercepts=init_note_intercepts,
        init_note_factors=init_note_factors,
    )

    prelim_user_df = build_user_params_frame(prelim_dataset.user_ids, prelim_result, "bw_pre")
    final_user_df = build_user_params_frame(final_dataset.user_ids, final_result, "bw_final")
    prelim_note_df = build_note_params_frame(
        prelim_dataset,
        prelim_note_metadata,
        prelim_result,
        prelim_note_rating_counts,
        args,
        "bw_pre",
    )
    final_note_rating_counts = np.bincount(final_dataset.note_idx, minlength=final_dataset.n_notes).astype(np.int64)
    final_note_df = build_note_params_frame(
        final_dataset,
        final_note_metadata,
        final_result,
        final_note_rating_counts,
        args,
        "bw_final",
    )

    enriched = final_base_features.merge(final_user_df, on="participantId", how="left")
    enriched = enriched.merge(
        prelim_user_df[["participantId", "bw_pre_rater_intercept", "bw_pre_rater_factor_1"]],
        on="participantId",
        how="left",
    )
    enriched = enriched.merge(helpfulness_df, on="participantId", how="left")

    feature_columns = [
        "bw_final_rater_intercept",
        "bw_final_rater_factor_1",
        "bw_rater_agree_ratio",
        "bw_mean_note_score",
        "bw_crh_crnh_ratio_difference",
    ]
    cluster_matrix = standardize(enriched, feature_columns)
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    k_metrics, best_k, labels_by_k = evaluate_k_values(
        cluster_matrix,
        k_values,
        args.random_state,
        args.silhouette_sample_size,
    )

    enriched["cluster"] = labels_by_k[best_k].astype(int)
    cluster_summary = summarize_clusters(enriched, feature_columns)
    crosstab = build_old_cluster_crosstab(enriched)

    final_user_counts_df = pd.DataFrame(
        {
            "participantId": final_dataset.user_ids.astype(object),
            "rating_count": np.bincount(final_dataset.user_idx, minlength=final_dataset.n_users).astype(np.int64),
        }
    )
    final_note_counts_df = pd.DataFrame(
        {
            "noteId": final_dataset.note_ids.astype(np.int64),
            "rating_count": final_note_rating_counts.astype(np.int64),
        }
    )

    save_outputs(
        enriched=enriched,
        cluster_summary=cluster_summary,
        k_metrics=k_metrics,
        helpfulness_df=helpfulness_df,
        prelim_note_df=prelim_note_df,
        final_note_df=final_note_df,
        prelim_user_df=prelim_user_df,
        final_user_df=final_user_df,
        final_user_counts_df=final_user_counts_df,
        final_note_counts_df=final_note_counts_df,
        crosstab=crosstab,
        feature_columns=feature_columns,
        prelim_result=prelim_result,
        final_result=final_result,
        prelim_dataset=prelim_dataset,
        final_dataset=final_dataset,
        best_k=best_k,
        base_user_count=len(base_features),
        args=args,
    )

    log(f"Finished Birdwatch-style matrix-factorized clustering for {len(enriched)} users.")
    log(f"Best K by silhouette: {best_k}")
    log(
        f"Preliminary ratings={prelim_dataset.n_ratings} | Final ratings={final_dataset.n_ratings} | "
        f"Users after helpfulness filter={final_dataset.n_users}"
    )
    log(k_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
