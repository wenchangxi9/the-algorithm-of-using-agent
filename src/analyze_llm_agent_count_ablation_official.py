from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import cluster_communitynotes_users_matrix_factorized as mf_mod
from cluster_communitynotes_users_matrix_factorized import (
    MFResult,
    RatingDataset,
    apply_min_count_filter,
    compress_dataset,
    determine_preliminary_note_labels,
    fit_biased_rank1_als,
)


CRH = "CURRENTLY_RATED_HELPFUL"
CRNH = "CURRENTLY_RATED_NOT_HELPFUL"
NMR = "NEEDS_MORE_RATINGS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate agent-count curves using an official-style Birdwatch aggregation rule "
            "instead of majority vote."
        )
    )
    parser.add_argument(
        "--votes-csv",
        type=Path,
        default=Path("artifacts/llm_runs/mf_continuous_n024_gpt54nano_20260507_run1/agent_votes.csv"),
    )
    parser.add_argument(
        "--notes-csv",
        type=Path,
        default=Path("artifacts/llm_runs/mf_continuous_n024_gpt54nano_20260507_run1/note_predictions.csv"),
    )
    parser.add_argument(
        "--single-agent-metadata",
        type=Path,
        default=Path(""),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/official_style_results/mf_continuous_n024_gpt54nano_20260507_run1"),
    )
    parser.add_argument(
        "--agent-counts",
        type=str,
        default="1,3,6,12,18,24,36,48,60,72",
    )
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ratings-per-rater", type=int, default=10)
    parser.add_argument("--min-raters-per-note", type=int, default=5)
    parser.add_argument("--als-iterations", type=int, default=12)
    parser.add_argument("--als-tol", type=float, default=1e-4)
    parser.add_argument("--global-intercept-lambda", type=float, default=0.15)
    parser.add_argument("--user-intercept-lambda", type=float, default=0.15)
    parser.add_argument("--note-intercept-lambda", type=float, default=0.15)
    parser.add_argument("--user-factor-lambda", type=float, default=0.03)
    parser.add_argument("--note-factor-lambda", type=float, default=0.03)
    parser.add_argument("--crh-threshold", type=float, default=0.40)
    parser.add_argument("--crnh-intercept-threshold", type=float, default=-0.05)
    parser.add_argument("--crnh-note-factor-multiplier", type=float, default=-0.80)
    parser.add_argument("--min-rater-agree-ratio", type=float, default=0.66)
    parser.add_argument(
        "--verbose-mf",
        action="store_true",
        help="Print ALS and filtering progress from the imported matrix-factorization module.",
    )
    return parser.parse_args()


def load_single_agent_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return float(metadata["metrics"]["accuracy_vs_status"])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
        "recall_not_helpful": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "recall_helpful": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def summarize_repeats(repeat_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "recall_not_helpful",
        "recall_helpful",
        "note_coverage",
        "resolved_share",
    ]
    rows: list[dict[str, float | int]] = []
    for agent_count, group in repeat_df.groupby("agent_count"):
        row: dict[str, float | int] = {"agent_count": int(agent_count), "repeats": int(len(group))}
        for col in metric_cols:
            values = group[col].astype(float)
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=0))
            row[f"{col}_p10"] = float(values.quantile(0.10))
            row[f"{col}_p50"] = float(values.quantile(0.50))
            row[f"{col}_p90"] = float(values.quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("agent_count").reset_index(drop=True)


def plot_core_metrics(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9.5, 5.5))
    x = summary_df["agent_count"].to_numpy()
    for col, label, color in [
        ("accuracy_mean", "Accuracy", "#1f6f8b"),
        ("balanced_accuracy_mean", "Balanced accuracy", "#2f8f46"),
        ("f1_mean", "F1", "#b84a39"),
    ]:
        plt.plot(x, summary_df[col].to_numpy() * 100, marker="o", linewidth=2.2, label=label, color=color)
    plt.title("Official-Style MF Aggregation: Core Metrics")
    plt.xlabel("Number of LLM persona agents")
    plt.ylabel("Metric on resolved notes (%)")
    plt.xticks(x)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_coverage(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9.5, 5.5))
    x = summary_df["agent_count"].to_numpy()
    plt.plot(x, summary_df["note_coverage_mean"].to_numpy() * 100, marker="o", linewidth=2.2, color="#7a4fb5", label="Coverage")
    plt.plot(x, summary_df["resolved_share_mean"].to_numpy() * 100, marker="o", linewidth=2.2, color="#cc7a00", label="Resolved share")
    plt.title("Official-Style MF Aggregation: Coverage")
    plt.xlabel("Number of LLM persona agents")
    plt.ylabel("Share of all notes (%)")
    plt.xticks(x)
    plt.ylim(0, 105)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_dataset(subset_votes: pd.DataFrame, note_ids: np.ndarray, agent_ids: np.ndarray) -> RatingDataset:
    note_to_idx = {note_id: idx for idx, note_id in enumerate(note_ids.tolist())}
    user_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids.tolist())}
    ordered = subset_votes.copy()
    ordered["user_idx"] = ordered["agent_id"].map(user_to_idx).astype("int32")
    ordered["note_idx"] = ordered["noteId"].map(note_to_idx).astype("int32")
    return RatingDataset(
        user_ids=agent_ids.astype(object),
        note_ids=note_ids.astype(object),
        user_idx=ordered["user_idx"].to_numpy(dtype=np.int32),
        note_idx=ordered["note_idx"].to_numpy(dtype=np.int32),
        helpful_num=ordered["helpful_num"].to_numpy(dtype=np.float32),
        created_at_millis=np.zeros(len(ordered), dtype=np.int64),
    )


def fit_mf(dataset: RatingDataset, args: argparse.Namespace, seed: int) -> MFResult:
    return fit_biased_rank1_als(
        dataset=dataset,
        user_intercept_lambda=args.user_intercept_lambda,
        note_intercept_lambda=args.note_intercept_lambda,
        user_factor_lambda=args.user_factor_lambda,
        note_factor_lambda=args.note_factor_lambda,
        global_intercept_lambda=args.global_intercept_lambda,
        max_iterations=args.als_iterations,
        tol=args.als_tol,
        random_state=seed,
    )


def compute_rater_agree_ratio(
    dataset: RatingDataset,
    note_crh: np.ndarray,
    note_crnh: np.ndarray,
) -> np.ndarray:
    label_mask = note_crh[dataset.note_idx] | note_crnh[dataset.note_idx]
    helpful_mask = dataset.helpful_num == 1.0
    not_helpful_mask = dataset.helpful_num == 0.0
    successful = (note_crh[dataset.note_idx] & helpful_mask) | (note_crnh[dataset.note_idx] & not_helpful_mask)
    rating_count = np.bincount(dataset.user_idx[label_mask], minlength=dataset.n_users)
    successful_count = np.bincount(
        dataset.user_idx[label_mask],
        weights=successful[label_mask].astype(np.float32),
        minlength=dataset.n_users,
    )
    return np.divide(
        successful_count,
        rating_count,
        out=np.full(dataset.n_users, np.nan, dtype=np.float32),
        where=rating_count > 0,
    )


def official_style_predict(
    subset_votes: pd.DataFrame,
    note_ids: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    agent_ids = np.array(sorted(subset_votes["agent_id"].astype(str).unique()), dtype=object)
    if len(agent_ids) == 0:
        return np.full(len(note_ids), -1, dtype=np.int8), np.zeros(len(note_ids), dtype=bool)

    dataset = build_dataset(subset_votes, note_ids, agent_ids)
    active_mask, user_keep, note_keep, _, note_rating_counts = apply_min_count_filter(
        dataset,
        min_ratings_per_rater=args.min_ratings_per_rater,
        min_raters_per_note=args.min_raters_per_note,
    )
    if not note_keep.any():
        return np.full(len(note_ids), -1, dtype=np.int8), np.zeros(len(note_ids), dtype=bool)

    prelim_dataset, _, kept_note_old_idx = compress_dataset(dataset, active_mask, user_keep, note_keep)
    prelim_note_full_keep = np.zeros(dataset.n_notes, dtype=bool)
    prelim_note_full_keep[kept_note_old_idx] = True
    prelim_counts_kept = note_rating_counts[kept_note_old_idx]
    prelim_result = fit_mf(prelim_dataset, args, seed)
    prelim_crh, prelim_crnh, _ = determine_preliminary_note_labels(
        prelim_result.note_intercepts,
        prelim_result.note_factors,
        prelim_counts_kept,
        args.min_raters_per_note,
        args.crh_threshold,
        args.crnh_intercept_threshold,
        args.crnh_note_factor_multiplier,
    )

    # Expand preliminary labels back to the full selected-note universe.
    prelim_crh_full = np.zeros(dataset.n_notes, dtype=bool)
    prelim_crnh_full = np.zeros(dataset.n_notes, dtype=bool)
    prelim_crh_full[kept_note_old_idx] = prelim_crh
    prelim_crnh_full[kept_note_old_idx] = prelim_crnh

    rater_agree_ratio = compute_rater_agree_ratio(dataset, prelim_crh_full, prelim_crnh_full)
    helpful_raters = np.nan_to_num(rater_agree_ratio, nan=-1.0) >= args.min_rater_agree_ratio
    if not helpful_raters.any():
        return np.full(len(note_ids), -1, dtype=np.int8), np.zeros(len(note_ids), dtype=bool)

    final_active_mask, final_user_keep, final_note_keep, _, final_note_counts = apply_min_count_filter(
        dataset,
        min_ratings_per_rater=args.min_ratings_per_rater,
        min_raters_per_note=args.min_raters_per_note,
        initial_active_mask=helpful_raters[dataset.user_idx],
    )
    if not final_note_keep.any():
        return np.full(len(note_ids), -1, dtype=np.int8), np.zeros(len(note_ids), dtype=bool)

    final_dataset, _, final_note_old_idx = compress_dataset(dataset, final_active_mask, final_user_keep, final_note_keep)
    final_counts_kept = final_note_counts[final_note_old_idx]
    final_result = fit_mf(final_dataset, args, seed + 1)
    final_crh, final_crnh, enough = determine_preliminary_note_labels(
        final_result.note_intercepts,
        final_result.note_factors,
        final_counts_kept,
        args.min_raters_per_note,
        args.crh_threshold,
        args.crnh_intercept_threshold,
        args.crnh_note_factor_multiplier,
    )

    pred = np.full(dataset.n_notes, -1, dtype=np.int8)
    pred[final_note_old_idx[final_crh]] = 1
    pred[final_note_old_idx[final_crnh]] = 0
    resolved = np.zeros(dataset.n_notes, dtype=bool)
    resolved[final_note_old_idx] = enough
    resolved &= pred >= 0
    return pred, resolved


def main() -> None:
    args = parse_args()
    if not args.verbose_mf:
        mf_mod.log = lambda _message: None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    agent_counts = [int(v.strip()) for v in args.agent_counts.split(",") if v.strip()]

    votes = pd.read_csv(args.votes_csv.resolve(), low_memory=False)
    notes = pd.read_csv(args.notes_csv.resolve(), low_memory=False)
    votes["noteId"] = votes["noteId"].astype(str)
    votes["agent_id"] = votes["agent_id"].astype(str)
    valid_votes = votes[votes["predicted_label"].isin([0, 1])].copy()
    valid_votes["helpful_num"] = valid_votes["predicted_label"].astype(float)

    note_table = notes[["noteId", "true_label"]].drop_duplicates().copy()
    note_table["noteId"] = note_table["noteId"].astype(str)
    note_ids = note_table["noteId"].to_numpy(dtype=object)
    true_lookup = note_table.set_index("noteId")["true_label"].astype(int)

    available_agents = np.array(sorted(valid_votes["agent_id"].unique().tolist()), dtype=object)
    repeat_rows: list[dict[str, float | int | str]] = []

    for count in agent_counts:
        if count < 1 or count > len(available_agents):
            raise ValueError(f"Invalid agent count {count}; available agents={len(available_agents)}")
        repeats = 1 if count == len(available_agents) else args.repeats
        for repeat in range(repeats):
            selected_agents = (
                available_agents
                if count == len(available_agents)
                else rng.choice(available_agents, size=count, replace=False)
            )
            subset = valid_votes[valid_votes["agent_id"].isin(selected_agents)].copy()
            pred, resolved = official_style_predict(subset, note_ids, args, seed=args.seed + repeat + count * 1000)
            pred_mask = pred >= 0
            y_true = true_lookup.loc[note_ids[pred_mask]].to_numpy(dtype=int)
            y_pred = pred[pred_mask].astype(int)
            metrics = compute_metrics(y_true, y_pred)
            metrics.update(
                {
                    "agent_count": count,
                    "repeat": repeat,
                    "note_coverage": float(pred_mask.mean()),
                    "resolved_share": float(resolved.mean()),
                    "selected_agents": ",".join(selected_agents.tolist()),
                }
            )
            repeat_rows.append(metrics)

    repeat_df = pd.DataFrame(repeat_rows)
    summary_df = summarize_repeats(repeat_df)
    single_agent_accuracy = load_single_agent_accuracy(args.single_agent_metadata)

    repeat_df.to_csv(output_dir / "agent_count_official_repeats.csv", index=False)
    summary_df.to_csv(output_dir / "agent_count_official_summary.csv", index=False)
    plot_core_metrics(summary_df, output_dir / "agent_count_official_core_metric_curves.png")
    plot_coverage(summary_df, output_dir / "agent_count_official_coverage_curves.png")

    metadata = {
        "votes_csv": str(args.votes_csv),
        "notes_csv": str(args.notes_csv),
        "notes": int(len(note_ids)),
        "available_agents": int(len(available_agents)),
        "agent_counts": agent_counts,
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "single_global_gpt_agent_accuracy": single_agent_accuracy,
        "aggregation": "official_style_birdwatch_rank1_mf_with_rater_agreement_filter",
        "summary": summary_df.to_dict(orient="records"),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
