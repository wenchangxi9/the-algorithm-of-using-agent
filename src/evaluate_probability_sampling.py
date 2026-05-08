from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import analyze_llm_agent_count_ablation_official as official_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate probabilistic multi-sampling baselines for MF-continuous agents. "
            "Each LLM judgment is converted into P(Helpful) using its confidence, then "
            "synthetic panels are sampled under fixed agent budgets."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--agent-counts", type=str, default="12,24,36,48")
    parser.add_argument("--model-tag", type=str, default="gpt54nano")
    parser.add_argument("--date-tag", type=str, default="20260507")
    parser.add_argument("--run-tag", type=str, default="run1")
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument(
        "--resolved-repeats",
        type=int,
        default=300,
        help=(
            "Monte Carlo repeats for official-style resolved evaluation. "
            "This is separate because each repeat fits a rank-1 MF resolver."
        ),
    )
    parser.add_argument(
        "--calibrated-repeats",
        type=int,
        default=500,
        help="Monte Carlo repeats for calibrated selective screening over sampled helpful-share scores.",
    )
    parser.add_argument("--calibrated-coverage-target", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=Path("artifacts/comparison_tables/probability_sampling_summary.csv"),
    )
    parser.add_argument(
        "--output-repeats-csv",
        type=Path,
        default=Path("artifacts/comparison_tables/probability_sampling_repeats.csv"),
    )
    return parser.parse_args()


def parse_counts(text: str) -> list[int]:
    counts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not counts:
        raise ValueError("No agent counts were provided.")
    return counts


def label_probability(labels: pd.Series, confidence: pd.Series) -> pd.Series:
    y = pd.to_numeric(labels, errors="coerce")
    conf = pd.to_numeric(confidence, errors="coerce").clip(0, 100) / 100.0
    p = pd.Series(np.nan, index=labels.index, dtype=float)
    valid = y.isin([0, 1]) & conf.notna()
    p.loc[valid & y.eq(1)] = conf.loc[valid & y.eq(1)]
    p.loc[valid & y.eq(0)] = 1.0 - conf.loc[valid & y.eq(0)]
    return p.clip(0.0, 1.0)


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    recall_not_helpful = tn / (tn + fp) if (tn + fp) else 0.0
    recall_helpful = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "balanced_accuracy": float((recall_not_helpful + recall_helpful) / 2.0),
        "recall_not_helpful": float(recall_not_helpful),
        "recall_helpful": float(recall_helpful),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(len(arr)) if len(arr) else 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "ci95_low": float(arr.mean() - half_width),
        "ci95_high": float(arr.mean() + half_width),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def load_inputs(repo_root: Path, agent_count: int, model_tag: str, date_tag: str, run_tag: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variant = f"mf_continuous_n{agent_count:03d}"
    run_dir = repo_root / "artifacts" / "llm_runs" / f"{variant}_{model_tag}_{date_tag}_{run_tag}"
    roster_path = repo_root / "artifacts" / "agent_variants" / variant / "agent_roster.csv"
    notes_path = run_dir / "note_predictions.csv"
    votes_path = run_dir / "agent_votes.csv"
    if not notes_path.exists():
        raise FileNotFoundError(notes_path)
    if not votes_path.exists():
        raise FileNotFoundError(votes_path)
    if not roster_path.exists():
        raise FileNotFoundError(roster_path)
    return (
        pd.read_csv(notes_path, low_memory=False),
        pd.read_csv(votes_path, low_memory=False),
        pd.read_csv(roster_path, low_memory=False),
    )


def fill_probability_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    missing_cells = int(matrix.isna().sum().sum())
    note_means = matrix.mean(axis=1, skipna=True)
    filled = matrix.T.fillna(note_means).T.fillna(0.5)
    return filled.clip(0.0, 1.0), missing_cells


def parent_probability_matrix(
    notes: pd.DataFrame,
    votes: pd.DataFrame,
    roster: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    mapping = roster[["cluster", "parent_cluster", "agent_count"]].copy()
    mapping["cluster"] = pd.to_numeric(mapping["cluster"], errors="coerce").astype("Int64")
    mapping["parent_cluster"] = pd.to_numeric(mapping["parent_cluster"], errors="coerce").astype("Int64")
    mapping["agent_count"] = pd.to_numeric(mapping["agent_count"], errors="coerce").fillna(1).astype(int)

    merged = votes.merge(mapping[["cluster", "parent_cluster"]], on="cluster", how="left")
    merged["p_helpful"] = label_probability(merged["predicted_label"], merged["confidence"])
    merged = merged[merged["p_helpful"].notna() & merged["parent_cluster"].notna()].copy()
    merged["noteId"] = merged["noteId"].astype(str)

    grouped = (
        merged.groupby(["noteId", "parent_cluster"], as_index=False)["p_helpful"]
        .mean()
    )
    parent_order = sorted(mapping["parent_cluster"].dropna().astype(int).unique().tolist())
    note_order = notes["noteId"].astype(str).tolist()
    matrix = grouped.pivot(index="noteId", columns="parent_cluster", values="p_helpful")
    matrix = matrix.reindex(index=note_order, columns=parent_order)
    filled, missing_cells = fill_probability_matrix(matrix)

    counts = (
        mapping.groupby("parent_cluster")["agent_count"]
        .sum()
        .reindex(parent_order)
        .to_numpy(dtype=int)
    )
    return filled.to_numpy(dtype=float), counts, missing_cells


def representative_probability_matrix(
    notes: pd.DataFrame,
    votes: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    df = votes.copy()
    df["p_helpful"] = label_probability(df["predicted_label"], df["confidence"])
    df = df[df["p_helpful"].notna()].copy()
    df["noteId"] = df["noteId"].astype(str)
    note_order = notes["noteId"].astype(str).tolist()
    agent_order = sorted(votes["agent_id"].astype(str).unique().tolist())
    matrix = df.pivot_table(index="noteId", columns="agent_id", values="p_helpful", aggfunc="mean")
    matrix = matrix.reindex(index=note_order, columns=agent_order)
    filled, missing_cells = fill_probability_matrix(matrix)
    return filled.to_numpy(dtype=float), missing_cells


def evaluate_parent_binomial(
    y_true: np.ndarray,
    p_matrix: np.ndarray,
    parent_counts: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    total_agents = int(parent_counts.sum())
    rows: list[dict[str, float | int]] = []
    for repeat in range(repeats):
        helpful_votes = np.zeros(p_matrix.shape[0], dtype=int)
        for parent_idx, count in enumerate(parent_counts):
            helpful_votes += rng.binomial(int(count), p_matrix[:, parent_idx])
        y_pred = (helpful_votes / total_agents >= 0.5).astype(int)
        rows.append({"repeat": repeat, **confusion_metrics(y_true, y_pred)})
    return rows


def best_selective_thresholds(
    y_true: np.ndarray,
    score: np.ndarray,
    min_coverage: float,
) -> tuple[float, float]:
    low_grid = np.arange(0.00, 0.51, 0.01, dtype=float)
    high_grid = np.arange(0.50, 0.96, 0.01, dtype=float)
    low_values, high_values = np.meshgrid(low_grid, high_grid, indexing="ij")
    lows = low_values.ravel()
    highs = high_values.ravel()
    valid_pair = lows < highs
    lows = lows[valid_pair]
    highs = highs[valid_pair]

    scores = score.reshape(1, -1)
    y = y_true.reshape(1, -1).astype(int)
    predict_zero = scores <= lows.reshape(-1, 1)
    predict_one = scores >= highs.reshape(-1, 1)
    resolved = predict_zero | predict_one
    resolved_count = resolved.sum(axis=1)
    valid = resolved_count > 0
    if min_coverage > 0:
        valid &= (resolved_count / float(len(score))) >= min_coverage
    if not valid.any():
        return 0.0, 1.0

    tp = (predict_one & (y == 1)).sum(axis=1).astype(float)
    tn = (predict_zero & (y == 0)).sum(axis=1).astype(float)
    fp = (predict_one & (y == 0)).sum(axis=1).astype(float)
    fn = (predict_zero & (y == 1)).sum(axis=1).astype(float)
    accuracy = np.divide(tp + tn, resolved_count, out=np.zeros_like(tp), where=resolved_count > 0)
    recall_not_helpful = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    recall_helpful = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    balanced = (recall_not_helpful + recall_helpful) / 2.0
    coverage = resolved_count / float(len(score))

    invalid = ~valid
    accuracy[invalid] = -1.0
    balanced[invalid] = -1.0
    coverage[invalid] = -1.0
    # Lexicographic argmax: accuracy first, balanced accuracy second, coverage third.
    best_idx = int(np.lexsort((coverage, balanced, accuracy))[-1])
    return float(lows[best_idx]), float(highs[best_idx])


def apply_selective_thresholds(
    y_true: np.ndarray,
    score: np.ndarray,
    low: float,
    high: float,
) -> dict[str, float | int]:
    pred = np.full(len(score), -1, dtype=int)
    pred[score <= low] = 0
    pred[score >= high] = 1
    mask = pred >= 0
    if mask.any():
        metrics = confusion_metrics(y_true[mask], pred[mask])
    else:
        metrics = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "recall_not_helpful": 0.0,
            "recall_helpful": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    metrics["coverage"] = float(mask.mean())
    metrics["resolved_notes"] = int(mask.sum())
    return metrics


def crossfit_calibrated_screening(
    y_true: np.ndarray,
    score: np.ndarray,
    coverage_target: float,
    folds: int,
    seed: int,
) -> dict[str, float | int]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows: list[dict[str, float | int]] = []
    for train_idx, test_idx in cv.split(score.reshape(-1, 1), y_true):
        low, high = best_selective_thresholds(y_true[train_idx], score[train_idx], coverage_target)
        row = apply_selective_thresholds(y_true[test_idx], score[test_idx], low, high)
        fold_rows.append(row)

    pooled_tn = int(sum(int(row["tn"]) for row in fold_rows))
    pooled_fp = int(sum(int(row["fp"]) for row in fold_rows))
    pooled_fn = int(sum(int(row["fn"]) for row in fold_rows))
    pooled_tp = int(sum(int(row["tp"]) for row in fold_rows))
    pooled_y = np.array([1] * (pooled_tp + pooled_fn) + [0] * (pooled_tn + pooled_fp), dtype=int)
    pooled_pred = np.array([1] * pooled_tp + [0] * pooled_fn + [0] * pooled_tn + [1] * pooled_fp, dtype=int)
    metrics = confusion_metrics(pooled_y, pooled_pred) if len(pooled_y) else confusion_metrics(np.array([], dtype=int), np.array([], dtype=int))
    resolved_notes = pooled_tn + pooled_fp + pooled_fn + pooled_tp
    metrics["coverage"] = float(resolved_notes / len(y_true))
    metrics["resolved_notes"] = int(resolved_notes)
    return metrics


def sample_parent_helpful_share(
    p_matrix: np.ndarray,
    parent_counts: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    helpful_votes = np.zeros(p_matrix.shape[0], dtype=int)
    for parent_idx, count in enumerate(parent_counts):
        helpful_votes += rng.binomial(int(count), p_matrix[:, parent_idx])
    return helpful_votes / float(parent_counts.sum())


def sample_agent_helpful_share(
    p_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random(p_matrix.shape) < p_matrix).mean(axis=1)


def evaluate_calibrated_screening(
    y_true: np.ndarray,
    method: str,
    repeats: int,
    rng: np.random.Generator,
    coverage_target: float,
    p_matrix: np.ndarray,
    parent_counts: np.ndarray | None = None,
    folds: int = 5,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for repeat in range(repeats):
        if method == "parent_cluster_binomial":
            if parent_counts is None:
                raise ValueError("parent_counts is required for parent_cluster_binomial")
            score = sample_parent_helpful_share(p_matrix, parent_counts, rng)
        elif method == "representative_agent_bernoulli":
            score = sample_agent_helpful_share(p_matrix, rng)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        metrics = crossfit_calibrated_screening(
            y_true=y_true,
            score=score,
            coverage_target=coverage_target,
            folds=folds,
            seed=42,
        )
        rows.append({"repeat": repeat, **metrics})
    return rows


def sample_parent_votes(
    note_ids: list[str],
    p_matrix: np.ndarray,
    parent_counts: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parent_idx, count in enumerate(parent_counts):
        probs = p_matrix[:, parent_idx]
        for local_idx in range(int(count)):
            draws = (rng.random(len(note_ids)) < probs).astype(np.int8)
            rows.extend(
                {
                    "noteId": note_id,
                    "agent_id": f"parent_{parent_idx:02d}_sample_{local_idx + 1:03d}",
                    "helpful_num": int(draw),
                }
                for note_id, draw in zip(note_ids, draws)
            )
    return pd.DataFrame(rows)


def evaluate_agent_bernoulli(
    y_true: np.ndarray,
    p_matrix: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    total_agents = int(p_matrix.shape[1])
    rows: list[dict[str, float | int]] = []
    for repeat in range(repeats):
        helpful_votes = (rng.random(p_matrix.shape) < p_matrix).sum(axis=1)
        y_pred = (helpful_votes / total_agents >= 0.5).astype(int)
        rows.append({"repeat": repeat, **confusion_metrics(y_true, y_pred)})
    return rows


def sample_agent_votes(
    note_ids: list[str],
    p_matrix: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    draws = (rng.random(p_matrix.shape) < p_matrix).astype(np.int8)
    rows: list[dict[str, object]] = []
    for agent_idx in range(p_matrix.shape[1]):
        rows.extend(
            {
                "noteId": note_id,
                "agent_id": f"representative_sample_{agent_idx + 1:03d}",
                "helpful_num": int(draw),
            }
            for note_id, draw in zip(note_ids, draws[:, agent_idx])
        )
    return pd.DataFrame(rows)


def evaluate_resolved_sampling(
    y_true: np.ndarray,
    note_ids: list[str],
    method: str,
    repeats: int,
    rng: np.random.Generator,
    official_args: argparse.Namespace,
    p_matrix: np.ndarray,
    parent_counts: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    note_id_array = np.asarray(note_ids, dtype=object)
    rows: list[dict[str, float | int]] = []
    for repeat in range(repeats):
        if method == "parent_cluster_binomial":
            if parent_counts is None:
                raise ValueError("parent_counts is required for parent_cluster_binomial")
            sampled_votes = sample_parent_votes(note_ids, p_matrix, parent_counts, rng)
        elif method == "representative_agent_bernoulli":
            sampled_votes = sample_agent_votes(note_ids, p_matrix, rng)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        pred, resolved = official_mod.official_style_predict(
            sampled_votes,
            note_id_array,
            official_args,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        if resolved.any():
            metrics = confusion_metrics(y_true[resolved], pred[resolved])
        else:
            metrics = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "recall_not_helpful": 0.0,
                "recall_helpful": 0.0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }
        rows.append(
            {
                "repeat": repeat,
                "resolved_notes": int(resolved.sum()),
                "coverage": float(resolved.mean()),
                **metrics,
            }
        )
    return rows


def method_summary(
    repeat_df: pd.DataFrame,
    agent_count: int,
    method: str,
    notes_evaluated: int,
    missing_probability_cells: int,
    deterministic_metrics: dict[str, float | int],
) -> dict[str, float | int | str]:
    acc = summarize(repeat_df["accuracy"].astype(float).tolist())
    bal = summarize(repeat_df["balanced_accuracy"].astype(float).tolist())
    return {
        "agent_count": agent_count,
        "method": method,
        "repeats": int(len(repeat_df)),
        "notes_evaluated": int(notes_evaluated),
        "missing_probability_cells_imputed": int(missing_probability_cells),
        "deterministic_expected_accuracy": float(deterministic_metrics["accuracy"]),
        "deterministic_expected_balanced_accuracy": float(deterministic_metrics["balanced_accuracy"]),
        "accuracy_mean": acc["mean"],
        "accuracy_std": acc["std"],
        "accuracy_ci95_low": acc["ci95_low"],
        "accuracy_ci95_high": acc["ci95_high"],
        "accuracy_p10": acc["p10"],
        "accuracy_p50": acc["p50"],
        "accuracy_p90": acc["p90"],
        "balanced_accuracy_mean": bal["mean"],
        "balanced_accuracy_std": bal["std"],
        "balanced_accuracy_ci95_low": bal["ci95_low"],
        "balanced_accuracy_ci95_high": bal["ci95_high"],
        "balanced_accuracy_p10": bal["p10"],
        "balanced_accuracy_p50": bal["p50"],
        "balanced_accuracy_p90": bal["p90"],
    }


def resolved_method_summary(
    repeat_df: pd.DataFrame,
    agent_count: int,
    method: str,
    notes_evaluated: int,
) -> dict[str, float | int | str]:
    acc = summarize(repeat_df["accuracy"].astype(float).tolist())
    bal = summarize(repeat_df["balanced_accuracy"].astype(float).tolist())
    cov = summarize(repeat_df["coverage"].astype(float).tolist())
    resolved = summarize(repeat_df["resolved_notes"].astype(float).tolist())
    return {
        "agent_count": agent_count,
        "method": method,
        "repeats": int(len(repeat_df)),
        "notes_evaluated": int(notes_evaluated),
        "resolved_accuracy_mean": acc["mean"],
        "resolved_accuracy_std": acc["std"],
        "resolved_accuracy_ci95_low": acc["ci95_low"],
        "resolved_accuracy_ci95_high": acc["ci95_high"],
        "resolved_accuracy_p10": acc["p10"],
        "resolved_accuracy_p50": acc["p50"],
        "resolved_accuracy_p90": acc["p90"],
        "resolved_balanced_accuracy_mean": bal["mean"],
        "resolved_balanced_accuracy_std": bal["std"],
        "resolved_balanced_accuracy_ci95_low": bal["ci95_low"],
        "resolved_balanced_accuracy_ci95_high": bal["ci95_high"],
        "coverage_mean": cov["mean"],
        "coverage_std": cov["std"],
        "coverage_ci95_low": cov["ci95_low"],
        "coverage_ci95_high": cov["ci95_high"],
        "coverage_p10": cov["p10"],
        "coverage_p50": cov["p50"],
        "coverage_p90": cov["p90"],
        "resolved_notes_mean": resolved["mean"],
        "resolved_notes_p50": resolved["p50"],
    }


def calibrated_method_summary(
    repeat_df: pd.DataFrame,
    agent_count: int,
    method: str,
    notes_evaluated: int,
    coverage_target: float,
) -> dict[str, float | int | str]:
    acc = summarize(repeat_df["accuracy"].astype(float).tolist())
    bal = summarize(repeat_df["balanced_accuracy"].astype(float).tolist())
    cov = summarize(repeat_df["coverage"].astype(float).tolist())
    resolved = summarize(repeat_df["resolved_notes"].astype(float).tolist())
    return {
        "agent_count": agent_count,
        "method": method,
        "repeats": int(len(repeat_df)),
        "notes_evaluated": int(notes_evaluated),
        "coverage_target": float(coverage_target),
        "calibrated_resolved_accuracy_mean": acc["mean"],
        "calibrated_resolved_accuracy_std": acc["std"],
        "calibrated_resolved_accuracy_ci95_low": acc["ci95_low"],
        "calibrated_resolved_accuracy_ci95_high": acc["ci95_high"],
        "calibrated_resolved_accuracy_p10": acc["p10"],
        "calibrated_resolved_accuracy_p50": acc["p50"],
        "calibrated_resolved_accuracy_p90": acc["p90"],
        "calibrated_resolved_balanced_accuracy_mean": bal["mean"],
        "calibrated_resolved_balanced_accuracy_std": bal["std"],
        "calibrated_resolved_balanced_accuracy_ci95_low": bal["ci95_low"],
        "calibrated_resolved_balanced_accuracy_ci95_high": bal["ci95_high"],
        "coverage_mean": cov["mean"],
        "coverage_std": cov["std"],
        "coverage_ci95_low": cov["ci95_low"],
        "coverage_ci95_high": cov["ci95_high"],
        "coverage_p10": cov["p10"],
        "coverage_p50": cov["p50"],
        "coverage_p90": cov["p90"],
        "resolved_notes_mean": resolved["mean"],
        "resolved_notes_p50": resolved["p50"],
    }


def main() -> None:
    args = parse_args()
    official_mod.mf_mod.log = lambda _message: None
    repo_root = args.repo_root.resolve()
    counts = parse_counts(args.agent_counts)
    all_repeats: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    all_resolved_repeats: list[pd.DataFrame] = []
    resolved_summaries: list[dict[str, float | int | str]] = []
    all_calibrated_repeats: list[pd.DataFrame] = []
    calibrated_summaries: list[dict[str, float | int | str]] = []
    official_args = argparse.Namespace(
        min_ratings_per_rater=10,
        min_raters_per_note=5,
        als_iterations=12,
        als_tol=1e-4,
        global_intercept_lambda=0.15,
        user_intercept_lambda=0.15,
        note_intercept_lambda=0.15,
        user_factor_lambda=0.03,
        note_factor_lambda=0.03,
        crh_threshold=0.40,
        crnh_intercept_threshold=-0.05,
        crnh_note_factor_multiplier=-0.80,
        min_rater_agree_ratio=0.66,
    )

    for count in counts:
        notes, votes, roster = load_inputs(repo_root, count, args.model_tag, args.date_tag, args.run_tag)
        notes = notes.copy()
        notes["noteId"] = notes["noteId"].astype(str)
        note_ids = notes["noteId"].astype(str).tolist()
        y_true = pd.to_numeric(notes["true_label"], errors="coerce").astype(int).to_numpy()

        parent_p, parent_counts, parent_missing = parent_probability_matrix(notes, votes, roster)
        expected_parent_share = (parent_p * parent_counts.reshape(1, -1)).sum(axis=1) / float(parent_counts.sum())
        parent_det = confusion_metrics(y_true, (expected_parent_share >= 0.5).astype(int))
        parent_rows = evaluate_parent_binomial(
            y_true,
            parent_p,
            parent_counts,
            args.repeats,
            np.random.default_rng(args.seed + count * 1009),
        )
        parent_df = pd.DataFrame(parent_rows)
        parent_df.insert(0, "method", "parent_cluster_binomial")
        parent_df.insert(0, "agent_count", count)
        all_repeats.append(parent_df)
        summaries.append(
            method_summary(
                parent_df,
                count,
                "parent_cluster_binomial",
                len(notes),
                parent_missing,
                parent_det,
            )
        )
        if args.resolved_repeats > 0:
            parent_resolved_rows = evaluate_resolved_sampling(
                y_true=y_true,
                note_ids=note_ids,
                method="parent_cluster_binomial",
                repeats=args.resolved_repeats,
                rng=np.random.default_rng(args.seed + count * 3001),
                official_args=official_args,
                p_matrix=parent_p,
                parent_counts=parent_counts,
            )
            parent_resolved_df = pd.DataFrame(parent_resolved_rows)
            parent_resolved_df.insert(0, "method", "parent_cluster_binomial")
            parent_resolved_df.insert(0, "agent_count", count)
            all_resolved_repeats.append(parent_resolved_df)
            resolved_summaries.append(
                resolved_method_summary(
                    parent_resolved_df,
                    count,
                    "parent_cluster_binomial",
                    len(notes),
                )
            )
        if args.calibrated_repeats > 0:
            parent_calibrated_rows = evaluate_calibrated_screening(
                y_true=y_true,
                method="parent_cluster_binomial",
                repeats=args.calibrated_repeats,
                rng=np.random.default_rng(args.seed + count * 5003),
                coverage_target=args.calibrated_coverage_target,
                p_matrix=parent_p,
                parent_counts=parent_counts,
            )
            parent_calibrated_df = pd.DataFrame(parent_calibrated_rows)
            parent_calibrated_df.insert(0, "method", "parent_cluster_binomial")
            parent_calibrated_df.insert(0, "agent_count", count)
            all_calibrated_repeats.append(parent_calibrated_df)
            calibrated_summaries.append(
                calibrated_method_summary(
                    parent_calibrated_df,
                    count,
                    "parent_cluster_binomial",
                    len(notes),
                    args.calibrated_coverage_target,
                )
            )

        agent_p, agent_missing = representative_probability_matrix(notes, votes)
        expected_agent_share = agent_p.mean(axis=1)
        agent_det = confusion_metrics(y_true, (expected_agent_share >= 0.5).astype(int))
        agent_rows = evaluate_agent_bernoulli(
            y_true,
            agent_p,
            args.repeats,
            np.random.default_rng(args.seed + count * 2003),
        )
        agent_df = pd.DataFrame(agent_rows)
        agent_df.insert(0, "method", "representative_agent_bernoulli")
        agent_df.insert(0, "agent_count", count)
        all_repeats.append(agent_df)
        summaries.append(
            method_summary(
                agent_df,
                count,
                "representative_agent_bernoulli",
                len(notes),
                agent_missing,
                agent_det,
            )
        )
        if args.resolved_repeats > 0:
            agent_resolved_rows = evaluate_resolved_sampling(
                y_true=y_true,
                note_ids=note_ids,
                method="representative_agent_bernoulli",
                repeats=args.resolved_repeats,
                rng=np.random.default_rng(args.seed + count * 4001),
                official_args=official_args,
                p_matrix=agent_p,
            )
            agent_resolved_df = pd.DataFrame(agent_resolved_rows)
            agent_resolved_df.insert(0, "method", "representative_agent_bernoulli")
            agent_resolved_df.insert(0, "agent_count", count)
            all_resolved_repeats.append(agent_resolved_df)
            resolved_summaries.append(
                resolved_method_summary(
                    agent_resolved_df,
                    count,
                    "representative_agent_bernoulli",
                    len(notes),
                )
            )
        if args.calibrated_repeats > 0:
            agent_calibrated_rows = evaluate_calibrated_screening(
                y_true=y_true,
                method="representative_agent_bernoulli",
                repeats=args.calibrated_repeats,
                rng=np.random.default_rng(args.seed + count * 6007),
                coverage_target=args.calibrated_coverage_target,
                p_matrix=agent_p,
            )
            agent_calibrated_df = pd.DataFrame(agent_calibrated_rows)
            agent_calibrated_df.insert(0, "method", "representative_agent_bernoulli")
            agent_calibrated_df.insert(0, "agent_count", count)
            all_calibrated_repeats.append(agent_calibrated_df)
            calibrated_summaries.append(
                calibrated_method_summary(
                    agent_calibrated_df,
                    count,
                    "representative_agent_bernoulli",
                    len(notes),
                    args.calibrated_coverage_target,
                )
            )

    summary_df = pd.DataFrame(summaries).sort_values(["agent_count", "method"]).reset_index(drop=True)
    repeats_df = pd.concat(all_repeats, ignore_index=True)
    resolved_summary_df = pd.DataFrame(resolved_summaries)
    if not resolved_summary_df.empty:
        resolved_summary_df = resolved_summary_df.sort_values(["agent_count", "method"]).reset_index(drop=True)
    resolved_repeats_df = pd.concat(all_resolved_repeats, ignore_index=True) if all_resolved_repeats else pd.DataFrame()
    calibrated_summary_df = pd.DataFrame(calibrated_summaries)
    if not calibrated_summary_df.empty:
        calibrated_summary_df = calibrated_summary_df.sort_values(["agent_count", "method"]).reset_index(drop=True)
    calibrated_repeats_df = pd.concat(all_calibrated_repeats, ignore_index=True) if all_calibrated_repeats else pd.DataFrame()

    summary_path = args.output_summary_csv if args.output_summary_csv.is_absolute() else repo_root / args.output_summary_csv
    repeats_path = args.output_repeats_csv if args.output_repeats_csv.is_absolute() else repo_root / args.output_repeats_csv
    resolved_summary_path = summary_path.with_name("probability_sampling_resolved_summary.csv")
    resolved_repeats_path = repeats_path.with_name("probability_sampling_resolved_repeats.csv")
    calibrated_summary_path = summary_path.with_name("probability_sampling_calibrated_summary.csv")
    calibrated_repeats_path = repeats_path.with_name("probability_sampling_calibrated_repeats.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    repeats_df.to_csv(repeats_path, index=False, encoding="utf-8-sig")
    resolved_summary_df.to_csv(resolved_summary_path, index=False, encoding="utf-8-sig")
    resolved_repeats_df.to_csv(resolved_repeats_path, index=False, encoding="utf-8-sig")
    calibrated_summary_df.to_csv(calibrated_summary_path, index=False, encoding="utf-8-sig")
    calibrated_repeats_df.to_csv(calibrated_repeats_path, index=False, encoding="utf-8-sig")

    print(summary_path)
    print(summary_df.to_string(index=False))
    if not resolved_summary_df.empty:
        print(resolved_summary_path)
        print(resolved_summary_df.to_string(index=False))
    if not calibrated_summary_df.empty:
        print(calibrated_summary_path)
        print(calibrated_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
