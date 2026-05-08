from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    counts = parse_counts(args.agent_counts)
    all_repeats: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []

    for count in counts:
        notes, votes, roster = load_inputs(repo_root, count, args.model_tag, args.date_tag, args.run_tag)
        notes = notes.copy()
        notes["noteId"] = notes["noteId"].astype(str)
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

    summary_df = pd.DataFrame(summaries).sort_values(["agent_count", "method"]).reset_index(drop=True)
    repeats_df = pd.concat(all_repeats, ignore_index=True)

    summary_path = args.output_summary_csv if args.output_summary_csv.is_absolute() else repo_root / args.output_summary_csv
    repeats_path = args.output_repeats_csv if args.output_repeats_csv.is_absolute() else repo_root / args.output_repeats_csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    repeats_df.to_csv(repeats_path, index=False, encoding="utf-8-sig")

    print(summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
