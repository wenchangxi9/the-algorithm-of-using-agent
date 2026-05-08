from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optimized aggregation over completed MF-continuous multi-agent runs and summarize results."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--agent-counts", type=str, default="12,24,36,48,72")
    parser.add_argument("--model-tag", type=str, default="gpt54nano")
    parser.add_argument("--date-tag", type=str, default="20260507")
    parser.add_argument("--run-tag", type=str, default="run1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_expected_votes(project_root: Path, agent_count: int) -> int:
    roster = project_root / "analysis" / "communitynotes_mf_continuous_agent_count_variants" / f"mf_continuous_n{agent_count:03d}" / "agent_roster.csv"
    if not roster.exists():
        raise FileNotFoundError(roster)
    agent_total = len(pd.read_csv(roster))
    return agent_total * 258


def vote_rows(path: Path) -> int:
    if not path.exists():
        return 0
    # Count rows without loading the full CSV into memory.
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def run_optimizer(
    project_root: Path,
    agent_count: int,
    model_tag: str,
    date_tag: str,
    run_tag: str,
    folds: int,
    inner_folds: int,
    seed: int,
    force: bool,
) -> Path | None:
    variant = f"mf_continuous_n{agent_count:03d}"
    run_dir = project_root / "analysis" / f"llm_persona_multiagent_{variant}_{model_tag}_{date_tag}_{run_tag}"
    votes_csv = run_dir / "agent_votes.csv"
    notes_csv = run_dir / "note_predictions.csv"
    expected = read_expected_votes(project_root, agent_count)
    observed = vote_rows(votes_csv)
    if observed < expected or not notes_csv.exists():
        print(
            json.dumps(
                {
                    "agent_count": agent_count,
                    "status": "skipped_incomplete",
                    "observed_vote_rows": observed,
                    "expected_vote_rows": expected,
                    "run_dir": str(run_dir),
                },
                ensure_ascii=False,
            )
        )
        return None

    output_dir = project_root / "analysis" / f"mf_continuous_n{agent_count:03d}_optimized_aggregation_{date_tag}"
    metadata = output_dir / "run_metadata.json"
    if metadata.exists() and not force:
        print(
            json.dumps(
                {
                    "agent_count": agent_count,
                    "status": "exists",
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
            )
        )
        return output_dir

    cmd = [
        sys.executable,
        str(project_root / "optimize_llm_multiagent_aggregation.py"),
        "--note-predictions-csv",
        str(notes_csv),
        "--agent-votes-csv",
        str(votes_csv),
        "--output-dir",
        str(output_dir),
        "--folds",
        str(folds),
        "--inner-folds",
        str(inner_folds),
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, cwd=project_root, check=True)
    return output_dir


def summarize_outputs(output_dirs: list[Path], output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for out in output_dirs:
        if out is None:
            continue
        agent_count = int(out.name.split("_n", 1)[1].split("_", 1)[0])
        method_summary = pd.read_csv(out / "nested_cv_method_summary.csv")
        selective = pd.read_csv(out / "nested_cv_selective_summary_by_coverage.csv")

        majority = method_summary[method_summary["method"].eq("majority_threshold_050")].iloc[0]
        best_full = method_summary.sort_values(["accuracy", "balanced_accuracy"], ascending=False).iloc[0]
        rows.append(
            {
                "agent_count": agent_count,
                "metric_type": "full_majority",
                "method": majority["method"],
                "accuracy": float(majority["accuracy"]),
                "balanced_accuracy": float(majority["balanced_accuracy"]),
                "coverage": 1.0,
                "resolved_notes": 258,
                "recall_helpful": float(majority["recall_helpful"]),
                "recall_not_helpful": float(majority["recall_not_helpful"]),
            }
        )
        rows.append(
            {
                "agent_count": agent_count,
                "metric_type": "full_optimized_nested_cv",
                "method": best_full["method"],
                "accuracy": float(best_full["accuracy"]),
                "balanced_accuracy": float(best_full["balanced_accuracy"]),
                "coverage": 1.0,
                "resolved_notes": 258,
                "recall_helpful": float(best_full["recall_helpful"]),
                "recall_not_helpful": float(best_full["recall_not_helpful"]),
            }
        )

        for target, group in selective.groupby("coverage_target"):
            best = group.sort_values(["pooled_accuracy", "pooled_balanced_accuracy"], ascending=False).iloc[0]
            rows.append(
                {
                    "agent_count": agent_count,
                    "metric_type": f"resolved_nested_cv_target_{float(target):.2f}",
                    "method": best["method"],
                    "accuracy": float(best["pooled_accuracy"]),
                    "balanced_accuracy": float(best["pooled_balanced_accuracy"]),
                    "coverage": float(best["coverage_mean"]),
                    "resolved_notes": int(best["pooled_resolved_notes"]),
                    "recall_helpful": "",
                    "recall_not_helpful": "",
                }
            )

    summary = pd.DataFrame(rows).sort_values(["agent_count", "metric_type"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    counts = [int(v.strip()) for v in args.agent_counts.split(",") if v.strip()]
    output_dirs: list[Path] = []
    for count in counts:
        output_dir = run_optimizer(
            project_root=project_root,
            agent_count=count,
            model_tag=args.model_tag,
            date_tag=args.date_tag,
            run_tag=args.run_tag,
            folds=args.folds,
            inner_folds=args.inner_folds,
            seed=args.seed,
            force=args.force,
        )
        if output_dir is not None:
            output_dirs.append(output_dir)

    summary_path = project_root / "analysis" / f"mf_continuous_optimized_aggregation_grid_summary_{args.date_tag}.csv"
    summary = summarize_outputs(output_dirs, summary_path)
    print(summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
