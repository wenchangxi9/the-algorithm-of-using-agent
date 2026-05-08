from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_AGENT_COUNTS = "12,24,36,48,72,96,120"
DEFAULT_SELECTION_FEATURES = [
    "bw_final_rater_intercept",
    "bw_final_rater_factor_1",
    "bw_rater_agree_ratio",
    "bw_mean_note_score",
    "bw_crh_crnh_ratio_difference",
    "share_helpful",
    "share_not_helpful",
    "evidence_focus_rate",
    "strict_rejection_rate",
    "redundancy_rejection_rate",
    "ratings_per_active_day",
    "notes_authored",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build multiple MF-continuous persona populations for agent-count ablations "
            "and summarize allocation/diversity diagnostics."
        )
    )
    parser.add_argument(
        "--input-features",
        type=Path,
        default=Path("artifacts/persona_inputs/user_features_with_mf_persona_clusters.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/agent_variants"),
    )
    parser.add_argument("--agent-counts", type=str, default=DEFAULT_AGENT_COUNTS)
    parser.add_argument("--min-agents-per-parent-cluster", type=int, default=3)
    parser.add_argument(
        "--selection-features",
        type=str,
        default=",".join(DEFAULT_SELECTION_FEATURES),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-kmeans-iterations", type=int, default=300)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rebuild a variant when run_metadata.json already exists.",
    )
    return parser.parse_args()


def parse_counts(text: str) -> list[int]:
    counts = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not counts:
        raise ValueError("No agent counts were provided.")
    if any(count < 1 for count in counts):
        raise ValueError("Agent counts must be positive.")
    return sorted(set(counts))


def run_builder(
    input_features: Path,
    output_dir: Path,
    total_agents: int,
    min_agents: int,
    selection_features: list[str],
    random_state: int,
    max_kmeans_iterations: int,
    skip_existing: bool,
) -> None:
    metadata_path = output_dir / "run_metadata.json"
    if skip_existing and metadata_path.exists():
        return

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("build_mf_continuous_persona_agents.py")),
        "--input-features",
        str(input_features),
        "--output-dir",
        str(output_dir),
        "--total-agents",
        str(total_agents),
        "--min-agents-per-parent-cluster",
        str(min_agents),
        "--selection-features",
        ",".join(selection_features),
        "--random-state",
        str(random_state),
        "--max-kmeans-iterations",
        str(max_kmeans_iterations),
    ]
    subprocess.run(cmd, check=True)


def average_pairwise_distance(matrix: np.ndarray) -> tuple[float, float, float]:
    if len(matrix) < 2:
        return 0.0, 0.0, 0.0
    distances: list[float] = []
    for i, j in combinations(range(len(matrix)), 2):
        distances.append(float(np.linalg.norm(matrix[i] - matrix[j])))
    arr = np.asarray(distances, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def summarize_variant(output_dir: Path, selection_features: list[str]) -> dict[str, object]:
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    roster = pd.read_csv(output_dir / "agent_roster.csv", low_memory=False)
    summary = pd.read_csv(output_dir / "cluster_summary.csv", low_memory=False)
    persona_map = pd.read_csv(output_dir / "persona_id_map.csv", low_memory=False)

    parent_counts = roster.groupby("parent_cluster")["agent_count"].sum().sort_index()
    parent_shares = parent_counts / float(parent_counts.sum())
    entropy = -float(np.sum(parent_shares.to_numpy() * np.log(np.maximum(parent_shares.to_numpy(), 1e-12))))
    normalized_entropy = entropy / float(np.log(len(parent_shares))) if len(parent_shares) > 1 else 0.0
    effective_parent_clusters = float(np.exp(entropy))

    feature_cols = [col for col in selection_features if col in summary.columns]
    feature_matrix = summary[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True)).fillna(0.0)
    if len(feature_matrix) >= 2 and feature_matrix.shape[1] > 0:
        scaled = StandardScaler().fit_transform(feature_matrix.to_numpy(dtype=float))
        mean_dist, min_dist, max_dist = average_pairwise_distance(scaled)
    else:
        mean_dist = min_dist = max_dist = 0.0

    row: dict[str, object] = {
        "agent_count": int(metadata["total_agents"]),
        "variant_dir": str(output_dir),
        "parent_clusters": int(metadata["parent_clusters"]),
        "requested_min_agents_per_parent_cluster": int(metadata["min_agents_per_parent_cluster"]),
        "min_parent_agents": int(parent_counts.min()),
        "max_parent_agents": int(parent_counts.max()),
        "parent_allocation_json": json.dumps({str(int(k)): int(v) for k, v in parent_counts.items()}, ensure_ascii=False),
        "parent_allocation_entropy": normalized_entropy,
        "effective_parent_clusters": effective_parent_clusters,
        "max_parent_share": float(parent_shares.max()),
        "representative_cells": int(len(summary)),
        "min_cell_users": int(summary["users"].min()),
        "median_cell_users": float(summary["users"].median()),
        "max_cell_users": int(summary["users"].max()),
        "mean_cell_users": float(summary["users"].mean()),
        "mean_medoid_centroid_distance": float(summary["medoid_centroid_distance"].mean()),
        "p90_medoid_centroid_distance": float(summary["medoid_centroid_distance"].quantile(0.90)),
        "mean_pairwise_representative_distance": mean_dist,
        "min_pairwise_representative_distance": min_dist,
        "max_pairwise_representative_distance": max_dist,
        "mean_agreement": float(persona_map["bw_rater_agree_ratio"].mean()),
        "std_agreement": float(persona_map["bw_rater_agree_ratio"].std(ddof=1)),
        "mean_helpful_share": float(persona_map["share_helpful"].mean()),
        "std_helpful_share": float(persona_map["share_helpful"].std(ddof=1)),
        "mean_not_helpful_share": float(persona_map["share_not_helpful"].mean()),
        "std_not_helpful_share": float(persona_map["share_not_helpful"].std(ddof=1)),
    }
    return row


def write_method_note(output_root: Path, counts: list[int]) -> None:
    note = f"""
# MF-continuous agent-count variants

这个目录用于比较不同 agent budget 下的 MF-continuous persona construction。它不是把 72 个 agent 当成默认最优，而是把 agent 数量本身作为 ablation 变量。

默认生成的 agent 数量为：{", ".join(str(c) for c in counts)}。

## 判断标准

每个 variant 都使用同一套理论构造：先保留官方 MF 的 6 个父 cluster，再在每个父 cluster 内对连续 MF/行为特征做 KMeans vector quantization，生成局部代表 persona。

选择最终 agent 数时建议同时看：

- downstream 指标：full accuracy、official-style resolved accuracy、coverage、balanced accuracy；
- 稳定性：多次重复下的均值、方差、95% CI；
- 多样性：`mean_pairwise_representative_distance`、`std_helpful_share`、`std_agreement`；
- 覆盖均衡：`parent_allocation_entropy`、`effective_parent_clusters`、`max_parent_share`；
- 代表性：`mean_medoid_centroid_distance` 越低，说明局部代表越贴近其子群中心。

原则上，不应该选择“数字最大”的 agent 数，而应该选择在 dev set 上 accuracy/coverage 已经接近饱和、继续加 agent 边际收益很小、且多样性和稳定性较好的最小 agent 数。

## 文件

- `mf_continuous_nXXX/`：对应 agent 数量的 persona/roster 文件。
- `agent_count_variant_summary.csv`：所有数量的 allocation 和多样性诊断。
- `run_metadata.json`：批量构造参数。
"""
    (output_root / "method_notes_zh.md").write_text(note.strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    counts = parse_counts(args.agent_counts)
    selection_features = [feature.strip() for feature in args.selection_features.split(",") if feature.strip()]

    rows: list[dict[str, object]] = []
    for count in counts:
        output_dir = output_root / f"mf_continuous_n{count:03d}"
        run_builder(
            input_features=args.input_features,
            output_dir=output_dir,
            total_agents=count,
            min_agents=args.min_agents_per_parent_cluster,
            selection_features=selection_features,
            random_state=args.random_state + count * 17,
            max_kmeans_iterations=args.max_kmeans_iterations,
            skip_existing=args.skip_existing,
        )
        rows.append(summarize_variant(output_dir, selection_features))

    summary = pd.DataFrame(rows).sort_values("agent_count").reset_index(drop=True)
    summary.to_csv(output_root / "agent_count_variant_summary.csv", index=False)

    metadata = {
        "method": "mf_continuous_agent_count_variants",
        "input_features": str(args.input_features),
        "output_root": str(args.output_root),
        "agent_counts": counts,
        "min_agents_per_parent_cluster": int(args.min_agents_per_parent_cluster),
        "selection_features": selection_features,
        "random_state": int(args.random_state),
        "max_kmeans_iterations": int(args.max_kmeans_iterations),
        "variant_summary_csv": str(output_root / "agent_count_variant_summary.csv"),
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_method_note(output_root, counts)

    print(summary.to_string(index=False))
    print(f"[ok] wrote {output_root / 'agent_count_variant_summary.csv'}")


if __name__ == "__main__":
    main()
