from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from build_matrix_factorized_multiagent_inputs import (
    SUMMARY_MEAN_COLUMNS,
    authoring_label,
    build_system_prompt as build_cluster_system_prompt,
    helpful_tendency_label,
    recent_shift_label,
    relative_band,
)


DEFAULT_SELECTION_FEATURES = [
    # Official-style MF judgment space.
    "bw_final_rater_intercept",
    "bw_final_rater_factor_1",
    "bw_rater_agree_ratio",
    "bw_mean_note_score",
    "bw_crh_crnh_ratio_difference",
    # Observable behavioral axes that make the textual persona interpretable.
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
            "Build a 72-agent population by quantizing the continuous official-MF contributor "
            "feature space inside each parent MF cluster."
        )
    )
    parser.add_argument(
        "--input-features",
        type=Path,
        default=Path("artifacts/persona_inputs/user_features_with_mf_persona_clusters.csv"),
        help="Merged user feature table containing behavior features and official-MF cluster assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/agent_variants/mf_continuous_n072"),
    )
    parser.add_argument("--total-agents", type=int, default=72)
    parser.add_argument(
        "--min-agents-per-parent-cluster",
        type=int,
        default=3,
        help=(
            "Minimum representatives per original MF cluster. This prevents small but distinct "
            "viewpoint clusters from being represented by a single stochastic agent."
        ),
    )
    parser.add_argument(
        "--selection-features",
        type=str,
        default=",".join(DEFAULT_SELECTION_FEATURES),
        help="Comma-separated numeric features used for within-cluster vector quantization.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-kmeans-iterations",
        type=int,
        default=300,
    )
    return parser.parse_args()


def allocate_parent_counts(parent_sizes: pd.Series, total_agents: int, min_agents: int) -> pd.Series:
    parent_sizes = parent_sizes.sort_index().astype(int)
    n_clusters = len(parent_sizes)
    if total_agents < n_clusters:
        raise ValueError("total_agents must be at least the number of parent clusters.")
    if min_agents < 1:
        raise ValueError("min_agents must be >= 1.")

    base_min = min(min_agents, total_agents // n_clusters)
    allocation = pd.Series(base_min, index=parent_sizes.index, dtype="int64")
    remaining = total_agents - int(allocation.sum())
    if remaining <= 0:
        return allocation

    weights = parent_sizes / float(parent_sizes.sum())
    raw = weights * remaining
    allocation += np.floor(raw).astype(int)
    shortfall = total_agents - int(allocation.sum())
    remainders = raw - np.floor(raw)
    order = sorted(
        parent_sizes.index.tolist(),
        key=lambda c: (-float(remainders.loc[c]), -int(parent_sizes.loc[c]), int(c)),
    )
    for cluster in order[:shortfall]:
        allocation.loc[cluster] += 1
    return allocation.astype(int)


def load_feature_table(path: Path, selection_features: list[str]) -> pd.DataFrame:
    required = set(SUMMARY_MEAN_COLUMNS) | set(selection_features) | {"participantId", "cluster"}
    optional = {"old_behavior_cluster", "old_cluster", "persona_cluster"}
    header = pd.read_csv(path.resolve(), nrows=0).columns.tolist()
    usecols = [col for col in header if col in required | optional]
    missing = required - set(usecols)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = pd.read_csv(path.resolve(), usecols=usecols, low_memory=False)
    df = df.rename(columns={"cluster": "parent_cluster"})
    df["participantId"] = df["participantId"].astype(str)
    df["parent_cluster"] = df["parent_cluster"].astype(int)
    for col in set(SUMMARY_MEAN_COLUMNS) | set(selection_features):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_matrix(group: pd.DataFrame, selection_features: list[str]) -> np.ndarray:
    x = group[selection_features].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    lower = x.quantile(0.01)
    upper = x.quantile(0.99)
    x = x.clip(lower=lower, upper=upper, axis=1)
    return StandardScaler().fit_transform(x.to_numpy(dtype=float))


def quantize_parent_cluster(
    group: pd.DataFrame,
    n_representatives: int,
    selection_features: list[str],
    random_state: int,
    max_iter: int,
) -> pd.DataFrame:
    if n_representatives < 1:
        raise ValueError("n_representatives must be >= 1")
    if len(group) < n_representatives:
        raise ValueError("Cannot select more representatives than users in a parent cluster.")

    x_scaled = prepare_matrix(group, selection_features)
    model = KMeans(
        n_clusters=n_representatives,
        n_init=20,
        max_iter=max_iter,
        random_state=random_state,
    )
    local_labels = model.fit_predict(x_scaled)
    centers = model.cluster_centers_
    distances = ((x_scaled - centers[local_labels]) ** 2).sum(axis=1)

    work = group.copy()
    work["_local_label"] = local_labels
    work["_centroid_distance"] = distances

    rows: list[dict[str, object]] = []
    for local_label in range(n_representatives):
        subgroup = work[work["_local_label"] == local_label].copy()
        medoid = subgroup.sort_values("_centroid_distance", ascending=True).iloc[0]
        numeric_summary = subgroup[SUMMARY_MEAN_COLUMNS].mean(numeric_only=True).to_dict()
        row: dict[str, object] = {
            **numeric_summary,
            "parent_cluster": int(medoid["parent_cluster"]),
            "local_representative": int(local_label),
            "users": int(len(subgroup)),
            "medoid_participantId": str(medoid["participantId"]),
            "medoid_centroid_distance": float(medoid["_centroid_distance"]),
        }
        for feature in selection_features:
            row[f"medoid_{feature}"] = float(medoid[feature])
        if "old_behavior_cluster" in subgroup.columns:
            row["dominant_old_behavior_cluster"] = int(subgroup["old_behavior_cluster"].mode().iat[0])
        elif "old_cluster" in subgroup.columns:
            row["dominant_old_behavior_cluster"] = int(subgroup["old_cluster"].mode().iat[0])
        if "persona_cluster" in subgroup.columns:
            row["dominant_old_persona_cluster"] = int(subgroup["persona_cluster"].mode().iat[0])
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values(
        ["parent_cluster", "bw_final_rater_factor_1", "bw_final_rater_intercept", "bw_rater_agree_ratio"],
        kind="mergesort",
    ).reset_index(drop=True)


def band_short(value: float, series: pd.Series, high: str, mid: str, low: str) -> str:
    band = relative_band(value, series)
    if band in {"极高", "较高"}:
        return high
    if band == "中等":
        return mid
    return low


def build_representative_labels(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        margin = float(row.share_helpful) - float(row.share_not_helpful)
        factor = float(row.bw_final_rater_factor_1)
        agree_name = band_short(float(row.bw_rater_agree_ratio), summary["bw_rater_agree_ratio"], "高一致", "中一致", "低一致")
        strict_name = band_short(float(row.strict_rejection_rate), summary["strict_rejection_rate"], "严格", "平衡", "宽松")
        if factor >= 0.15:
            viewpoint = "正向视角"
        elif factor <= -0.15:
            viewpoint = "反向视角"
        else:
            viewpoint = "中间视角"
        if margin >= 0.18:
            tendency = "偏Helpful"
        elif margin <= -0.12:
            tendency = "偏NotHelpful"
        else:
            tendency = "均衡"

        persona_name = f"{agree_name}{strict_name}{viewpoint}{tendency}代表"

        ratings_band = relative_band(float(row.ratings_given), summary["ratings_given"])
        activity_label = "高频" if ratings_band in {"极高", "较高"} else ("中频" if ratings_band == "中等" else "低频")

        if float(row.notes_authored) < 1:
            author_label_short = "几乎不写 note"
        elif float(row.notes_authored) < 4:
            author_label_short = "少写 note"
        elif float(row.notes_authored) < 10:
            author_label_short = "常写 note"
        else:
            author_label_short = "高产 note 作者"

        evidence_band = relative_band(float(row.evidence_focus_rate), summary["evidence_focus_rate"])
        strict_band = relative_band(float(row.strict_rejection_rate), summary["strict_rejection_rate"])
        if evidence_band in {"极高", "较高"} and strict_band in {"极高", "较高"}:
            style_label = "证据导向且门槛较高"
        elif evidence_band in {"极高", "较高"}:
            style_label = "证据导向"
        elif strict_band in {"极高", "较高"}:
            style_label = "严格筛选"
        else:
            style_label = "平衡判断"

        shift_mag = abs(float(row.recent_helpful_shift)) + abs(float(row.recent_not_helpful_shift))
        if shift_mag < 0.05:
            volatility_label = "近期稳定"
        elif float(row.recent_helpful_shift) >= 0.08:
            volatility_label = "近期更偏 Helpful"
        elif float(row.recent_not_helpful_shift) >= 0.08:
            volatility_label = "近期更偏否决"
        else:
            volatility_label = "近期有轻微漂移"

        if margin >= 0.12:
            stance_label = "更容易给 Helpful"
        elif margin <= -0.08:
            stance_label = "更容易给 Not Helpful"
        else:
            stance_label = "Helpful / Not Helpful 相对均衡"

        rows.append(
            {
                "cluster": int(row.cluster),
                "parent_cluster": int(row.parent_cluster),
                "local_representative": int(row.local_representative),
                "persona_name": persona_name,
                "activity_label": activity_label,
                "author_label": author_label_short,
                "stance_label": stance_label,
                "style_label": style_label,
                "volatility_label": volatility_label,
            }
        )
    return pd.DataFrame(rows)


def add_agent_specific_context(prompt: str, row: pd.Series) -> str:
    helpful_label = helpful_tendency_label(float(row["share_helpful"]), float(row["share_not_helpful"]))
    recent_label = recent_shift_label(float(row["recent_helpful_shift"]), float(row["recent_not_helpful_shift"]))
    author_label = authoring_label(
        float(row["share_authored_crh"]),
        float(row["share_authored_crnh"]),
        float(row["notes_authored"]),
    )
    parent = int(row["parent_cluster"])
    local = int(row["local_representative"])
    cluster = int(row["cluster"])
    replacement = (
        f"你的固定身份是 Birdwatch 矩阵分解聚类得到的原始用户簇 #{parent} 中的代表型 agent "
        f"#{local}：{row['persona_name']}。\n"
        f"这个代表型 agent 不是整个簇的平均人，而是通过 MF 特征空间向量量化选出的局部子群画像；"
        f"该局部子群包含 {int(row['users'])} 个真实 contributor。"
    )
    old = f"你的固定身份是 Birdwatch 矩阵分解聚类得到的用户簇 #{cluster}：{row['persona_name']}。"
    prompt = prompt.replace(old, replacement)

    context = f"""

代表采样补充：
- 原始 MF 父 cluster：{parent}；代表 persona id：{cluster}；父 cluster 内局部代表编号：{local}。
- 这个 persona 的理论含义是父 cluster 内的一个局部 medoid/quantization cell，而不是重复复制的平均 agent。
- 局部子群规模：{int(row['users'])} 个真实 contributor；medoid contributor id：{row['medoid_participantId']}。
- 局部 Helpful / Not Helpful 比例：{float(row['share_helpful']) * 100:.1f}% / {float(row['share_not_helpful']) * 100:.1f}%，整体{helpful_label}。
- 局部 rater agreement：{float(row['bw_rater_agree_ratio']) * 100:.1f}%；局部潜在视角 factor：{float(row['bw_final_rater_factor_1']):.3f}。
- 局部作者侧倾向：{author_label}；近期变化：{recent_label}。
"""
    marker = "你在评分时要遵循这类人的真实偏好"
    if marker in prompt:
        prompt = prompt.replace(marker, context.strip() + "\n\n" + marker, 1)
    else:
        prompt = prompt + "\n\n" + context.strip()
    return prompt


def write_method_note(output_dir: Path, metadata: dict[str, object]) -> None:
    note = f"""
# MF-continuous persona agent construction

这个目录实现的是 `MF-continuous / quantized persona agents`，目的是替代“每个 MF cluster 只写一个平均 persona、再复制多次”的旧做法。

## 理论动机

Community Notes 的官方风格矩阵分解会把 contributor 压缩到连续的判断空间：rater intercept 表示整体宽松/严格，rater factor 表示潜在视角差异，agreement ratio 表示和稳定共识的一致程度。原来的 cluster-average agent 会把同一个 cluster 内的大量异质性压成一个平均人；复制这个平均 persona 只能增加采样噪声，不一定增加真实判断多样性。

因此这里把 agent construction 看成一个代表性采样/向量量化问题：在每个原始 MF cluster 内，对标准化后的 MF 特征和可解释行为特征做 KMeans quantization，再把每个 cell 的 medoid 和局部子群均值写成一个 persona。这样每个 agent 对应真实 contributor 空间中的一个局部代表，而不是纯 prompt 变体。

## 实现细节

- 父 cluster：官方风格 MF 聚类得到的 6 个 cluster。
- 总 agent 数：{metadata["total_agents"]}。
- 每个父 cluster 最小代表数：{metadata["min_agents_per_parent_cluster"]}。
- 选择特征：{", ".join(metadata["selection_features"])}
- allocation：先给每个父 cluster 最小 quota，剩余 agent 按父 cluster 人数比例分配。
- representative：每个父 cluster 内运行 KMeans；每个局部 cell 选择离 centroid 最近的真实 contributor 作为 medoid，同时用 cell 内全部 contributor 的均值生成 persona 画像。

## 文件

- `cluster_personas.csv`：72 个代表 persona，每行一个 system prompt。
- `agent_roster.csv`：兼容现有 multi-agent runner 的 roster；每个代表 persona 的 `agent_count=1`。
- `cluster_summary.csv`：每个代表 persona 的局部子群统计和 medoid 信息。
- `persona_id_map.csv`：代表 persona id 到原始 MF parent cluster 的映射。
- `run_metadata.json`：构造参数。
"""
    (output_dir / "method_notes_zh.md").write_text(dedent(note).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    selection_features = [feature.strip() for feature in args.selection_features.split(",") if feature.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    users = load_feature_table(args.input_features, selection_features)
    parent_sizes = users.groupby("parent_cluster").size().astype(int)
    allocation = allocate_parent_counts(parent_sizes, args.total_agents, args.min_agents_per_parent_cluster)

    summaries: list[pd.DataFrame] = []
    for parent_cluster, group in users.groupby("parent_cluster", sort=True):
        n_reps = int(allocation.loc[int(parent_cluster)])
        summaries.append(
            quantize_parent_cluster(
                group=group.reset_index(drop=True),
                n_representatives=n_reps,
                selection_features=selection_features,
                random_state=args.random_state + int(parent_cluster) * 1009,
                max_iter=args.max_kmeans_iterations,
            )
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary = summary.sort_values(
        ["parent_cluster", "bw_final_rater_factor_1", "bw_final_rater_intercept", "bw_rater_agree_ratio"],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "cluster", np.arange(len(summary), dtype=int))
    summary["avg_ratings_given"] = summary["ratings_given"]
    summary["avg_notes_authored"] = summary["notes_authored"]
    summary["avg_share_helpful"] = summary["share_helpful"]
    summary["avg_share_not_helpful"] = summary["share_not_helpful"]

    labels = build_representative_labels(summary)
    persona_df = labels.merge(summary, on=["cluster", "parent_cluster", "local_representative"], how="inner", validate="one_to_one")
    persona_df["system_prompt"] = [
        add_agent_specific_context(build_cluster_system_prompt(row, persona_df), row)
        for _, row in persona_df.iterrows()
    ]

    roster = (
        persona_df[
            [
                "cluster",
                "parent_cluster",
                "local_representative",
                "persona_name",
                "share_helpful",
                "share_not_helpful",
                "activity_burstiness",
            ]
        ]
        .rename(
            columns={
                "share_helpful": "prior_helpful_rate",
                "activity_burstiness": "volatility",
            }
        )
        .assign(
            stance_bias=lambda df: df["prior_helpful_rate"] - df["share_not_helpful"],
            agent_count=1,
            uses_global_fallback=False,
        )[
            [
                "cluster",
                "parent_cluster",
                "local_representative",
                "persona_name",
                "prior_helpful_rate",
                "volatility",
                "stance_bias",
                "agent_count",
                "uses_global_fallback",
            ]
        ]
    )

    persona_id_map = persona_df[
        [
            "cluster",
            "parent_cluster",
            "local_representative",
            "users",
            "medoid_participantId",
            "persona_name",
            "bw_final_rater_intercept",
            "bw_final_rater_factor_1",
            "bw_rater_agree_ratio",
            "share_helpful",
            "share_not_helpful",
        ]
    ].copy()

    summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    persona_df[
        [
            "cluster",
            "parent_cluster",
            "local_representative",
            "persona_name",
            "activity_label",
            "author_label",
            "stance_label",
            "style_label",
            "volatility_label",
            "system_prompt",
        ]
    ].to_csv(output_dir / "cluster_personas.csv", index=False)
    roster.to_csv(output_dir / "agent_roster.csv", index=False)
    persona_id_map.to_csv(output_dir / "persona_id_map.csv", index=False)

    metadata = {
        "method": "mf_continuous_quantized_persona_agents",
        "input_features": str(args.input_features),
        "output_dir": str(args.output_dir),
        "total_agents": int(args.total_agents),
        "parent_clusters": int(len(parent_sizes)),
        "min_agents_per_parent_cluster": int(args.min_agents_per_parent_cluster),
        "allocation": {str(int(cluster)): int(count) for cluster, count in allocation.items()},
        "selection_features": selection_features,
        "random_state": int(args.random_state),
        "max_kmeans_iterations": int(args.max_kmeans_iterations),
        "theory": (
            "Stratified vector quantization in the official-MF contributor space. "
            "Each persona is a local medoid/cell summary, preserving within-cluster heterogeneity."
        ),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    write_method_note(output_dir, metadata)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("[ok] wrote:")
    print(f"  {output_dir / 'cluster_personas.csv'}")
    print(f"  {output_dir / 'agent_roster.csv'}")
    print(f"  {output_dir / 'cluster_summary.csv'}")
    print(f"  {output_dir / 'persona_id_map.csv'}")


if __name__ == "__main__":
    main()
