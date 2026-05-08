# MF-continuous agent-count variants

这个目录用于比较不同 agent budget 下的 MF-continuous persona construction。它不是把 72 个 agent 当成默认最优，而是把 agent 数量本身作为 ablation 变量。

默认生成的 agent 数量为：12, 24, 36, 48, 72, 96, 120。

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
