# MF-continuous persona agent construction

这个目录实现的是 `MF-continuous / quantized persona agents`，目的是替代“每个 MF cluster 只写一个平均 persona、再复制多次”的旧做法。

## 理论动机

Community Notes 的官方风格矩阵分解会把 contributor 压缩到连续的判断空间：rater intercept 表示整体宽松/严格，rater factor 表示潜在视角差异，agreement ratio 表示和稳定共识的一致程度。原来的 cluster-average agent 会把同一个 cluster 内的大量异质性压成一个平均人；复制这个平均 persona 只能增加采样噪声，不一定增加真实判断多样性。

因此这里把 agent construction 看成一个代表性采样/向量量化问题：在每个原始 MF cluster 内，对标准化后的 MF 特征和可解释行为特征做 KMeans quantization，再把每个 cell 的 medoid 和局部子群均值写成一个 persona。这样每个 agent 对应真实 contributor 空间中的一个局部代表，而不是纯 prompt 变体。

## 实现细节

- 父 cluster：官方风格 MF 聚类得到的 6 个 cluster。
- 总 agent 数：36。
- 每个父 cluster 最小代表数：3。
- 选择特征：bw_final_rater_intercept, bw_final_rater_factor_1, bw_rater_agree_ratio, bw_mean_note_score, bw_crh_crnh_ratio_difference, share_helpful, share_not_helpful, evidence_focus_rate, strict_rejection_rate, redundancy_rejection_rate, ratings_per_active_day, notes_authored
- allocation：先给每个父 cluster 最小 quota，剩余 agent 按父 cluster 人数比例分配。
- representative：每个父 cluster 内运行 KMeans；每个局部 cell 选择离 centroid 最近的真实 contributor 作为 medoid，同时用 cell 内全部 contributor 的均值生成 persona 画像。

## 文件

- `cluster_personas.csv`：72 个代表 persona，每行一个 system prompt。
- `agent_roster.csv`：兼容现有 multi-agent runner 的 roster；每个代表 persona 的 `agent_count=1`。
- `cluster_summary.csv`：每个代表 persona 的局部子群统计和 medoid 信息。
- `persona_id_map.csv`：代表 persona id 到原始 MF parent cluster 的映射。
- `run_metadata.json`：构造参数。
