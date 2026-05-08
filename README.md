# Community Notes MF-Calibrated Multi-Agent Pipeline

这个仓库整理的是一条主线算法流程：

`Community Notes raw TSV -> official-style matrix factorization clustering -> MF-continuous persona agents -> LLM multi-agent voting -> official-style MF comparison -> calibrated aggregation with nested CV`

仓库里当前可复算的完整 agent 数量为 12、24、36、48；72-agent 的 persona variant 已保留，但是我还没有算

## Core results

当前 258-note evaluation pool 上的核心结果如下。注意各类指标的分母和含义不完全一样：

| agents | raw majority full accuracy | probability sampling full accuracy, mean | MC 95% CI | official-style MF resolved accuracy | official coverage | calibrated full nested-CV accuracy | calibrated resolved accuracy @~65% coverage | calibrated coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 75.69% | 73.90% | [73.85%, 73.95%] | 83.33% | 66 / 258 | 78.68% | 90.17% | 173 / 258 |
| 24 | 74.81% | 74.81% | [74.77%, 74.84%] | 80.88% | 68 / 258 | 81.01% | 91.62% | 167 / 258 |
| 36 | 74.03% | 74.25% | [74.22%, 74.28%] | 80.30% | 66 / 258 | 82.17% | 89.70% | 165 / 258 |
| 48 | 74.03% | 73.82% | [73.80%, 73.85%] | 83.16% | 95 / 258 | 84.11% | 93.49% | 169 / 258 |

Interpretation:

- `raw majority full accuracy`:直接把 agent 的 Helpful/Not Helpful 多数票当预测。
- `probability sampling full accuracy`:把每个 MF 父 cluster 建模为概率评分者，用 LLM rating 和 confidence 得到 `P(Helpful)`，再在同一 agent budget 下做 5000 次 Binomial Monte Carlo 采样。表中的 95% CI 只反映 Monte Carlo 采样随机性。
- `official-style MF resolved accuracy`:用 rank-1 MF 和阈值模拟 Community Notes 的 resolved 机制，只在 resolved subset 上计算准确率。
- `calibrated full nested-CV accuracy`:把 agent 输出的结构化信号聚合成特征，用外层 5-fold nested CV 在全量 258 条 note 上评估。
- `calibrated resolved accuracy`:同一个 calibrated model 加低/高阈值，只对模型最确定的 note 给 resolved 判断。

## Repository layout

```text
src/
  cluster_communitynotes_users_matrix_factorized.py
  build_matrix_factorized_multiagent_inputs.py
  build_mf_continuous_persona_agents.py
  build_mf_continuous_agent_count_variants.py
  run_llm_persona_multiagent_eval.py
  analyze_llm_agent_count_ablation_official.py
  optimize_llm_multiagent_aggregation.py
  evaluate_probability_sampling.py
  summarize_core_results.py

scripts/
  01_run_mf_clustering.ps1
  02_build_agent_variants.ps1
  03_run_llm_agents.ps1
  04_run_official_style_eval.ps1
  05_run_nested_cv_and_summary.ps1

data/
  raw_communitynotes/
  base_user_features/
  eval_pool_258/

artifacts/
  mf_clustering/
  persona_inputs/
  agent_variants/
  llm_runs/
  official_style_results/
  calibrated_aggregation/
  comparison_tables/

docs/
  method.md
  metrics.md
```

## Setup

```powershell
cd "C:\community note\communitynotes_mf_calibrated_pipeline"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
git lfs install
```

The repository uses Git LFS for CSV/TSV/JSON artifacts because several reproducibility files are tens of MB.

## Reproduce from packaged artifacts

To recompute calibrated nested CV and the final comparison table from the included LLM votes:

```powershell
.\scripts\05_run_nested_cv_and_summary.ps1 -AgentCounts "12,24,36,48"
```

The regenerated summary is written to:

```text
artifacts/comparison_tables/core_results_summary.csv
```

To recompute the probability-sampling comparison:

```powershell
python src/evaluate_probability_sampling.py --agent-counts "12,24,36,48" --repeats 5000 --seed 42
```

The regenerated sampling summaries are written to:

```text
artifacts/comparison_tables/probability_sampling_summary.csv
artifacts/comparison_tables/probability_sampling_repeats.csv
```

## Run a new LLM evaluation

Set your API environment first:

```powershell
$env:OPENAI_API_KEY = "..."
$env:OPENAI_BASE_URL = "..."   # optional, only if using a compatible endpoint
```

Then run one agent-count variant:

```powershell
.\scripts\03_run_llm_agents.ps1 -AgentCount 24 -Model "gpt-5.4-nano" -Concurrency 8
.\scripts\04_run_official_style_eval.ps1 -AgentCount 24
.\scripts\05_run_nested_cv_and_summary.ps1 -AgentCounts "24"
```

## Rebuild agent variants

The packaged `artifacts/persona_inputs/user_features_with_mf_persona_clusters.csv` is the MF-persona contributor table used to construct agents. To rebuild the 12/24/36/48/72 variants:

```powershell
.\scripts\02_build_agent_variants.ps1 -AgentCounts "12,24,36,48,72"
```

## Rebuild MF clustering from raw Community Notes TSVs

Put the extracted public export under `data/raw_communitynotes/extracted/`, then run:

```powershell
.\scripts\01_run_mf_clustering.ps1
```

This step recomputes the official-style contributor MF parameters and parent clusters. The later persona-construction step uses the cached MF-persona input table unless you also regenerate the contributor behavior/persona feature table.

More detail is in `docs/method.md` and `docs/metrics.md`.
