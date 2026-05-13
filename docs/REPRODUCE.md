# Reproduction outline

All commands below assume the server project root:

```bash
cd /data6/wenchangxi/community_note
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL
```

## 1. Rater clustering

```bash
python src/cluster_raters_with_official_mfcore_scorer.py
python src/search_k_existing_mfcore_rater_output.py
```

The current selected clustering uses k=16. See:

```text
artifacts/01_clustering/k_selection_metrics_k2_to_k32.csv
artifacts/01_clustering/cluster_summary_k16.csv
```

## 2. Representative 20k sample

```bash
python src/sample_representative_notes_20k.py
python src/check_semantic_topic_distribution_20k.py
```

The sampling checks are stored in `artifacts/02_sampling_20k/`.

## 3. Post fetching and successful sample

```bash
bash src/run_fetch_sample_posts.sh
python src/check_combined_success_sample.py
python src/attach_reason_labels_to_success_sample.py
```

The successful sample and reason labels are stored in `artifacts/03_success_sample_and_reason_labels/`.

## 4. Promptv4 228-note agent run

```bash
python src/run_16agent_binaryrating_balanced_228.py
```

Key outputs are stored in `artifacts/04_agent_runs_228_promptv4/`.

## 5. Aggregation and nested CV

```bash
python src/explore_binary_agent_tristate_parallel_fast_228.py \
  --run-dir /data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_promptv4_20260513 \
  --jobs 24
```

Outputs are stored in `artifacts/06_aggregation_228/`.

## 6. Feature-group ablation

```bash
python src/run_selected_feature_group_ablation.py \
  --run-dir /data6/wenchangxi/community_note/analysis/llm_16agent_binaryrating_balanced_228_promptv4_20260513 \
  --out-name selected_feature_group_ablation_binary_promptv4_20260513
```

Outputs are stored in `artifacts/07_ablation_228/`.
