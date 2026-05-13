# MultiCom promptv4 release bundle

This folder collects the current promptv4 research pipeline for the Community Notes / ComRate multi-agent evaluation experiments.

It is organized as a GitHub-ready bundle: scripts and key derived artifacts are included, while secrets, API keys, the full official Community Notes dump, and very large intermediate files are excluded.

## Folder map

- `src/`: scripts for rater clustering, representative sampling, post fetching, agent prediction, aggregation, nested CV, diagnostics, and ablations.
- `artifacts/01_clustering/`: matrix-factorization rater clustering outputs, including k-selection metrics and the selected k=16 cluster summary.
- `artifacts/02_sampling_20k/`: representative 20k note sample, year/status/topic distribution checks, and post-fetch status files.
- `artifacts/03_success_sample_and_reason_labels/`: successfully fetched notes with posts and reason-label artifacts.
- `artifacts/04_agent_runs_228_promptv4/`: latest promptv4 228-note 16-agent predictions, persona prompts, and run metadata.
- `artifacts/05_agent_runs_2000/`: 2000-note official-schema 16-agent run and nested-CV aggregation summaries.
- `artifacts/06_aggregation_228/`: promptv4 228-note aggregation model search outputs.
- `artifacts/07_ablation_228/`: promptv4 228-note selected feature-group ablation outputs.
- `artifacts/08_feature_ablation_agent_budget/`: feature-combination and agent-budget ablation tables used for paper-style comparisons.
- `docs/`: manifest, large-artifact notes, missing-file notes, and reproduction notes.

## Current 228-note promptv4 headline

The 228-note diagnostic sample is balanced across true labels: 76 Helpful, 76 Needs More Ratings, and 76 Not Helpful.

The selected feature-group ablation currently shows that `vote_plus_confidence` performs best among the tested LR nested-CV feature groups:

- Accuracy / balanced accuracy: 57.02%
- Helpful recall: 61.84%
- Needs More Ratings recall: 46.05%
- Not Helpful recall: 63.16%

See `artifacts/07_ablation_228/summary.csv` and `artifacts/07_ablation_228/best_confusion.csv`.

## Notes

Do not commit API keys or files under the original server `secrets/` directory.

The full official Community Notes raw data is not included in this bundle. Reproduction scripts expect those official TSV/CSV files to be available under the local/server data root used by the project.
