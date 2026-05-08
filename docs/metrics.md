# Metric definitions

This project reports several accuracies. They should not be mixed.

## Raw majority full accuracy

For each note, each LLM persona agent outputs a binary judgment:

```text
HELPFUL -> 1
NOT_HELPFUL -> 0
```

The raw majority prediction is:

```text
predict Helpful if helpful_votes / total_valid_votes >= 0.5
otherwise predict Not Helpful
```

Accuracy is computed on notes for which a valid majority prediction exists. This is a simple baseline. It is not the official Community Notes algorithm.

## Official-style MF resolved accuracy

This baseline treats LLM agents as raters and notes as items, then applies a simplified Community Notes-style rank-1 biased matrix factorization:

```text
rating_ij ~= global_intercept
            + rater_intercept_i
            + note_intercept_j
            + rater_factor_i * note_factor_j
```

After fitting, notes are assigned to:

```text
CRH  = Currently Rated Helpful
CRNH = Currently Rated Not Helpful
NMR  = Needs More Ratings
```

Only CRH/CRNH notes are counted as resolved. Accuracy is computed only on that resolved subset, and coverage reports how many of the 258 notes received a resolved label.

## Calibrated full nested-CV accuracy

This is our main full-coverage aggregation result. For each note, structured agent outputs are converted into feature vectors:

```text
llm_helpful_share
llm_total_votes
llm_mean_confidence
llm_mean_addresses_core_claim
llm_mean_changes_reader_understanding
llm_mean_note_needed
llm_mean_evidence_strength
llm_misses_key_points_rate
llm_too_minor_rate
equal_cluster_helpful_share
cluster_helpful_share_std
cluster_helpful_share_min
cluster_helpful_share_max
equal_cluster_note_needed
equal_cluster_changes_reader_understanding
equal_cluster_evidence_strength
equal_cluster_misses_key_points_rate
equal_cluster_too_minor_rate
helpful_vote_margin_from_half
helpful_vote_entropy
quality_signal_mean
failure_signal_mean
confidence_weighted_helpful_share
```

A logistic model estimates:

```text
P(Helpful | x) = sigmoid(w^T x)
```

Final full prediction uses a probability threshold selected inside inner CV, never on the held-out outer fold.

## Calibrated resolved nested-CV accuracy

The resolved version uses two thresholds:

```text
if P(Helpful) >= high_threshold:
    resolve as Helpful
elif P(Helpful) <= low_threshold:
    resolve as Not Helpful
else:
    leave unresolved
```

The low/high thresholds are selected in inner CV under a target coverage constraint. Reported accuracy is computed only on the resolved subset selected in the held-out outer folds.

## Nested CV protocol

The final reported calibrated results use outer stratified 5-fold CV.

For each outer fold:

1. Hold out one fold as the test fold.
2. On the remaining training folds, run inner stratified 4-fold CV.
3. Inner CV selects model family, logistic regularization strength, class weighting, full threshold, and resolved low/high thresholds.
4. Refit the selected model on the full outer-training split.
5. Evaluate once on the untouched outer-test split.

This prevents the resolved thresholds and model hyperparameters from being tuned on the same examples used for final evaluation.
