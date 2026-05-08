# Method

## 1. Official-style contributor matrix factorization

The first stage starts from the sparse Community Notes rating matrix. Rows are contributors, columns are notes, and observed entries are helpfulness ratings.

The model is a biased rank-1 matrix factorization:

```text
rating_ij ~= mu + alpha_i + beta_j + u_i * v_j
```

where:

- `mu` is the global intercept;
- `alpha_i` is the contributor tendency to rate notes as helpful or not helpful;
- `beta_j` is the note-level intercept;
- `u_i` and `v_j` are one-dimensional latent factors.

The implementation follows the important shape of the Community Notes scoring pipeline:

1. Filter the sparse matrix by minimum ratings per contributor and per note.
2. Fit a preliminary MF model.
3. Use preliminary note states to estimate contributor helpfulness.
4. Filter low-helpfulness contributors.
5. Fit the final MF model.
6. Cluster contributors using final MF parameters and helpfulness features.

This stage is not KMeans over raw behavior features. KMeans is used only after MF, as a discretization step over MF-derived contributor coordinates.

## 2. MF-continuous persona construction

The official-style MF stage gives a small number of parent contributor clusters. Instead of using each parent cluster as one agent, this project constructs a controllable number of agents by preserving the parent clusters and splitting their internal continuous space.

For a target agent count such as 12, 24, 36, or 48:

1. Allocate agent budget across parent MF clusters roughly proportional to cluster size, with a minimum number of representatives per parent cluster.
2. Inside each parent cluster, run vector quantization over MF and behavior dimensions:
   - final rater intercept;
   - final rater latent factor;
   - agreement ratio;
   - mean note score;
   - CRH-CRNH score difference;
   - helpful/not-helpful tendency;
   - evidence and strictness features;
   - activity and authoring features.
3. Select local representative personas.
4. Write a persona prompt and an agent roster for each target agent count.

This makes agent count an experimental variable rather than assuming 72 agents is automatically optimal.

## 3. Structured LLM agent voting

Each persona agent judges every note/post pair and outputs a fixed JSON structure:

```json
{
  "rating": "HELPFUL or NOT_HELPFUL",
  "confidence": 0,
  "addresses_core_claim": 0,
  "changes_reader_understanding": 0,
  "note_needed": 0,
  "evidence_strength": 0,
  "misses_key_points": "YES or NO",
  "too_minor_or_tangential": "YES or NO",
  "rationale": "short reason"
}
```

The binary vote is useful, but the additional dimensions are important because Community Notes helpfulness is not just a popularity vote. A note can receive many positive-looking votes while still being weak if it misses the core claim, adds only a minor correction, or lacks evidence.

## 4. Official-style MF comparison on agent votes

To compare against the Community Notes mechanism, LLM votes can be treated as a sparse note-rater matrix and passed through the same simplified MF resolver:

```text
agent-note votes -> biased rank-1 MF -> CRH / CRNH / NMR
```

This gives an official-style resolved accuracy and coverage. It is intentionally selective: unresolved notes are not counted in accuracy.

## 5. Calibrated aggregation with nested CV

The main improvement is calibrated aggregation. Instead of relying on raw majority vote, the system converts all structured agent outputs into note-level features and trains a logistic model:

```text
P(Helpful | x) = sigmoid(w^T x)
```

The feature vector includes vote share, confidence, quality signals, failure signals, cluster-level disagreement, entropy, and confidence-weighted helpful share.

Evaluation uses nested cross-validation:

- outer 5-fold CV estimates final performance;
- inner 4-fold CV selects model family, regularization, class weighting, and thresholds;
- the held-out outer fold is never used for threshold selection.

This is why the calibrated result is a stronger scientific claim than just trying many thresholds and reporting the best one.
