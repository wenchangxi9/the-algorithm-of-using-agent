# Base user feature input

`user_features_with_behavior_features.csv` is the cached contributor-level behavior table used to define the rater universe and to attach interpretable behavioral features to MF clusters.

It is an input feature table, not the final clustering method. The final clusters used in this project are produced by the official-style matrix-factorization pipeline in `src/cluster_communitynotes_users_matrix_factorized.py`.

For a fully fresh rebuild from raw Community Notes TSVs, regenerate this table from contributor histories first, then run the MF clustering script.
