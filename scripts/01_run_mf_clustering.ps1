param(
    [string]$Python = "python",
    [string]$DataRoot = "data/raw_communitynotes/extracted",
    [string]$BaseFeatures = "data/base_user_features/user_features_with_behavior_features.csv",
    [string]$OutputDir = "artifacts/mf_clustering"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

& $Python "src/cluster_communitynotes_users_matrix_factorized.py" `
  --data-root $DataRoot `
  --base-features $BaseFeatures `
  --output-dir $OutputDir
