param(
    [string]$Python = "python",
    [string]$InputFeatures = "artifacts/persona_inputs/user_features_with_mf_persona_clusters.csv",
    [string]$OutputRoot = "artifacts/agent_variants",
    [string]$AgentCounts = "12,24,36,48,72",
    [int]$MinAgentsPerParentCluster = 3
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

& $Python "src/build_mf_continuous_agent_count_variants.py" `
  --input-features $InputFeatures `
  --output-root $OutputRoot `
  --agent-counts $AgentCounts `
  --min-agents-per-parent-cluster $MinAgentsPerParentCluster `
  --skip-existing
