param(
    [int]$AgentCount = 24,
    [string]$Python = "python",
    [string]$Model = "gpt-5.4-nano",
    [int]$Concurrency = 8,
    [string]$ModelTag = "gpt54nano",
    [string]$DateTag = "20260507",
    [string]$RunTag = "run1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is not set. Set it before running LLM evaluation."
}

$Variant = "mf_continuous_n{0:D3}" -f $AgentCount
$PersonaDir = "artifacts/agent_variants/$Variant"
$OutputDir = "artifacts/llm_runs/${Variant}_${ModelTag}_${DateTag}_${RunTag}"

& $Python "src/run_llm_persona_multiagent_eval.py" `
  --model $Model `
  --dataset-csv "data/eval_pool_258/eval_claim_pool_258.csv" `
  --ratings-cache-csv "data/eval_pool_258/ratings_cache_258.csv" `
  --persona-labels-file "$PersonaDir/cluster_personas.csv" `
  --persona-summary-file "$PersonaDir/cluster_summary.csv" `
  --agent-roster-file "$PersonaDir/agent_roster.csv" `
  --output-dir $OutputDir `
  --concurrency $Concurrency `
  --resume
