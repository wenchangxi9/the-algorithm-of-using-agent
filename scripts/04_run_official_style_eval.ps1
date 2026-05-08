param(
    [int]$AgentCount = 24,
    [string]$Python = "python",
    [string]$ModelTag = "gpt54nano",
    [string]$DateTag = "20260507",
    [string]$RunTag = "run1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Variant = "mf_continuous_n{0:D3}" -f $AgentCount
$RunDir = "artifacts/llm_runs/${Variant}_${ModelTag}_${DateTag}_${RunTag}"
$OutputDir = "artifacts/official_style_results/${Variant}_${ModelTag}_${DateTag}_${RunTag}"

& $Python "src/analyze_llm_agent_count_ablation_official.py" `
  --votes-csv "$RunDir/agent_votes.csv" `
  --notes-csv "$RunDir/note_predictions.csv" `
  --output-dir $OutputDir `
  --agent-counts $AgentCount `
  --repeats 1
