param(
    [string]$AgentCounts = "12,24,36,48",
    [string]$Python = "python",
    [string]$ModelTag = "gpt54nano",
    [string]$DateTag = "20260507",
    [string]$RunTag = "run1",
    [int]$Folds = 5,
    [int]$InnerFolds = 4,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Counts = $AgentCounts.Split(",") | ForEach-Object { [int]$_.Trim() } | Where-Object { $_ -gt 0 }

foreach ($Count in $Counts) {
    $Variant = "mf_continuous_n{0:D3}" -f $Count
    $RunDir = "artifacts/llm_runs/${Variant}_${ModelTag}_${DateTag}_${RunTag}"
    $OutputDir = "artifacts/calibrated_aggregation/${Variant}_optimized_aggregation_${DateTag}"

    & $Python "src/optimize_llm_multiagent_aggregation.py" `
      --note-predictions-csv "$RunDir/note_predictions.csv" `
      --agent-votes-csv "$RunDir/agent_votes.csv" `
      --output-dir $OutputDir `
      --folds $Folds `
      --inner-folds $InnerFolds `
      --seed $Seed
}

& $Python "src/summarize_core_results.py" `
  --agent-counts $AgentCounts `
  --model-tag $ModelTag `
  --date-tag $DateTag `
  --run-tag $RunTag
