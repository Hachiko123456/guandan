param(
    [Parameter(Position=0)]
    [ValidateSet('doctor','check','prepare','run','status','watch','fetch','publish','current','prune','help')]
    [string]$Action = 'doctor',
    [string]$Delivery,
    [string]$Qa,
    [string]$RunId,
    [switch]$Build,
    [switch]$Current,
    [switch]$DryRun,
    [switch]$NoWait,
    [switch]$Execute,
    [int]$KeepPrevious = 1
)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo '.venv-kaggle\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Kaggle environment missing. See docs/kaggle_pipeline.md for setup.'
}
$Entry = Join-Path $PSScriptRoot 'kaggle_pipeline.py'
if ($Action -eq 'help') {
    & $Python $Entry --help
    exit $LASTEXITCODE
}
$PipelineArgs = @($Entry, $Action)
if ($Delivery) { $PipelineArgs += @('--delivery', $Delivery) }
if ($Qa) { $PipelineArgs += @('--qa', $Qa) }
if ($RunId) { $PipelineArgs += @('--run-id', $RunId) }
if ($Build) { $PipelineArgs += '--build' }
if ($Current) { $PipelineArgs += '--current' }
if ($DryRun) { $PipelineArgs += '--dry-run' }
if ($NoWait) { $PipelineArgs += '--no-wait' }
if ($Execute) { $PipelineArgs += '--execute' }
if ($Action -eq 'prune') { $PipelineArgs += @('--keep-previous', [string]$KeepPrevious) }
& $Python @PipelineArgs
exit $LASTEXITCODE
