param(
    [string]$Delivery,
    [switch]$DryRun,
    [switch]$NoWait,
    [switch]$Watch,
    [switch]$Status,
    [string]$RunId,
    [switch]$Prune,
    [switch]$ExecutePrune,
    [ValidateRange(0, 20)]
    [int]$KeepPrevious = 1
)

$ErrorActionPreference = 'Stop'
$Pipeline = Join-Path $PSScriptRoot 'kaggle.ps1'
if (-not (Test-Path -LiteralPath $Pipeline -PathType Leaf)) {
    throw "Kaggle pipeline entry is missing: $Pipeline"
}

$Modes = @($Watch.IsPresent, $Status.IsPresent, $Prune.IsPresent) | Where-Object { $_ }
if ($Modes.Count -gt 1) {
    throw 'Choose only one of -Watch, -Status, or -Prune.'
}
if (($Watch -or $Status) -and -not $RunId) {
    throw '-Watch and -Status require -RunId.'
}
if ($ExecutePrune -and -not $Prune) {
    throw '-ExecutePrune requires -Prune.'
}

if ($Prune) {
    $Arguments = @('prune', '-KeepPrevious', $KeepPrevious)
    if ($ExecutePrune) { $Arguments += '-Execute' }
    & $Pipeline @Arguments
    exit $LASTEXITCODE
}
if ($Watch) {
    & $Pipeline watch -RunId $RunId
    exit $LASTEXITCODE
}
if ($Status) {
    & $Pipeline status -RunId $RunId
    exit $LASTEXITCODE
}

if (-not $DryRun) {
    & $Pipeline check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Arguments = @('run')
if ($Delivery) { $Arguments += @('-Delivery', $Delivery) } else { $Arguments += '-Current' }
if ($DryRun) { $Arguments += '-DryRun' }
if ($NoWait) { $Arguments += '-NoWait' }
& $Pipeline @Arguments
exit $LASTEXITCODE
