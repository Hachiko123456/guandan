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
    if ($ExecutePrune) {
        & $Pipeline prune -KeepPrevious $KeepPrevious -Execute
    } else {
        & $Pipeline prune -KeepPrevious $KeepPrevious
    }
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

if ($Delivery) {
    if ($DryRun) {
        & $Pipeline run -Delivery $Delivery -DryRun
    } elseif ($NoWait) {
        & $Pipeline run -Delivery $Delivery -NoWait
    } else {
        & $Pipeline run -Delivery $Delivery
    }
} else {
    if ($DryRun) {
        & $Pipeline run -Current -DryRun
    } elseif ($NoWait) {
        & $Pipeline run -Current -NoWait
    } else {
        & $Pipeline run -Current
    }
}
exit $LASTEXITCODE
