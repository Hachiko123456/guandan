param(
    [switch]$DryRun,
    [switch]$NoWait
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Pipeline = Join-Path $PSScriptRoot 'kaggle.ps1'
$Delivery = Join-Path $Repo 'runs\t4x2_serial_compare_delivery_20261023\delivery_v12'

if (-not (Test-Path -LiteralPath $Pipeline -PathType Leaf)) {
    throw "Kaggle pipeline entry is missing: $Pipeline"
}
if (-not (Test-Path -LiteralPath $Delivery -PathType Container)) {
    throw "Validated v12 delivery is missing: $Delivery"
}

if ($DryRun) {
    & $Pipeline doctor
    if ($LASTEXITCODE -notin 0, 2) { exit $LASTEXITCODE }
    & $Pipeline run -Delivery $Delivery -DryRun
    exit $LASTEXITCODE
}

& $Pipeline check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RunArguments = @('run', '-Delivery', $Delivery)
if ($NoWait) { $RunArguments += '-NoWait' }
& $Pipeline @RunArguments
exit $LASTEXITCODE
