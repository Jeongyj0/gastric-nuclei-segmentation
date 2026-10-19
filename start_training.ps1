[CmdletBinding()]
param([switch]$PlanOnly)

$ErrorActionPreference = 'Stop'
$trainingPython = Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1
$launcher = Join-Path $PSScriptRoot 'start_training.py'
$launcherArguments = @()
if ($PlanOnly) {
    $launcherArguments += '--plan-only'
}
& $trainingPython.Source -X utf8 -B -u $launcher @launcherArguments
exit $LASTEXITCODE
