param(
    [string]$Time = "06:00"
)

$ErrorActionPreference = "Stop"

$taskName = "AP Flood EWS Daily Refresh"
$project = "C:\Users\user\Desktop\flood_monitor_clean"
$runner = Join-Path $project "run_daily_refresh.cmd"

if (-not (Test-Path $runner)) {
    throw "Missing runner: $runner"
}

if ($Time -notmatch "^\d{2}:\d{2}$") {
    throw "Use 24-hour HH:mm format, for example 06:00 or 18:30."
}

$timeValue = [datetime]::ParseExact(
    $Time,
    "HH:mm",
    [System.Globalization.CultureInfo]::InvariantCulture
)

$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$runner`"" `
    -WorkingDirectory $project

$trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At $timeValue

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Refresh AP Flood EWS Sentinel-1, weather, CNN, LSTM, hybrid, maps and alerts every 24 hours." `
    -Force |
Out-Null

Write-Host "INSTALLED: $taskName" -ForegroundColor Green
Write-Host "Daily time: $Time" -ForegroundColor Cyan
Write-Host "The computer must be on and connected to the internet." -ForegroundColor Yellow

Get-ScheduledTaskInfo -TaskName $taskName |
Select-Object LastRunTime, LastTaskResult, NextRunTime |
Format-List
