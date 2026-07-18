$taskName = "AP Flood EWS Daily Refresh"

if (-not (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) {
    throw "Task is not installed. Run install_daily_refresh_task.ps1 first."
}

Start-ScheduledTask -TaskName $taskName
Write-Host "Daily refresh started in the background." -ForegroundColor Green
Write-Host "Check status with .\check_daily_refresh_task.ps1" -ForegroundColor Cyan
