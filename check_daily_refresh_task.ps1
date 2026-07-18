$taskName = "AP Flood EWS Daily Refresh"

$task = Get-ScheduledTask `
    -TaskName $taskName `
    -ErrorAction SilentlyContinue

if ($null -eq $task) {
    Write-Host "Daily refresh task is NOT installed." -ForegroundColor Yellow
    exit 1
}

$info = Get-ScheduledTaskInfo -TaskName $taskName

[PSCustomObject]@{
    TaskName       = $taskName
    State          = $task.State
    LastRunTime    = $info.LastRunTime
    LastTaskResult = $info.LastTaskResult
    NextRunTime    = $info.NextRunTime
} | Format-List

$log = "C:\Users\user\Desktop\flood_monitor_clean\outputs\reports\daily_refresh_task.log"
if (Test-Path $log) {
    Write-Host "`nLatest log lines:" -ForegroundColor Cyan
    Get-Content $log -Tail 25
}
