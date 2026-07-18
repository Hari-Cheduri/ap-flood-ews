$taskName = "AP Flood EWS Daily Refresh"

Unregister-ScheduledTask `
    -TaskName $taskName `
    -Confirm:$false `
    -ErrorAction SilentlyContinue

Write-Host "Removed: $taskName" -ForegroundColor Green
