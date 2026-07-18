<# Safe cleanup for C:\Users\user\Desktop\flood_monitor_clean. #>
param(
    [string]$ProjectRoot = "$HOME\Desktop\flood_monitor_clean",
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$quarantine = "$HOME\Desktop\flood_monitor_clean_quarantine_$stamp"

function Get-SizeMB([string]$Path) {
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force `
        -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return [math]::Round($sum / 1MB, 2)
}

function Quarantine([string]$Source, [string]$Category) {
    if (-not (Test-Path -LiteralPath $Source)) { return }
    if (-not $Apply) {
        Write-Host "WOULD MOVE: $Source" -ForegroundColor Yellow
        return
    }
    $destinationDir = Join-Path $quarantine $Category
    New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
    Move-Item -LiteralPath $Source -Destination $destinationDir -Force
    Write-Host "MOVED: $Source" -ForegroundColor Green
}

Write-Host ""
Write-Host "AP Flood EWS clean-folder maintenance" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Current size: $(Get-SizeMB $ProjectRoot) MB"
Write-Host "Mode: $(if ($Apply) {'APPLY'} else {'PREVIEW'})"

if ($Apply) {
    New-Item -ItemType Directory -Path $quarantine -Force | Out-Null
    Write-Host "Quarantine: $quarantine" -ForegroundColor Cyan
}

$tempFolders = @(
    "cnn_pipeline_new", "cnn_tf213_fix", "lstm_pipeline_new",
    "multi_district_new", "live_map_alert_new", "freshness_upgrade_new",
    "dashboard_live_new", "dashboard_ui_fix_new", "docs_update",
    "backup_before_ui_fix", "cleanup_reports", "legacy"
)
foreach ($name in $tempFolders) {
    Quarantine (Join-Path $ProjectRoot $name) "temporary_and_backups"
}

$checkpoints = @(
    "models\cnn_checkpoints",
    "models\lstm_checkpoints",
    "models\final_v1"
)
foreach ($relative in $checkpoints) {
    Quarantine (Join-Path $ProjectRoot $relative) "training_checkpoints"
}

$cacheDirs = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -Directory -Force `
    -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -in @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints")
    }
foreach ($dir in $cacheDirs) {
    if ($Apply) {
        Remove-Item -LiteralPath $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "DELETED CACHE: $($dir.FullName)" -ForegroundColor DarkGreen
    } else {
        Write-Host "WOULD DELETE CACHE: $($dir.FullName)" -ForegroundColor DarkYellow
    }
}

$cacheFiles = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Force `
    -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in @(".pyc", ".pyo") }
foreach ($file in $cacheFiles) {
    if ($Apply) { Remove-Item -LiteralPath $file.FullName -Force }
}

if ($Apply) {
    $reportDir = Join-Path $ProjectRoot "maintenance_reports"
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    $files = Get-ChildItem -LiteralPath $ProjectRoot -Recurse -File -Force `
        -ErrorAction SilentlyContinue | Where-Object {
            $_.FullName -notmatch "\\maintenance_reports\\" -and
            $_.FullName -notmatch "\\.git\\"
        }
    $rows = foreach ($group in ($files | Group-Object Length | Where-Object Count -gt 1)) {
        foreach ($file in $group.Group) {
            try {
                $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
                [PSCustomObject]@{ Hash=$hash; SizeKB=[math]::Round($file.Length/1KB,2); Path=$file.FullName }
            } catch { Write-Warning "Could not hash $($file.FullName)" }
        }
    }
    $rows | Group-Object Hash | Where-Object Count -gt 1 | ForEach-Object { $_.Group } |
        Sort-Object Hash, Path | Export-Csv `
        (Join-Path $reportDir "exact_duplicates.csv") -NoTypeInformation

    Write-Host "Final size: $(Get-SizeMB $ProjectRoot) MB" -ForegroundColor Green
    Write-Host "Quarantine: $quarantine" -ForegroundColor Green
    Write-Host "Duplicate report: $reportDir\exact_duplicates.csv" -ForegroundColor Cyan
    Write-Host "No arbitrary duplicate files were deleted." -ForegroundColor Cyan
} else {
    Write-Host "Preview complete. No files were changed." -ForegroundColor Cyan
    Write-Host "Run the same command with -Apply after reviewing the list." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Protected: final models, thresholds, scaler, datasets, district results," -ForegroundColor Cyan
Write-Host "maps, dashboard, README, requirements, SMS configuration, and source code."
