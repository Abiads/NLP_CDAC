<#
.SYNOPSIS
    Organize a new day's zip from Ma'am's CloudFoundry downloads.

.DESCRIPTION
    Downloads are zips like "DAY 2.zip" containing .py files and PDFs.
    This script extracts and organizes them into the notes/ structure.

.EXAMPLE
    .\scripts\add-day.ps1 -ZipPath "C:\Downloads\DAY 2.zip" -DayNumber 2
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ZipPath,

    [Parameter(Mandatory=$true)]
    [int]$DayNumber
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$dayFolder = "day-{0:D2}" -f $DayNumber
$destBase = Join-Path $repoRoot "notes\$dayFolder"
$tempDir = Join-Path $env:TEMP "nlp_cdac_day_$DayNumber"

Write-Host "[Package] Processing Day $DayNumber from: $ZipPath" -ForegroundColor Cyan

# Create target directories
foreach ($sub in @("slides", "code", "pdfs")) {
    New-Item -ItemType Directory -Path (Join-Path $destBase $sub) -Force | Out-Null
}

# Extract zip
if (Test-Path $tempDir) { Remove-Item $tempDir -Recurse -Force }
Expand-Archive -Path $ZipPath -DestinationPath $tempDir -Force
Write-Host "[OK] Extracted to temp directory"

# Find the actual content folder (zips often have a subfolder)
$contentDir = $tempDir
$subDirs = Get-ChildItem $tempDir -Directory
if ($subDirs.Count -eq 1) {
    $contentDir = $subDirs[0].FullName
}

# Copy Python files to code/
$pyFiles = Get-ChildItem $contentDir -Filter "*.py"
foreach ($f in $pyFiles) {
    $newName = $f.Name.ToLower() -replace '\s+', '_'
    Copy-Item $f.FullName (Join-Path $destBase "code\$newName") -Force
    Write-Host "  [FILE] Code: $newName"
}

# Copy PDFs to pdfs/
$pdfFiles = Get-ChildItem $contentDir -Filter "*.pdf"
foreach ($f in $pdfFiles) {
    Copy-Item $f.FullName (Join-Path $destBase "pdfs\") -Force
    Write-Host "  [FILE] PDF: $($f.Name)"
}

# Copy PPTX to slides/
$pptxFiles = Get-ChildItem $contentDir -Filter "*.pptx"
foreach ($f in $pptxFiles) {
    Copy-Item $f.FullName (Join-Path $destBase "slides\") -Force
    Write-Host "  [CHART] Slide: $($f.Name)"
}

# Copy original zip to raw-downloads/
$rawDir = Join-Path $repoRoot "raw-downloads"
New-Item -ItemType Directory -Path $rawDir -Force | Out-Null
Copy-Item $ZipPath $rawDir -Force

# Cleanup temp
Remove-Item $tempDir -Recurse -Force

Write-Host ""
Write-Host "[OK] Day $DayNumber organized at: notes\$dayFolder\" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Add slides (.pptx) manually if they were shared separately"
Write-Host "  2. Create notes\$dayFolder\README.md with topic summary"
Write-Host "  3. git add notes\$dayFolder && git commit"

