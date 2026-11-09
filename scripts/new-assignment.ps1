<#
.SYNOPSIS
    Create a new assignment notebook from template.

.EXAMPLE
    .\scripts\new-assignment.ps1 -Number 2 -Title "Text Classification"
#>

param(
    [Parameter(Mandatory=$true)]
    [int]$Number,

    [Parameter(Mandatory=$true)]
    [string]$Title
)

$repoRoot = Split-Path $PSScriptRoot -Parent
$folder = "assignment-{0:D2}" -f $Number
$destDir = Join-Path $repoRoot "assignments\$folder"
$nbPath = Join-Path $destDir "assignment_{0:D2}.ipynb" -f $Number

New-Item -ItemType Directory -Path $destDir -Force | Out-Null

$notebook = @"
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment $Number — $Title\n",
    "**Course:** NLP & Computer Vision (PGCP-AI Aug 2026)\n",
    "**Student:** abs6187 (23f2000876@ds.study.iitm.ac.in)\n",
    "\n",
    "---\n",
    "\n",
    "## TODO: Add your work below"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Your code here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"@

$notebook | Out-File -FilePath $nbPath -Encoding utf8

Write-Host "✅ Created: assignments\$folder\assignment_$($Number.ToString('D2')).ipynb" -ForegroundColor Green

