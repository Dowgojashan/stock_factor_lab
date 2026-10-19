# Build the thesis: XeLaTeX -> biber -> XeLaTeX x2.  Run in this folder:  .\build.ps1
# Requires MiKTeX with xelatex/biber on PATH (open a NEW PowerShell after installing).
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
xelatex -interaction=nonstopmode -halt-on-error main.tex
biber main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
Write-Host "Done: $PSScriptRoot\main.pdf"
