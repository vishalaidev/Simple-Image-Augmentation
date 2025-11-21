$ErrorActionPreference = "Stop"

$AppName = "djsvs-ai-augment"
$Python = $env:PYTHON_BIN
if ([string]::IsNullOrWhiteSpace($Python)) { $Python = "python" }

Write-Host "Using interpreter: $Python"

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
  Write-Host "pyinstaller not found. Install with: $Python -m pip install pyinstaller"
  exit 1
}

pyinstaller --noconfirm `
  --noconsole `
  --onefile `
  --hidden-import PIL._tkinter_finder `
  --name $AppName `
  augment.py

Write-Host "Built binary: dist\$AppName.exe"
