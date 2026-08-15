# Convenience script to start the CLASSify Desktop dev environment on Windows.
#
# Usage (in PowerShell):
#   .\scripts\dev.ps1              # starts backend only (use Vite + browser for frontend dev)
#   .\scripts\dev.ps1 -Frontend    # starts backend + Vite (frontend dev with hot-reload)
#   .\scripts\dev.ps1 -Shell       # starts desktop shell (native window, no Vite needed)

param(
    [switch]$Frontend,
    [switch]$Shell
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# Ensure Node.js is on PATH (installer may have added it after this shell started)
$nodeDir = "C:\Program Files\nodejs"
if ((Test-Path $nodeDir) -and ($env:PATH -notlike "*$nodeDir*")) {
    $env:PATH = "$nodeDir;$env:PATH"
}

# Set dev environment variables
$env:CLASSIFY_DEV_MODE = "true"
$env:MPLBACKEND = "Agg"

if (-not $env:CLASSIFY_DATA_DIR) {
    $env:CLASSIFY_DATA_DIR = Join-Path $env:TEMP "classify-dev-data"
    Write-Host "Using temp data dir: $env:CLASSIFY_DATA_DIR" -ForegroundColor Cyan
}

if ($Shell) {
    # ── Desktop shell mode (native window) ──
    # Serves the built frontend from the backend — no Vite needed.
    # Run `cd frontend && npm run build` first to get the latest UI.
    Write-Host "Starting CLASSify Desktop (native window)..." -ForegroundColor Green
    $env:CLASSIFY_DEV_MODE = "false"  # shell handles its own mode
    python -m classify_desktop
    exit 0
}

# Start backend (use uvicorn directly — reload mode causes route discovery issues)
Write-Host "Starting backend API on http://127.0.0.1:8000 ..." -ForegroundColor Green
$backend = Start-Process -NoNewWindow -PassThru -FilePath python `
    -ArgumentList "-m", "uvicorn", "classify_api.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $repoRoot

if ($Frontend) {
    Start-Sleep -Seconds 2
    Write-Host "Starting frontend dev server on http://localhost:5173 ..." -ForegroundColor Green
    $npmCmd = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (-not $npmCmd) {
        $npmCmd = (Get-Command npm -ErrorAction SilentlyContinue).Source
    }
    if (-not $npmCmd) {
        Write-Host "npm not found. Install Node.js, then run 'cd frontend; npm ci; npm run dev' manually." -ForegroundColor Red
    } else {
        $frontend = Start-Process -NoNewWindow -PassThru -FilePath $npmCmd `
            -ArgumentList "run", "dev" `
            -WorkingDirectory (Join-Path $repoRoot "frontend")
    }

    Write-Host ""
    Write-Host "Backend:  http://127.0.0.1:8000  (API docs at /api/docs)" -ForegroundColor Yellow
    Write-Host "Frontend: http://localhost:5173  (hot-reload)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press Ctrl+C to stop both." -ForegroundColor Yellow

    $procs = @($backend)
    if ($frontend) { $procs += $frontend }
    try {
        Wait-Process -Id $procs.Id
    } finally {
        foreach ($p in $procs) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Host ""
    Write-Host "Backend running at http://127.0.0.1:8000  (API docs at /api/docs)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "For frontend dev with hot-reload: .\scripts\dev.ps1 -Frontend" -ForegroundColor Cyan
    Write-Host "For native desktop window:        .\scripts\dev.ps1 -Shell" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
    try {
        Wait-Process -Id $backend.Id
    } finally {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}

