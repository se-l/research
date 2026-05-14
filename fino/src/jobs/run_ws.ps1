
# run_pricer.ps1

$ErrorActionPreference = "Stop"

$projectDir = "C:/repos/research/fino"
$scriptPath = "$projectDir/src/jobs/run_ws.jl"

Write-Host "[run_ws] Starting..." -ForegroundColor Cyan

# Run Julia directly — inherits console so all output is visible
$process = Start-Process -FilePath "julia" `
    -ArgumentList "--threads=auto --project=`"$projectDir`" `"$scriptPath`"" `
    -NoNewWindow `
    -PassThru

Write-Host "[run_ws] Process started (PID $($process.Id)). Press Ctrl+C to stop." -ForegroundColor Cyan

try {
    $process.WaitForExit()
} finally {
    if (-not $process.HasExited) {
        Write-Host "`n[run_ws] Stopping WS..." -ForegroundColor Red
        $process.Kill()
    }
    Write-Host "[run_ws] WS exited with code $($process.ExitCode)." -ForegroundColor Cyan
}