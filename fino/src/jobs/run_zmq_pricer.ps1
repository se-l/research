
$ErrorActionPreference = "Stop"

$projectDir = "C:/repos/research/fino"
$scriptPath = "$projectDir/src/jobs/run_zmq_pricer.jl"

Write-Host "[run_zmq_pricer] Starting $($scriptPath)..." -ForegroundColor Cyan

# Run Julia directly — inherits console so all output is visible
$process = Start-Process -FilePath "julia" `
    -ArgumentList "--threads=auto --project=`"$projectDir`" `"$scriptPath`"" `
    -NoNewWindow `
    -PassThru

Write-Host "[run_zmq_pricer] Process started (PID $($process.Id)). Press Ctrl+C to stop." -ForegroundColor Cyan

try {
    $process.WaitForExit()
} finally {
    if (-not $process.HasExited) {
        Write-Host "`n[run_zmq_pricer] Stopping WS..." -ForegroundColor Red
        $process.Kill()
    }
    Write-Host "[run_zmq_pricer] WS exited with code $($process.ExitCode)." -ForegroundColor Cyan
}