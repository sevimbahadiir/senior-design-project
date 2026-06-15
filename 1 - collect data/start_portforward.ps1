$services = @(
    @{ Name = "Prometheus"; Namespace = "istio-system"; Service = "svc/prometheus"; Local = 9090; Remote = 9090; HealthPath = "/-/healthy" },
    @{ Name = "Grafana";    Namespace = "istio-system"; Service = "svc/grafana";    Local = 3000; Remote = 3000; HealthPath = "/api/health" },
    @{ Name = "RobotShop";  Namespace = "robot-shop";   Service = "svc/web";        Local = 8080; Remote = 8080; HealthPath = "/" }
)

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Robot Shop RL -- Port Forward Manager (Health Check)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

$jobs = @()

foreach ($svc in $services) {
    $name       = $svc.Name
    $namespace  = $svc.Namespace
    $service    = $svc.Service
    $local      = $svc.Local
    $remote     = $svc.Remote
    $healthPath = $svc.HealthPath

    Write-Host "  Starting port-forward: $name  ($local -> $remote)" -ForegroundColor Green

    $job = Start-Job -ScriptBlock {
        param($name, $namespace, $service, $local, $remote, $healthPath)

        $pfProcess = $null

        function Start-PortForward {
            if ($pfProcess -and !$pfProcess.HasExited) {
                try { $pfProcess.Kill() } catch {}
            }
            $existing = Get-NetTCPConnection -LocalPort $local -ErrorAction SilentlyContinue
            foreach ($conn in $existing) {
                if ($conn.OwningProcess -gt 0) {
                    try { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue } catch {}
                }
            }
            Start-Sleep -Seconds 2
            $pfArgs = "port-forward -n $namespace $service ${local}:${remote}"
            return Start-Process -FilePath "kubectl" -ArgumentList $pfArgs -PassThru -WindowStyle Hidden
        }

        $pfProcess = Start-PortForward
        Write-Output "[$name] Port-forward started (PID $($pfProcess.Id))"
        Start-Sleep -Seconds 5

        while ($true) {
            Start-Sleep -Seconds 15

            $healthy = $false
            try {
                $url = "http://localhost:${local}${healthPath}"
                $response = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
                if ($response.StatusCode -lt 500) {
                    $healthy = $true
                }
            } catch {}

            if (-not $healthy) {
                $ts = Get-Date -Format "HH:mm:ss"
                Write-Output "[$ts] [$name] Health check failed -- restarting..."
                $pfProcess = Start-PortForward
                Write-Output "[$ts] [$name] Restarted (PID $($pfProcess.Id))"
                Start-Sleep -Seconds 5
            }
        }
    } -ArgumentList $name, $namespace, $service, $local, $remote, $healthPath

    $jobs += $job
}

Write-Host ""
Write-Host "  All port-forwards running with health checks." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Prometheus  -->  http://localhost:9090" -ForegroundColor White
Write-Host "  Grafana     -->  http://localhost:3000" -ForegroundColor White
Write-Host "  Robot Shop  -->  http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "  Health check interval : 15 seconds" -ForegroundColor White
Write-Host "  Auto-restart          : enabled" -ForegroundColor White
Write-Host ""
Write-Host "  Press Ctrl+C to stop all port-forwards." -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

try {
    while ($true) {
        Start-Sleep -Seconds 10
        foreach ($job in $jobs) {
            $output = Receive-Job -Job $job -ErrorAction SilentlyContinue
            foreach ($line in $output) {
                if ($line) {
                    Write-Host "  $line" -ForegroundColor DarkGray
                }
            }
        }
    }
} finally {
    Write-Host ""
    Write-Host "  Stopping all port-forwards..." -ForegroundColor Yellow
    $jobs | Stop-Job
    $jobs | Remove-Job
    Write-Host "  Stopped." -ForegroundColor Green
    Write-Host ""
}