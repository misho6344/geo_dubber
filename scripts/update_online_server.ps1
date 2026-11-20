$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

Write-Host "Updating GeoDubber from GitHub..."
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw "Git pull failed. The running server was not changed." }

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Online Python environment not found: $python" }

Write-Host "Checking Python syntax..."
& $python -m py_compile "core\gemini_speakers.py" "online_api.py"
if ($LASTEXITCODE -ne 0) { throw "Python syntax check failed. Server was not restarted." }

Write-Host "Checking the new Gemini prompt..."
& $python -c "from core.gemini_speakers import build_prompt; p=build_prompt([{'start':0,'end':1,'text':'ტესტი'}]); assert 'Analyze this Georgian-language audio recording.' in p; assert 'If two different people sound alike, treat them as separate speakers.' in p; assert 'from 1 to 5' in p; print('PROMPT_OK')"
if ($LASTEXITCODE -ne 0) { throw "Gemini prompt verification failed. Server was not restarted." }

$listener = Get-NetTCPConnection -LocalPort 10000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
    if ($oldProcess -and $oldProcess.CommandLine -match "uvicorn" -and $oldProcess.CommandLine -match "online_api:app") {
        Write-Host "Stopping existing GeoDubber API..."
        Stop-Process -Id $listener.OwningProcess -Force
        Start-Sleep -Seconds 2
    } else {
        throw "Port 10000 is occupied by another process. It was not stopped."
    }
}

$logs = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stdout = Join-Path $logs "online-api-update.out.log"
$stderr = Join-Path $logs "online-api-update.err.log"

Write-Host "Starting updated GeoDubber API..."
Start-Process -FilePath $python -ArgumentList @("-m","uvicorn","online_api:app","--host","127.0.0.1","--port","10000") -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden

$healthy = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:10000/healthz" -TimeoutSec 2
        if ($health.status -eq "ok") { $healthy = $true; break }
    } catch {}
}
if (-not $healthy) { throw "Updated API did not become healthy. Check logs\online-api-update.err.log." }

Write-Host "SUCCESS: GeoDubber is running with the new Gemini speaker prompt."
Write-Host "The existing Tailscale Funnel remains in place; no API keys or project data were changed."
