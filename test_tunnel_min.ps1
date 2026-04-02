param(
    [string]$SshHost = "cci-proxy.cn-sh-01.sensecore.cn",
    [int]$Port = 23686,
    [string]$User = "root",
    [string]$Key = "$env:USERPROFILE\.ssh\id_rsa",

    [string]$LlmRemoteIp = "10.119.30.230",
    [string]$EmbRemoteIp = "10.119.27.199"
)

$ErrorActionPreference = "Stop"

function Check-Port([int]$p) {
    (Test-NetConnection 127.0.0.1 -Port $p -WarningAction SilentlyContinue).TcpTestSucceeded
}

Write-Host "== 1) SSH认证测试 =="

if (-not (Test-Path $Key)) {
    Write-Host "[FAIL] 私钥不存在: $Key" -ForegroundColor Red
    exit 1
}

$auth = & ssh -i $Key -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -p $Port "$User@$SshHost" "echo OK" 2>&1
if ($LASTEXITCODE -ne 0 -or ($auth -join "`n") -notmatch "OK") {
    Write-Host "[FAIL] SSH认证失败" -ForegroundColor Red
    $auth | ForEach-Object { Write-Host $_ }
    exit 1
}
Write-Host "[OK] SSH认证通过" -ForegroundColor Green

Write-Host "== 2) 建立隧道并检查端口 =="

# 清理旧 ssh 进程（避免端口占用）
Get-Process ssh -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$proc = Start-Process ssh -ArgumentList @(
    "-N",
    "-i", $Key,
    "-o", "ServerAliveInterval=60",
    "-o", "StrictHostKeyChecking=no",
    "-p", "$Port",
    "-L", "8080`:$LlmRemoteIp`:8000",
    "-L", "9090`:$EmbRemoteIp`:8000",
    "$User@$SshHost"
) -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 3

if ($proc.HasExited) {
    Write-Host "[FAIL] 隧道进程已退出，PID=$($proc.Id)" -ForegroundColor Red
    exit 1
}

$ok8080 = Check-Port 8080
$ok9090 = Check-Port 9090

Write-Host "8080: $ok8080"
Write-Host "9090: $ok9090"

if ($ok8080 -and $ok9090) {
    Write-Host "[OK] 最小测试通过，隧道PID=$($proc.Id)" -ForegroundColor Green
    Write-Host "关闭隧道命令: Stop-Process -Id $($proc.Id) -Force"
    exit 0
} else {
    Write-Host "[FAIL] 端口未就绪，终止隧道" -ForegroundColor Red
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
