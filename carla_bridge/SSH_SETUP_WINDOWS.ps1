# Windows 一键把本机公钥装到 AutoDL 服务器，之后 VSCode/ssh 免密登录。
#
# 用法（Windows PowerShell，不是 CMD）：
#     powershell -ExecutionPolicy Bypass -File SSH_SETUP_WINDOWS.ps1
# 或者显式指定：
#     powershell -ExecutionPolicy Bypass -File SSH_SETUP_WINDOWS.ps1 -HostName region-41.seetacloud.com -Port 38924 -User root
#
# 会做三件事：
#   1) 没有密钥就生成 %USERPROFILE%\.ssh\id_ed25519
#   2) 把公钥追加到服务器 /root/.ssh/authorized_keys（会提示输入一次密码）
#   3) 在 %USERPROFILE%\.ssh\config 里写好 Host 块（含 IdentityFile 和保活）

param(
    [string]$HostName = "region-41.seetacloud.com",
    [int]$Port = 38924,
    [string]$User = "root"
)

$ErrorActionPreference = "Stop"
$sshDir = Join-Path $env:USERPROFILE ".ssh"
$key = Join-Path $sshDir "id_ed25519"
$pub = "$key.pub"

if (!(Test-Path $sshDir)) { New-Item -ItemType Directory -Path $sshDir | Out-Null }

Write-Host "== 1) 检查/生成密钥 ==" -ForegroundColor Cyan
if (!(Test-Path $pub)) {
    Write-Host "   没有密钥，生成 $key ..."
    ssh-keygen -t ed25519 -f $key -N '""' -C "$env:USERNAME@$(hostname)"
} else {
    Write-Host "   已有密钥：$pub"
}
Write-Host ("   公钥指纹: " + (ssh-keygen -lf $pub))

Write-Host "== 2) 安装公钥到服务器（这一步会要一次密码）==" -ForegroundColor Cyan
$remoteCmd = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo INSTALLED_OK"
Get-Content $pub | ssh -p $Port "$User@$HostName" $remoteCmd
if ($LASTEXITCODE -ne 0) { throw "公钥安装失败（密码错了？网络不通？）" }

Write-Host "== 3) 写 ~/.ssh/config ==" -ForegroundColor Cyan
$cfg = Join-Path $sshDir "config"
$block = @"
Host $HostName
  HostName $HostName
  Port $Port
  User $User
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 30
  ServerAliveCountMax 6
"@
if ((Test-Path $cfg) -and (Select-String -Path $cfg -Pattern "Host $HostName" -Quiet)) {
    Write-Host "   config 里已有该 Host，跳过（如需更新请手动改 $cfg）"
} else {
    Add-Content -Path $cfg -Value $block -Encoding ascii
    Write-Host "   已追加到 $cfg"
}

Write-Host "== 4) 验证免密 ==" -ForegroundColor Cyan
ssh -o BatchMode=yes -o ConnectTimeout=15 -p $Port "$User@$HostName" "echo OK_FROM_SERVER; hostname"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 免密成功，现在 VSCode Remote-SSH 连 $HostName 不会再要密码了" -ForegroundColor Green
} else {
    Write-Host "❌ 还不行：检查 sshd 是否允许 PubkeyAuthentication，或看上面的报错" -ForegroundColor Red
}
