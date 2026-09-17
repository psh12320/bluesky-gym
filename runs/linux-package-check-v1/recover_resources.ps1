$ErrorActionPreference = 'Stop'
$repoRoot = 'C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym'
$runtimeRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot 'runs\linux-runtime-v1')).Path
if ($runtimeRoot -ne (Join-Path $repoRoot 'runs\linux-runtime-v1')) { throw 'Unexpected runtime path' }
$targets = @('.venv', 'tmp') | ForEach-Object { (Resolve-Path -LiteralPath (Join-Path $runtimeRoot $_)).Path }
foreach ($target in $targets) {
    if (-not $target.StartsWith($runtimeRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Cleanup target outside runtime' }
    if ((Split-Path -Leaf $target) -notin @('.venv','tmp')) { throw 'Unexpected cleanup target' }
    if ((Get-Item -LiteralPath $target -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Cleanup root is a link' }
}
$installState = Get-Content -LiteralPath (Join-Path $runtimeRoot 'offline-install-state.json') -Raw | ConvertFrom-Json
if ($installState.status -ne 'failed' -or $installState.pip_exit_code -eq 0) { throw 'Installer must have stopped before cleanup' }
$auditPath = Join-Path $runtimeRoot 'resource-recovery.json'
if (Test-Path -LiteralPath $auditPath) { throw 'Preserve previous recovery evidence' }
$audit = [ordered]@{ started_at_utc = [DateTime]::UtcNow.ToString('o'); status = 'recovering'; reason = 'Local disk below 600 MB and severe memory pressure'; installer_stop = 'Project installer PID 333 verified by exact runtime path, then SIGINT; tool confirmed exit 1'; cache_helper_stop = 'Project helper PID 5700 verified by script path, then SIGTERM; tool confirmed exit 1'; free_disk_before = (Get-PSDrive C).Free; removed_disposable_paths = $targets; preserved = @('all Windows models and evaluations', 'Linux package wheelhouse and published hashes', 'Python runtime archive', 'failed installation logs and reports'); offline_install_state_sha256 = (Get-FileHash -LiteralPath (Join-Path $runtimeRoot 'offline-install-state.json') -Algorithm SHA256).Hash.ToLower(); native_simulator_validation = 'not performed' }
$audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding utf8
foreach ($target in $targets) { Remove-Item -LiteralPath $target -Recurse -Force }
$audit.status = 'complete'
$audit.completed_at_utc = [DateTime]::UtcNow.ToString('o')
$audit.free_disk_after = (Get-PSDrive C).Free
$audit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $auditPath -Encoding utf8
$audit | ConvertTo-Json -Depth 5
