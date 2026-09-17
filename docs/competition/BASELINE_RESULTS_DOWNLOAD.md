# Retrieve the completed cluster PPO baseline

The receipt currently contains the terminal transcript for jobs 851486–851489. Download the original checkpoints, CSVs, curves and complete logs for an independent audit. These commands leave the experiment source and results intact and reuse the existing cluster installation.

In your existing xlogin shell:

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl/cluster-ppo-baseline-v1"
archive="$HOME/cs4246-rl/ppo-results-851486.tar.gz"
test ! -e "$archive"
tar --exclude='*/workers' --exclude='*/tests' --exclude='*/simulator' \
    -czf "$archive" cluster-manifest.json jobs/ppo_baseline_v1.json \
    ppo-baseline-*.log runs/cluster-ppo-baseline-v1
tar -tzf "$archive" >/dev/null
sha256sum "$archive"
)
```

If the archive already exists, the command stops without overwriting it. Download that archive instead; keep any reported error if its integrity check fails.

Then run in Windows PowerShell:

```powershell
$baselineResults = "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\ppo-results-851486.tar.gz"
if (Test-Path -LiteralPath $baselineResults) { throw "Results archive already exists; preserve it for inspection." }
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:cs4246-rl/ppo-results-851486.tar.gz $baselineResults
if ($LASTEXITCODE -ne 0) { throw "Download failed; retain the output for diagnosis." }
Get-FileHash -Algorithm SHA256 -LiteralPath $baselineResults
```

Compare the two SHA256 values and share the cluster value in this task. Passwords stay in the SSH prompts. Once downloaded to the repository, the archive can be inspected locally without pasting its full contents.

The next, separate cluster study is described in STATIC_FILTER_CLUSTER.md. Do not rerun the completed baseline to collect its artifacts.
