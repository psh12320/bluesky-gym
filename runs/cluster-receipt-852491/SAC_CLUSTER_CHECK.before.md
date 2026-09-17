# SAC cluster implementation check

Status: ready for manual submission; no job has been submitted from this workspace. The underlying SAC source passed 201 local RL tests and 558 physical-metric checks. This bundle adds a short cluster check while preserving every original source file byte-for-byte.

The job reuses `$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python`. It installs nothing, requests one H200, 16 CPUs and 64 GB for up to 30 minutes, and uses eight simulator workers. It trains fresh SAC for 20k live transitions, verifies that completed aircraft and padding were recorded, audits checkpoints, reloads the actor explicitly onto CUDA, and checks two reused development worlds through both vector and native evaluation. The resulting 180 physical metrics must match exactly. These checks establish cluster correctness, not competitive performance.

The 100k SAC/PPO development comparison remains a separate local experiment. This short cluster job uses eight gradient rounds per vector decision to preserve its gradient-round ratio per simulator decision with eight workers; its parallel collection schedule differs and its results must not be presented as a matched performance comparison.

## Upload in Windows PowerShell

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\cluster-sac-check-v1.zip" shri@xlogin.comp.nus.edu.sg:cs4246-rl/
if ($LASTEXITCODE -ne 0) { throw "Upload failed; preserve the output for diagnosis." }
```

## Extract and submit in the existing xlogin shell

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl"
test -x "$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python"
python3 - <<'PY'
from pathlib import Path
import hashlib, zipfile
archive = Path("cluster-sac-check-v1.zip")
expected = "f246182da64c516a436c138a389e8f2e575a360c6973c7d353a3f996f71129b9"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, "Upload checksum mismatch"
destination = Path("cluster-sac-check-v1")
destination.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(destination)
PY
cd cluster-sac-check-v1
bash jobs/submit_sac_cluster_check.sh
)
```

The submission wrapper refuses repeated launches using an exclusive lock directory and records the job ID. If extraction or submission fails, keep the output and existing directory for diagnosis. Existing baseline/static-study jobs, checkpoints and environments are unchanged.

## Retrieve the completion report in xlogin

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl/cluster-sac-check-v1"
cat runs/cluster-sac-check-v1/submissions.tsv
job_id=$(awk 'NR==2 {print $2}' runs/cluster-sac-check-v1/submissions.tsv)
case "$job_id" in ''|*[!0-9]*) echo "Missing or invalid saved job ID" >&2; exit 1 ;; esac
squeue -j "$job_id" || true
tail -n 35 "sac-check-${job_id}.log"
result="runs/cluster-sac-check-v1/job-${job_id}/complete.json"
if [ -f "$result" ]; then
    cat "$result"
else
    echo "No completion report yet; retain the log output above."
fi
)
```

Paste that output back into the task. Queue disappearance alone is not success; the completion report and final source verification are required. The known `sacct` accounting failure is avoided here.
