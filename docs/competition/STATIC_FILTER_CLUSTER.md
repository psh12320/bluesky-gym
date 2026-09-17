# Cluster PPO static-support experiment

Status: ready for submission. All 149 automated checks and the isolated simulator integration checks passed. No performance jobs have been submitted from this workspace.

This study trains PPO at one million live aircraft transitions with eight simulator worlds, fixed route guidance, and the existing static-area projection. The aircraft-to-aircraft conflict filter is off. Three seeds (50800, 50900, 51000) each compare real closest-approach inputs with identical-size zero channels and identical initial network weights. Each run evaluates its own untrained policy, 100k and 300k checkpoints, final policy, and the unchanged classical benchmark on twenty development worlds.

This tests learning and predictive inputs within static support. It does not by itself isolate the effect of static support, establish an official competition score, or provide unseen-scenario evidence.

Training requests one H200, 16 CPUs and 64 GB; eight CPUs serve simulator workers. Evaluations request two CPUs and 8 GB on the normal partition. The first training row gates the remaining five on technical success. Other GPU tasks run one at a time. The existing virtual environment is reused; no dependency reinstall is performed. Existing jobs and source directories are untouched.

## Upload from Windows PowerShell

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\cluster-ppo-static-v1.zip" shri@xlogin.comp.nus.edu.sg:cs4246-rl/
```

## Submit from the existing xlogin shell

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl"
test -x "$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python"
python3 - <<'PY'
from pathlib import Path
import hashlib, zipfile
archive = Path("cluster-ppo-static-v1.zip")
expected = "a0ad40c1e209cc7d33675e20250f02348a3f4d2e90b903ac69f0c1ba77dafb53"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, "Upload checksum mismatch"
destination = Path("cluster-ppo-static-v1")
destination.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(destination)
PY
cd cluster-ppo-static-v1
bash jobs/submit_ppo_static_v1.sh
)
```

The submission script refuses duplicate launches and writes runs/cluster-ppo-static-v1/submissions.tsv. If any step fails, keep its output and directory for diagnosis rather than deleting it and repeating the command.

## Inspect completion on xlogin

```bash
(
cd "$HOME/cs4246-rl/cluster-ppo-static-v1" || exit 1
cat runs/cluster-ppo-static-v1/submissions.tsv
squeue -u "$USER"
tail -n 8 ppo-static-*.log
)
```

Every row should print a train completion and an evaluate completion. A vanished queue entry alone is not proof of success. The known Slurm accounting error does not change the saved experiment logs.

## Collect all artifacts after evaluation finishes

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl/cluster-ppo-static-v1"
archive="$HOME/cs4246-rl/ppo-static-results-v1.tar.gz"
test ! -e "$archive"
tar --exclude='*/workers' --exclude='*/tests' -czf "$archive" \
    cluster-manifest.json jobs/ppo_static_v1.json ppo-static-*.log \
    runs/cluster-ppo-static-v1
tar -tzf "$archive" >/dev/null
sha256sum "$archive"
)
```

Download from Windows PowerShell:

```powershell
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:cs4246-rl/ppo-static-results-v1.tar.gz "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\ppo-static-results-v1.tar.gz"
```

The bundle includes scripts/analyze_ppo_static.py for auditing all six runs and aggregating within-seed learning, differences between the two input conditions, seed variability, learning curves, and classical comparisons. It rejects incomplete training, changed sources or dependencies, mismatched initialization, altered action support, and unpaired scenarios.
