# PPO exploration study on the university cluster

Status: prepared and locally checked; not submitted. This is a separate six-run study. Keep the completed baseline, SAC check, static-support study, and local evaluations intact.

The study trains shared multi-aircraft PPO for one million live aircraft transitions per run on three fresh seeds (52600, 52700, 52800). For each seed it compares gSDE resampling every decision with resampling every twelve decisions. Both arms start with identical complete policy tensors. Each retains route guidance and static-area filtering while aircraft-conflict filtering stays off. The main question is whether learning improves safety and flight performance beyond each run's own untrained policy.

Each GPU task requests one H200, 16 CPUs, and 64 GB, with eight isolated simulator workers. Only one GPU task from this study runs at once. CPU evaluations use separate normal-partition allocations (2 CPUs, 8 GB). The existing `onpolicy-v1/.venv-cluster` environment is reused; no dependency installation is needed. The first training row runs the packaged tests, checks explicit CUDA reload, and verifies 180 physical metrics against the original harness before releasing the remaining training rows.

The final analysis includes all six runs and checkpoints at initial, approximately 100k, approximately 300k, and the full budget. It checks initial tensor identity, scenario pairing, native scoring, source/model hashes, and actual optimizer steps. Per-seed learning curves, paired world intervals, and variability across training seeds are retained. Twenty reused development worlds are a screening set; no official or reserved unseen scenarios are consumed. This extends a promising but outcome-selected pilot that failed the restricted-area criterion. Eight workers, a larger batch, and CUDA differ from the local pilot, so a cross-study change cannot be attributed to budget alone.

Validation before delivery: 164 packaged RL tests passed; four Bash scripts passed syntax checks; a local fake scheduler verified all five submissions and their dependencies, refused duplicate launches, and stopped after first/partial/malformed submission failures. All 135 original gSDE source files are byte-identical in the new 143-file bundle. Actual gSDE CUDA execution is checked by the first scheduled job; local tests used CPU.

## Upload from Windows PowerShell

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\cluster-ppo-gsde-v1.zip" shri@xlogin.comp.nus.edu.sg:cs4246-rl/
if ($LASTEXITCODE -ne 0) { throw "Upload failed; retain the output." }
```

## Extract and submit in xlogin

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl"
test -x "$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python"
python3 - <<'PY'
from pathlib import Path
import hashlib, zipfile
archive = Path("cluster-ppo-gsde-v1.zip")
expected = "8a9a4496f94323cf896ca0b51c6d27fd1cb4a738b4ed484005ac2f20121818bd"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, "Upload checksum mismatch"
destination = Path("cluster-ppo-gsde-v1")
destination.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as bundle:
    root = destination.resolve()
    assert all((root / name).resolve().is_relative_to(root) for name in bundle.namelist())
    bundle.extractall(destination)
PY
cd cluster-ppo-gsde-v1
bash jobs/submit_ppo_gsde_v1.sh
)
```

Paste the printed `submissions.tsv` table back into this task. There are five scheduler submissions: first training row, remaining five training rows, first evaluation row, remaining evaluation rows, and final analysis. The rows within arrays are separate jobs. Completed application checks, not disappearance from `squeue`, establish completion. No commands here cancel or change another study.

If any submission fails, preserve the output and the existing directory. The wrapper intentionally refuses a second launch after partial submission, so recovery must inspect the recorded job IDs instead of submitting duplicates.

## Check results in xlogin

```bash
(
cd "$HOME/cs4246-rl/cluster-ppo-gsde-v1" || exit 1
cat runs/cluster-ppo-gsde-v1/submissions.tsv
job_ids=$(awk 'NR>1 {print $2}' runs/cluster-ppo-gsde-v1/submissions.tsv | paste -sd, -)
if [ -n "$job_ids" ]; then squeue -j "$job_ids" || true; fi
python3 - <<'PY'
from pathlib import Path
import json
root = Path("runs/cluster-ppo-gsde-v1")
plan = json.loads(Path("jobs/ppo_gsde_v1.json").read_text())
for row in plan["rows"]:
    folder = root / f"seed-{row['seed']}" / row["arm"]
    summary = folder / "train/training_summary.json"
    state = json.loads(summary.read_text()) if summary.exists() else {}
    print(row, {"training_status": state.get("status", "not complete"),
                "live_transitions": state.get("live_transitions"),
                "train_checks_complete": (folder / "train-complete.json").exists(),
                "evaluation_complete": (folder / "evaluate-complete.json").exists()})
result = root / "three-seed-results.json"
if result.exists():
    data = json.loads(result.read_text())
    for arm, values in data["fresh_seed_aggregate"].items():
        print(arm, json.dumps({"final_metrics": values["mean_physical_metrics"]["final"],
                               "own_initial_learning": values["own_initial_learning"],
                               "screen_checks": values["screen_checks_on_fresh_seed_mean"]}))
else:
    print("The complete six-run analysis is not available yet.")
PY
for log in ppo-gsde-summary-*.log; do
    if [ -f "$log" ]; then tail -n 20 "$log"; fi
done
)
```

## Preserve and download the complete results

After the six runs and final analysis finish, create an archive in xlogin:

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl/cluster-ppo-gsde-v1"
test -f runs/cluster-ppo-gsde-v1/three-seed-results.json
archive="$HOME/cs4246-rl/ppo-gsde-results-v1.tar.gz"
test ! -e "$archive"
tar --exclude='*/workers' --exclude='*/tests' --exclude='*/simulator' \
    -czf "$archive" cluster-manifest.json jobs/*gsde_v1* \
    scripts/analyze_rl_cohort.py ppo-gsde-*.log runs/cluster-ppo-gsde-v1
tar -tzf "$archive" >/dev/null
sha256sum "$archive"
)
```

Then download in Windows PowerShell:

```powershell
$gsdeResults = "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\ppo-gsde-results-v1.tar.gz"
if (Test-Path -LiteralPath $gsdeResults) { throw "Preserve the existing archive for inspection." }
scp -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg:cs4246-rl/ppo-gsde-results-v1.tar.gz $gsdeResults
if ($LASTEXITCODE -ne 0) { throw "Download failed; retain the output." }
Get-FileHash -Algorithm SHA256 -LiteralPath $gsdeResults
```

Compare the local and cluster SHA256 values and share the cluster value. The archive includes checkpoints, exact metrics, source records, learning curves, and logs; large simulator caches are omitted. Passwords stay in the SSH prompts.
