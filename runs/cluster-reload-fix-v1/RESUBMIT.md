# Recovery for cluster bootstrap job 851227

Job 851227 installed dependencies and passed its CUDA check, according to the
user's cluster output. Both serialization failures came from implicit automatic
device selection. The recovery bundle changes only the serialization-test reload
to `AircraftPolicy.load(path, device=model.device)` and the bootstrap script.
All other 82 source files are byte-identical to onpolicy-cluster-v1.zip.
The original archive, extracted source, logs and run directories stay in place.

The bootstrap honors ATC_PYTHON: it skips environment creation and all package
installation when an existing interpreter is supplied. It still checks installed
dependencies, reruns the inexpensive CUDA check, runs the 13 tests, then completes
the previously blocked PPO smoke training and evaluation. Each new job uses a new
log and check directory. The allocation remains one H200, 16 CPUs and 64 GB RAM
for eight simulator workers. This is bootstrap validation, not a new research run.

## Windows PowerShell

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\onpolicy-cluster-reload-fix-v1.zip" shri@xlogin.comp.nus.edu.sg:cs4246-rl/
```

## Existing xlogin session

The interpreter below is the environment created by the original onpolicy-v1
bootstrap. The executable check stops before submission if it is absent.
A subshell confines error handling to this block; it does not exit the SSH session.

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl"
printf '%s  %s\n' 'd3c98a95e061d827ab295b1e458dd42e30c308ef2d1c45902b23ce372b6802c3' 'onpolicy-cluster-reload-fix-v1.zip' | sha256sum -c -
export ATC_PYTHON="$PWD/onpolicy-v1/.venv-cluster/bin/python"
test -x "$ATC_PYTHON"
mkdir onpolicy-reload-fix-v1
python3 -m zipfile -e onpolicy-cluster-reload-fix-v1.zip onpolicy-reload-fix-v1
cd onpolicy-reload-fix-v1
python3 -m atc_rl.cluster verify
job_id=$(sbatch --parsable --export=ALL jobs/bootstrap_onpolicy.slurm)
job_id=${job_id%%;*}
printf '%s\n' "$job_id" > submission-job-id.txt
printf 'Submitted job %s. Log: %s/bootstrap-rl-%s.log\n' "$job_id" "$PWD" "$job_id"
squeue -j "$job_id"
)
```

The fresh-directory check intentionally refuses to overlay an existing recovery
source tree. Run this block once. If requeueing later, use a new extraction name
or submit directly from the already verified recovery directory.

## Check the submitted job

```bash
cd "$HOME/cs4246-rl/onpolicy-reload-fix-v1"
job_id=$(cat submission-job-id.txt)
sacct -j "$job_id" --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
tail -n 80 "bootstrap-rl-$job_id.log"
```

The log appears when the job starts. Results live under
`runs/cluster-check-JOBID/` in the recovery directory. Leave
`onpolicy-v1/bootstrap-rl-851227.log`, its source and its results unchanged.

## Local verification

- Corrected original bootstrap suite: 13 passed.
- Current development RL suite: 119 passed.
- Both serialization cases pass an explicit-device regression that rejects the
  original implicit `auto` selection. This is a CPU check, not a CUDA execution.
- Bash syntax passed. Stubbed execution verifies environment reuse, no installation,
  rejection of a missing interpreter, and preservation of existing outputs and environments.
- Both archive manifests and original SHA-256 are verified.

Actual cluster rerun is pending user upload and submission. No replacement Slurm
job has been submitted by this task.
