# Full evaluation of the fixed candidate on Slurm

This job evaluates the two already selected models with the decimal heading-command
correction. It does not train, choose checkpoints, or alter the original scoring
rules. The archived source has passed two-scenario integration checks per track
from a fresh extraction on Windows. The Slurm script and embedded Python have
passed syntax checks; execution on the university cluster remains unverified.

## Transfer the exact release

After the checkout and Python environment described in CLUSTER.md are available,
copy the candidate and the new job script from Windows PowerShell:

```powershell
Set-Location 'C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym'
scp -J shri@sjump.comp.nus.edu.sg .\output\candidates\decimal-heading-v1.zip .\jobs\evaluate_candidate.slurm shri@xlogin.comp.nus.edu.sg:
```

Enter passwords only into the SSH/SCP prompt. No upload or job submission has been
performed here. The archive includes both model weights, deployment manifests,
source files and development checks. The older source-v17 training bundle does
not contain the corrected loader and is not the scoring input for this job.

The exact archive SHA-256 is:

```text
da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c
```

## Submit one CPU job per track

In the authenticated cluster session, replace the partition and account placeholders
with values confirmed by sinfo and sacctmgr. These evaluations use one CPU process
and need no GPU. The template requests 6 GB and 12 hours; confirm that the selected
partition supports those requests. Omit account only if the cluster supplies one.

```bash
cd "$HOME/airtrafficcontrol/bluesky-gym"
export ATC_PYTHON="$PWD/.venv/bin/python"
export ATC_CANDIDATE_ARCHIVE="$HOME/decimal-heading-v1.zip"
export ATC_ENV=sa
sbatch --partition=YOUR_CPU_PARTITION --account=YOUR_ACCOUNT "$HOME/evaluate_candidate.slurm"
export ATC_ENV=ma
sbatch --partition=YOUR_CPU_PARTITION --account=YOUR_ACCOUNT "$HOME/evaluate_candidate.slurm"
squeue -u "$USER"
```

Python and archive paths must be absolute. Use the intended compatible Python 3.12
environment; the job does not install dependencies on the login node. Actual Linux
package versions are captured for each job. Different dependency builds or numeric
behavior can change trajectories, so a Windows reproduction is not assumed to
prove Linux reproduction.

Each job verifies the pinned archive hash before extraction and creates a new
runs/cluster-score-TRACK-JOBID directory. It extracts its own source/model copy,
verifies every listed file, and runs the two-scenario development check for its
track inside the allocation. That check enforces the compiled geography backend
and matching physical metrics. A mismatch stops before full scoring and must be
investigated rather than bypassed or reported as a successful reproduction.

The full scoring call then uses exactly 1000 scenarios, seed 42 on the first reset
only, and ten aircraft for MA. It uses the original harness through its two allowed
integration hooks. Completed output contains metrics.csv, independently recomputed
summary.json, packages.txt, the exact job script, archive hash, extracted source,
models and development-check evidence. slurm-score-JOBID.log records execution.
A second source verification follows scoring. The job refuses to reuse an existing
output directory. Its CSV finalization slots are not stable aircraft identities.

## Interpretation and return files

Treat these as reproductions of fixed candidates and retain every outcome. Seed 42
is not a model-selection set. If a local run has already used that stream, another
cluster run is a reproduction or revision comparison, not an independent test.
The corrected deployment has separate results from the original heading formatter.
No result is judge-verified unless the competition organizers confirm it.

Return the complete cluster-score-TRACK-JOBID folder and its Slurm log. A successful
submission to Slurm is not completion: inspect the terminal job state, final CSV,
summary, both source verifications and package/check evidence before using scores.

## Linux package availability check

The ten core package versions used locally have published wheels matching
CPython 3.12 on Linux x86-64 with glibc 2.35. The version constraints are in
jobs/constraints-evaluation.txt; this is not a complete dependency lock or a
claim that installation has succeeded. Preserve every job's actual packages.txt.

The [BlueSky 1.1.1 release](https://pypi.org/project/bluesky-simulator/1.1.1/#files)
provides a CPython 3.12 manylinux x86-64 wheel. Its downloaded SHA-256 was checked
against PyPI, its compiled _cgeo module is an x86-64 ELF shared library, and all
159 Python source files match the installed Windows simulator after newline
normalization. Linux native behavior has not been executed or compared. Evidence
is in runs/linux-package-check-v1/audit.json.

The [PyQt6 6.11.0 wheel](https://pypi.org/project/PyQt6/6.11.0/#files) requires
glibc 2.34 or newer on x86-64. Check the actual allocated compute node, not just
the login node, before relying on these wheel choices. For example:

```bash
uname -m
getconf GNU_LIBC_VERSION
python3 --version
```

The local WSL Ubuntu instance is x86-64 with glibc 2.35, but its default interpreter
is Python 3.10.12 and none of the inspected ML/simulator packages is installed.
It is not yet a ready Linux reproduction environment. No system packages, Python
environment or ML dependencies were installed during this inspection.

## Local native Linux preparation

After the package inspection, a separate local WSL reproduction environment was
started under runs/linux-runtime-v1. The Python 3.12.14 Linux standalone archive
was verified against its published SHA-256 and its interpreter executed successfully.
The isolated environment was created successfully at 14:15:27 UTC, including its
package installer. Its first dependency installation failed because the WSL network is unreachable.
A separate offline installation resolved successfully but was later stopped
because the host ran short of disk space and memory. Its incomplete virtual
environment was removed after the installer exited. No simulator evaluation or
university job is implied by these setup checks.

The published CPU-only PyTorch 2.14.0 wheel has no NVIDIA or Triton requirements
in its package metadata. Its binary HEAD request returned HTTP 403, so its size
was not established by that request; metadata retrieval succeeded. The exact
wheel URL and hash, Python archive identity and request results are retained in
runs/linux-package-check-v1/runtime-plan.json. See the
[PyTorch CPU package index](https://download.pytorch.org/whl/cpu/torch/) and the
[Python standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901).

The prepared local installer uses CPU PyTorch explicitly, retains the other
frozen package version constraints, requires four GB of free disk space before
installation, and keeps installation logs and the actual dependency report.
It does not change the Windows training environment or system Python. A separate
runner will check both tracks from fresh candidate extractions on their original
two-scenario development prefixes. All failures are retained, and a failed check
must be investigated. These scripts are local preparation artifacts; they have
not yet demonstrated a working Linux simulator environment.

The offline package set completed download and verification at 14:28:34 UTC.
It contains 43 wheels totaling 448,041,685 compressed bytes and 1,421,347,219
uncompressed bytes. All 42 downloaded wheels match their published SHA-256;
the remaining zmq metadata wheel was built locally from the unchanged published
966-byte source archive after inspection. It only declares a pyzmq dependency.
Its source and wheel hashes and build log are retained in zmq-build/.

The first binary-only download failed because zmq has no published wheel; that
attempt remains separate. A check of every other frozen dependency version found
compatible Linux wheels. The successful second set is wheelhouse-v2/, with
manifest SHA-256 2e61d74f7589b4df18440dfbe536bac9b4f514ccaa25bc32112b558242990e03.
Native Linux pip resolved the offline dependencies and started installation.
At 15:25 UTC the project installer was interrupted to protect existing scoring
jobs: free disk space had fallen below 600 MB. After the installer exited, only
its incomplete .venv/ and temporary files were removed; 3.79 GB was free afterward.
The complete wheelhouse, Python archive, installation report and every failed log
remain available. See runs/linux-runtime-v1/resource-recovery.json and
offline-install-state.json. There is no install-success.json. Dependency validation
and native simulator execution have not occurred. A future attempt must use an
adequately provisioned environment and a new attempt directory; do not treat the
old Python bootstrap record as proof that its removed virtual environment exists.
