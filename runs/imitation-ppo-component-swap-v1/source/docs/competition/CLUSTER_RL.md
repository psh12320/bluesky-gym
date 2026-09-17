# Current cluster experiment — 16 September 2026

Use [STATIC_FILTER_CLUSTER.md](STATIC_FILTER_CLUSTER.md) for the current upload, submission and result-collection commands. The ready bundle is runs/cluster-ppo-static-v1.zip (SHA-256 a0ad40c1e209cc7d33675e20250f02348a3f4d2e90b903ac69f0c1ba77dafb53). It reuses the installed virtual environment.

The previous three-seed baseline completed: jobs 851486/851487 trained about one million live transitions per seed, and jobs 851488/851489 evaluated the saved checkpoints. The supplied logs report 20%, 19% and 20% final clean completion, versus 19.5% initially and 95% for the cluster classical benchmark. Raw artifacts have been requested for independent auditing. The Slurm accounting query returned error 1054, so completion is established here from the saved experiment logs, not a successful accounting query.

The next cluster study has three paired training seeds with fixed route guidance and static-area filtering; traffic filtering is off. Real predictive inputs are compared with equal-size zero channels. All 149 local checks, isolated training/reload checks, legacy compatibility checks and the six job dry runs passed. No job ID has yet been received for this new study.

The local std-0.05 versus std-0.20 exploration pilot is separate, uses one worker, and does not modify the cluster bundle.

## Earlier setup and baseline instructions

The material below records the earlier setup and baseline protocol.

# Run the RL experiments on the university cluster

Bootstrap recovery job **851402 completed successfully** (62 seconds, exit 0:0),
reusing ~/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python. Its GPU training/reload
checks passed, and all 180 classical metrics across two scenarios matched Windows.
The user-supplied transcript is preserved under runs/cluster-receipt-851402/.
The full remote logs remain in their original cluster directories.

The next selected experiment is runs/cluster-ppo-baseline-v1.zip: three PPO seeds,
one million live transitions each, eight worlds, goal-relative actions, reward
scale 0.01, neutral mean, initial action standard deviation 0.05, no route guidance
and no conflict filter. Its fixed protocol is jobs/ppo_baseline_v1.json.
Use [the exact launch commands](../../runs/cluster-ppo-baseline-v1-preparation/LAUNCH.md).
The first training allocation checks the expanded RL test suite. Later training
and CPU evaluations depend on successful completed-budget training. All existing
source archives, environments, evaluations and logs are preserved. No reinstall
is needed.

The sections below retain the earlier setup and general experiment templates as
historical reference. The selected launch instructions above supersede their
raw-PPO commands. Keep passwords in the SSH/SCP prompt.

## Transfer from Windows PowerShell

The archive contains source, tests, job scripts and a hash manifest. It contains
no environment, credentials, existing evaluation files or historical checkpoints.
All local development remains under the user's airtrafficcontrol repository.

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\onpolicy-cluster-v1.zip" shri@xlogin.comp.nus.edu.sg:cs4246-rl/
```

## In the existing xlogin SSH session

Use a fresh extraction directory; do not overlay a source tree used by a running
job. The Python archive command works without an unzip installation.

```bash
cd ~/cs4246-rl
sha256sum onpolicy-cluster-v1.zip
mkdir onpolicy-v1
python3 -m zipfile -e onpolicy-cluster-v1.zip onpolicy-v1
cd onpolicy-v1
python3 -m atc_rl.cluster verify
sbatch jobs/bootstrap_onpolicy.slurm
```

Compare the printed SHA-256 with the local runs/onpolicy-cluster-v1.json record.
The setup job creates .venv-cluster inside this new source directory. It installs
pinned dependencies on a compute node, checks CUDA forward/backward execution,
runs the RL unit tests, trains a small eight-world PPO run, checks all 180 physical
metrics on two scenarios against the Windows reference, and reloads/evaluates the
checkpoint. The output is bootstrap-rl-JOBID.log; results live under
runs/cluster-check-JOBID/. Return that log if anything fails. The simulator requires
its compiled geography backend and OpenAP; a fallback is rejected.

```bash
squeue -u "$USER"
tail -n 60 bootstrap-rl-JOBID.log
sacct -j JOBID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

Replace JOBID with the number printed by sbatch. Account allusers and QoS normal
come from the supplied association, but only Slurm can confirm allocation access.
An allocation rejection should be preserved verbatim. If H200 availability is
poor, the same gpu job may request --gres=gpu:a100-40:1 or gpu:h100-96:1 instead;
record the actual hardware. Do not request H200 on gpu-long based on the displayed
resource list. Compute-node downloads and the user's storage quota are unverified.
The initial environment needs several GB; model and rollout-memory costs are much
smaller. Do not run simultaneous setup jobs against the same .venv-cluster.

## Canonical PPO baseline after setup succeeds

Current scheduling note: finish the bootstrap first. The raw local PPO million-step
run has completed with poor navigation. Do not launch the whole array/grid simply
because these templates exist. Select the next training recipe from the paired
RL pilot results, then record its complete settings and run the three seeds.
The commands below retain the original raw-baseline configuration for comparison.

```bash
sbatch --array=0-2%1 jobs/train_onpolicy.slurm
```

This runs three seeds (50100, 50200, 50300), initially one job at a time, one GPU,
16 CPUs, 64 GB RAM and eight independent simulator worlds per job. Each budget is
1,000,000 live aircraft transitions; padded slots are reported separately. The
first baseline has guidance off and filtering off. The soft stop is 9,900 seconds
from training start, with a final checkpoint after the current rollout. A job that
stops before its experience budget is a partial run, not a completed million-step
run. Periodic checkpoints limit lost progress if Slurm kills a job unexpectedly;
exact continuation of simulator/RNG state is not implemented.

The package is executed from its source root through PYTHONPATH. No project wheel
is installed: the existing upstream wheel configuration excludes atc_rl, and the
archive intentionally contains no Git metadata. This avoids altering packaging
files used by the preserved evaluations.

## Ablations and centralized-critic comparison

After the first PPO curve and throughput are inspected, submit each permitted
configuration with the same three seeds and live-transition budget:

```bash
sbatch --array=0-2%1 --export=ALL,ATC_GUIDANCE=1,ATC_FILTER=0 jobs/train_onpolicy.slurm
sbatch --array=0-2%1 --export=ALL,ATC_GUIDANCE=0,ATC_FILTER=1 jobs/train_onpolicy.slurm
sbatch --array=0-2%1 --export=ALL,ATC_GUIDANCE=1,ATC_FILTER=1 jobs/train_onpolicy.slurm
```

MAPPO uses ATC_ALGORITHM=mappo with the same four support configurations, after
the PPO baseline is established. Its actor remains local; only the critic receives
joint observations, aircraft activity and identity. Do not assume it will win.

Evaluate initial and saved policies with python -m atc_rl.evaluate on stream 20260
inside a CPU allocation. The evaluator takes --model RUN/initial-model.zip or
--model RUN/model.zip, --episodes 20, --seed 20260 and a fresh --out directory. It
reads guidance/filter settings from the training configuration, zeros critic
context for per-aircraft actor inference, verifies the checkpoint hash and retains
all nine aircraft metrics and scenario fingerprints. Use --classical for the fixed
benchmark. Preserve unsuccessful runs. Unseen streams 20301 and 20302 remain
reserved until candidate selection, as described in RL_PLAN.md.

## References

The GPU wheel is pinned to torch 2.14.0 with CUDA 12.6, present in the
[official PyTorch wheel index](https://download.pytorch.org/whl/cu126/torch/).
A working driver is checked inside the allocation rather than assumed from the
GPU name. Allocation syntax follows the
[Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html).

## Version 2 for subsequent experiments

The version-1 archive and any setup/job already using it remain unchanged and
valid for the initial cluster check. A separate onpolicy-cluster-v2.zip contains
the tested reward-scale option, matched initial actors for MAPPO, checkpoint
audits and a CPU-only evaluation job. Use a fresh onpolicy-v2 directory for this
archive. Do not overlay onpolicy-v1 or another running job's source.

After extracting and verifying version 2, an environment successfully prepared
by version 1 can be reused without reinstalling dependencies:

```bash
export ATC_PYTHON="$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python"
```

Otherwise submit version 2's bootstrap job. The same account, GPU and CPU
allocation rules apply. Reward-scale experiments set ATC_REWARD_SCALE; it defaults
to 1.0, while ATC_PROGRESS_SCALE defaults to 0.0. MAPPO can set ATC_ACTOR_REFERENCE
to that seed's zero-experience PPO initial-model.zip, with matching worker count.
A learned checkpoint is rejected as an initial actor reference. Canonical
three-seed runs still require separate initial references for each seed.

CPU evaluation example, after replacing the checkpoint path with an actual run:

```bash
export ATC_MODEL="runs/ACTUAL-TRAINING-RUN/model.zip"
sbatch jobs/evaluate_onpolicy.slurm
```

This requests two CPUs and 8 GB RAM on normal, with no GPU, and evaluates twenty
development worlds. Submit the corresponding initial-model.zip as a separate
job; compare both against the preserved classical benchmark. Each training job
now audits checkpoint integrity, experience counts and actor/critic dependencies
after training. Physical performance is still assessed only by evaluation.


## Goal-offset source for the next selected RL experiment

A separate immutable archive, runs/onpolicy-goal-offset-source-v1.zip, contains
the tested goal-offset action mapping and explicit action-reference metadata.
Its SHA-256 is abd2a67c99a9adfc5bd5540c33bb5f7659a04aa1ce108e5120b7c35b58b7e695.
It is not needed to finish an already-started version-1 bootstrap. Preserve that
job and source directory. After setup succeeds, transfer this archive into
~/cs4246-rl and extract into a fresh onpolicy-goal-offset-v1 directory. Verify
its manifest and reuse the successfully tested ATC_PYTHON environment.

For a selected goal-offset run, set ATC_ACTION_REFERENCE=goal_offset,
ATC_INITIAL_ACTION_STD=0.05 and ATC_NEUTRAL_ACTION_MEAN=1. Reward scale remains
1.0 and progress shaping remains 0 unless the registered experiment explicitly
changes them. Guidance and filtering are separately controlled. These are the
current pilot settings, not an assertion that the pilot has passed.

Evaluate checkpoints from this source using its evaluator: the original version-1
evaluator predates goal_offset and would interpret its outputs as direct turns.
The current comparison tool rejects that mismatch. Initial and trained policies
must use the same action reference and evaluator source. Record actual hardware,
live-transition throughput, padded slots, CPU/RAM usage and all training seeds.


## Reproducible support/seed matrices

MATRIX_RL.md describes a twelve-row PPO matrix (four guidance/filter combinations,
three seeds) and its matched MAPPO counterpart. This is staged infrastructure;
no matrix job has been submitted. The matrix rejects changed source, duplicate
or mismatched support/seed rows, incomplete PPO controls and trained initial-actor
references. Each run retains its exact command, plan hash and completed-budget
audit. Existing single-run jobs and frozen archives remain unchanged.
