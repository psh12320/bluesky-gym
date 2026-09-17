# University cluster handoff

For the fixed model evaluation, use EVALUATE_CANDIDATE.md and the separately
packaged decimal-heading-v1 release. The instructions below cover the historical
source-v17 training handoff and initial cluster smoke tests.

Source and local experiments are prepared. No cluster authentication or Slurm
submission has been performed by this checkout. Enter passwords only into your own
SSH/SCP prompts. The first allocation checks installation and GPU execution;
performance conclusions require separate measured runs.

## Connection and allocation information

Log in from Windows PowerShell:

```powershell
ssh -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg
```

Inside that authenticated session:

```bash
sinfo -o "%P %G %c %m %l"
sacctmgr -n show assoc user="$USER" format=Account,Partition,QOS
python3 --version
```

Select partition, account, QoS, GPU resource syntax and Python modules from the
actual cluster. The job templates request one GPU, but resource flags may need
cluster-specific overrides. Training belongs in a Slurm allocation, not on xlogin.
No credentials belong in the repository or in messages.

## Transfer and verify the source

From Windows PowerShell:

```powershell
Set-Location 'C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym'
scp -J shri@sjump.comp.nus.edu.sg .\runs\cluster\source-v17.zip shri@xlogin.comp.nus.edu.sg:airtrafficcontrol-source.zip
```

This bundle includes uncommitted experiment source, documentation and dependency
evidence. It excludes environments, credentials, checkpoints and replay buffers.
Earlier source bundles remain immutable. After later edits, create a newly named
bundle with `python -m atc.transfer pack --out runs/cluster/UNUSED_NAME.zip`.

For a new destination on the cluster:

```bash
mkdir -p "$HOME/airtrafficcontrol"
git clone --branch AI4REAL-NET-Competition https://github.com/psh12320/bluesky-gym.git "$HOME/airtrafficcontrol/bluesky-gym"
cd "$HOME/airtrafficcontrol/bluesky-gym"
git checkout --detach 00930013219af4c17e509c3efc84a8becd0f3546
python3 -m zipfile -e "$HOME/airtrafficcontrol-source.zip" .
python3 -m atc.transfer verify
```

These commands assume the checkout does not already exist. For subsequent work,
use a separate fresh checkout or review existing changes before overlaying files.
Verification checks all source hashes and the base commit. Keep source fixed
within a running experiment; its manifest remains in runs/cluster-transfer.

## Install and run the allocation check

Use Python 3.12, loading the cluster's module if required. If its executable is
python3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e . "bluesky-simulator==1.1.1"
export ATC_PYTHON="$PWD/.venv/bin/python"
```

Use a PyTorch build compatible with the allocated GPU and driver. The bundled
packages.txt describes the local Windows CPU environment; do not install its
Windows paths or CPU-only PyTorch wheel on the cluster. Every experiment records
its actual installed dependencies.

After selecting real partition/account values:

```bash
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT jobs/check_cluster.slurm
squeue -u "$USER"
```

Omit account if the cluster supplies a default, and add required QoS/GPU overrides.
The check uses two simulator workers per track, verifies CUDA, trains both MA and
SA briefly and evaluates them on two development scenarios. Inspect
slurm-check-JOBID.log and runs/cluster-check-{sa,ma}-JOBID/training_summary.json.
A smoke result checks execution, not policy quality.

## Current learning experiment

Local evidence favors retaining route guidance and the joint traffic/static filter:
they greatly reduce safety failures compared with the historical unfiltered learned
baseline. The current fast-reference residual controller initializes its deterministic
actor at the stronger classical route follower. Learning has not yet established a
clear overall advantage over that reference. Five-second MA control has completed its development comparison and is selected
for the frozen MA candidate; SA uses ten-second control. The initial cluster
training trial below still uses the original ten-second training recipe.

Start with one job to measure throughput and verify the residual setup on the GPU.
The following reproduces the requested local budget/configuration with a fresh
seed; GPU numerics and dependencies may change trajectories, so it is not an exact
replay of a CPU run. Replace the resource placeholders before submission.

```bash
unset ATC_RESUME ATC_RESUME_REPLAY ATC_NET_ARCH ATC_LEARNING_RATE ATC_TAU
unset ATC_CRITIC_WARMUP_UPDATES
export ATC_PYTHON="$PWD/.venv/bin/python"
export ATC_ALGORITHM=sac
export ATC_ENV=ma
export ATC_RECIPE=public_route_choice_fast_residual
export ATC_GUARD_TRAFFIC=1
export ATC_DEVICE=cuda
export ATC_WORKERS=1
export ATC_STEPS=25000
export ATC_LEARNING_STARTS=5000
export ATC_GRADIENT_STEPS=4
export ATC_BATCH_SIZE=256
export ATC_BUFFER_SIZE=100000
export ATC_CHECKPOINT_EVERY=25000
export ATC_MAX_WALL_SECONDS=1800
export ATC_SEED=4100
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT jobs/train_baseline.slurm
```

This uses the trainer's 64x64 network, learning rate 0.0003, tau 0.005, and no initial
actor hold. The saved run includes initial weights, periodic checkpoints, final
weights and replay. Periodic checkpoints precede the subsequent gradient updates;
at the local 25k budget the callback has 7,996 updates and the final model 8,000.
Always identify the actual evaluated file and hash.

A small network can be limited by CPU simulation and GPU launch overhead. Benchmark
before scaling. For more experience, use independent simulator processes with enough
CPU cores and one GPU; multi-GPU learning is not implemented. Keep the update ratio
explicit: local MA uses four gradient updates per ten counted aircraft transitions,
so sixteen worlds would use 64 updates per batch. Counted MA transitions include
inactive-aircraft padding; training_summary.json also gives live and skipped counts.
Changing worker count changes scenario collection, update timing and batch boundaries.
It is a new experiment, not a numerically matched continuation.

For fresh replicas, unset ATC_SEED and use an array such as `--array=0-2`; the template
then uses seeds 1000, 1100 and 1200. Preserve 2026, 2027 and 42 for evaluation. Do not
launch an array with a fixed ATC_SEED, since that would duplicate the same seed.
Choose a larger budget only after the pilot's throughput and evaluation are known.
The job stops before its time limit and may complete fewer transitions than requested.

## Evaluate and return evidence

Within a CPU allocation, replace RUN_DIRECTORY and use the same checkpoint-selection
rule on every training seed:

```bash
.venv/bin/python -m atc.evaluate --env ma --algorithm sac \
  --recipe public_route_choice_fast_residual \
  --model RUN_DIRECTORY/checkpoints/model_25000_steps.zip \
  --episodes 20 --seed 2026 --out RUN_DIRECTORY/validation-20
```

Compare on the same scenario stream against the trained reference and the matching
classical controller. Both learned and classical evaluations must use the same
route/filter/action configuration. Expand promising candidates to 200 scenarios;
retain all objective metrics, event counts and clean completion. A zero-shot SA
transfer still simulates all ten scripted intruders and reports zero SA training.
See atc/README.md for transfer and comparison commands.

Return configuration, training_summary.json, progress.csv, evaluation CSV/JSON,
packages.txt, provenance.json and the exact selected checkpoint. Keep replay on the
cluster for continuation. The source bundle excludes local trained weights; those
must be transferred separately if needed. Do not resume with mismatched replay.

The primary candidates were frozen before their seed-2027 evaluations. Those
evaluations are complete. Original-format SA seed-42 scoring has completed;
corrected SA/MA seed-42 scoring remains active. All three training seeds for both tracks have completed replication; the combined table retains every declared model. These streams must not be used for further candidate
selection.
The competition harness uses 1,000 scenarios; development tests are not competition
scores. A finalist still needs the original harness run, report and failure videos.
Current local evidence and rejected experiments are in docs/competition/PILOT_RESULTS.md;
public-source attribution is in docs/competition/PUBLIC_WORK.md.

## Optional experiments

- `ATC_CRITIC_WARMUP_UPDATES` holds the initial actor for a count of gradient updates
  while critics, targets and entropy temperature learn. It differs from replay-only
  `ATC_LEARNING_STARTS`. The local 2,000-update hold gives mixed results and is not
  the default. Resume inherits the saved schedule; omit an explicit override.
- `public_route_choice_fast_residual_interval5` changes decision timing to five
  seconds. Its saved metadata and replay must match. Equal gamma and transition
  counts do not imply equal discounting or experience per simulated second.
- The attributed public direct-control learner remains a historical comparison:
  recipe public_weights, 256x256 network, learning rate 0.000442773440394527,
  tau 0.008242901455713948, batch 256, learning-starts 50000 and buffer 1000000.
  It is not a reproduction of the public author's Rust collector or a demonstrated
  stronger local candidate.

- `public_route_choice_fast_residual_reach250` changes only the arrival reward to
  250. It uses fresh training/replay, with the same ten-second control and all
  safety settings. Fixed-policy replays preserve objective metrics and trajectory
  bytes exactly. The matched 25k pilot loses two MA arrivals and is not promoted.
  SA transfer has since completed 200 development and 200 held-out cases with
  all arrivals and is selected for SA. An independent training seed reaches 98%
  on the same held-out stream, so consistent superiority remains unproven. The
  historical initial cluster trial above is not changed to this SA configuration.
  All 89 regression tests passed after the recipe addition.
- `atc.record` creates offscreen development previews and checks all nine saved
  metrics. The first five scenarios on each track have exact replay checks. The
  source bundle includes the recorder; existing local GIFs/models remain separate.
