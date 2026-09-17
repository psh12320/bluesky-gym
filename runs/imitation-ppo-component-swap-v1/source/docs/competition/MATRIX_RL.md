# Three-seed PPO/MAPPO support matrix

Status: planner and validation controls tested locally; no matrix training or
Slurm submission has been performed. Finish cluster bootstrap and choose the
training recipe from the paired pilot results before spending the full grid.

Each matrix has twelve rows: guidance/filter off/off, off/on, on/off, on/on,
each with seeds 50100, 50200 and 50300. The actor and learning settings stay fixed.
Eight simulator workers use offsets 0, 10, ..., 70; their streams do not overlap
across these seeds. The planner rejects more than ten workers for this seed spacing.
All three seeds must be retained, including failed or weak runs.

Create plans inside the frozen source directory that will execute them. Source
hashes cover the simulator, training code, job scripts, dependency requirements
and project configuration. A source change invalidates the plan. Each row has a
unique output directory and an exclusive launch receipt; retries do not overwrite
partially completed work. A new attempt needs a fresh plan directory.

The following example encodes the currently investigated goal-offset, reward-scale
0.01 configuration. It is not a declaration that this recipe has passed selection:

```bash
python3 -m atc_rl.matrix create --plan runs/recipe-comparison/ppo.json \
  --action-reference goal_offset --initial-action-std 0.05 \
  --neutral-action-mean --reward-scale 0.01
python3 -m atc_rl.matrix run --plan runs/recipe-comparison/ppo.json --index 0 --dry-run
```

Dry-run prints the exact command without training. After recipe selection and
bootstrap, use the tested environment and submit from that frozen source root:

```bash
export ATC_PYTHON="$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python"
export ATC_PLAN="runs/recipe-comparison/ppo.json"
sbatch jobs/train_onpolicy_matrix.slurm
```

The array defaults to one row at a time, one H200, sixteen CPUs, 64 GB RAM and a
three-hour allocation. Each row requests one million live aircraft transitions
and saves checkpoints near each 100k. A soft stop after 9900 seconds retains the
last completed rollout. Partial runs are audited and preserved, but the matrix
job reports failure rather than describing a partial budget as complete.

After the PPO results establish the baseline, generate its matched MAPPO matrix:

```bash
python3 -m atc_rl.matrix create --plan runs/recipe-comparison/mappo.json \
  --ppo-plan runs/recipe-comparison/ppo.json
export ATC_PLAN="runs/recipe-comparison/mappo.json"
sbatch jobs/train_onpolicy_matrix.slurm
```

MAPPO inherits the complete PPO recipe. Each row references that same seed/support
row's initial-model.zip. Before launching, it requires complete PPO training,
matching configuration, a hash-verified zero-experience initial checkpoint and the
full checkpoint audit. The trainer then verifies and copies only that initial
actor; its centralized critic is initialized separately. A learned PPO actor is
not silently used as a warm start. No general advantage for MAPPO is assumed.

## Evaluate every registered checkpoint

After a training matrix finishes, its CPU evaluation array runs four points for
all twelve rows: initial, the first checkpoint at/above 100k, the first at/above
300k, and the completed final checkpoint. The forty-ninth task evaluates the fixed
classical controller. No performance-based checkpoint selection occurs. Actual
live counts are retained; a missing 100k checkpoint cannot be replaced by a
million-step checkpoint and still be labelled 100k.

```bash
export ATC_PLAN="runs/recipe-comparison/ppo.json"
sbatch jobs/evaluate_onpolicy_matrix.slurm
```

Submit after the training jobs succeed, or use Slurm's afterok dependency on the
actual training array job ID. Each task requests two CPUs, 8 GB RAM, no GPU and
three hours on normal, with at most four evaluation tasks running concurrently.
Each policy uses twenty paired development worlds from stream 20260. Matrix
completion, recipe, source provenance, checkpoint identity and native scoring
are checked before the result can enter analysis. Existing complete evaluations
may be reused after verification; partial outputs are preserved and rejected.
The classical result is shared when PPO and MAPPO plans live in the same folder.
Do not submit both classical tasks simultaneously.

```bash
python3 -m atc_rl.matrix_evaluate run --plan "$ATC_PLAN" --index 0 --dry-run
python3 -m atc_rl.matrix_evaluate analyze --plan "$ATC_PLAN"   --out runs/recipe-comparison/ppo-analysis
```

Analysis requires all twelve runs and all four checkpoints on identical scenario
fingerprints. It saves per-run learning curves, paired-world comparisons, a CSV
with every training seed, and a summary of seed means and sample standard
deviations. Per-world bootstrap intervals describe scenario variation, not the
uncertainty of three independent training runs. Negative seed results remain in
the aggregate. Missing, duplicate, extra or non-finite seed records are rejected.

Guidance and filtering are measured separately at initialization and after
training. The summary also reports their difference in learning changes and the
guidance/filter interaction for each seed. A large initial support benefit must
not be described as a learned gain. Comparisons with classical retain all nine
physical metrics and clean completion; no unofficial scalar score is invented.

Repeat with the matched MAPPO plan after its training finishes. Retain SAC as a
separate historical comparison until an equally controlled evaluation is ready.
Reserve streams 20301/20302 until candidate selection. Development matrix results
are not unseen-scenario or official competition evidence.
