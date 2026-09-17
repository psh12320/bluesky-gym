#!/bin/bash
#SBATCH --job-name=atc-ppo
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200-141:1
#SBATCH --time=03:00:00
#SBATCH --output=onpolicy-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
PYTHON_BIN="${ATC_PYTHON:-$SLURM_SUBMIT_DIR/.venv-cluster/bin/python}"
ALGORITHM="${ATC_ALGORITHM:-ppo}"
SEED="${ATC_SEED:-$((50100 + 100 * ${SLURM_ARRAY_TASK_ID:-0}))}"
WORKERS="${ATC_WORKERS:-8}"
GUIDANCE="${ATC_GUIDANCE:-0}"
FILTER="${ATC_FILTER:-0}"
RUN_DIR="${ATC_RUN_ROOT:-runs}/onpolicy-${ALGORITHM}-g${GUIDANCE}-f${FILTER}-seed${SEED}-${SLURM_JOB_ID}"
if (( WORKERS + 2 > ${SLURM_CPUS_PER_TASK:-0} )); then
    echo 'Request enough CPUs for simulator workers plus the learner and coordination.' >&2
    exit 2
fi
ARGS=(--algorithm "${ALGORITHM}" --workers "${WORKERS}" --live-steps "${ATC_LIVE_STEPS:-1000000}"
      --rollout-steps 256 --batch-size 1024 --epochs 10 --seed "${SEED}" --device cuda
      --checkpoint-live-steps 100000 --max-wall-seconds "${ATC_MAX_WALL_SECONDS:-9900}" --run-dir "${RUN_DIR}")
ARGS+=(--reward-scale "${ATC_REWARD_SCALE:-1.0}" --progress-scale "${ATC_PROGRESS_SCALE:-0.0}")
ARGS+=(--action-reference "${ATC_ACTION_REFERENCE:-direct}")
ARGS+=(--initial-action-std "${ATC_INITIAL_ACTION_STD:-0.6065306597126334}")
if [[ "${ATC_NEUTRAL_ACTION_MEAN:-0}" == 1 ]]; then ARGS+=(--neutral-action-mean); fi
if [[ -n "${ATC_ACTOR_REFERENCE:-}" ]]; then ARGS+=(--actor-reference "$ATC_ACTOR_REFERENCE"); fi
if [[ "${GUIDANCE}" == 1 ]]; then ARGS+=(--guidance); fi
if [[ "${FILTER}" == 1 ]]; then ARGS+=(--filter); fi
if [[ "${ATC_STATIC_FILTER:-0}" == 1 ]]; then ARGS+=(--static-filter); fi
if [[ "${ATC_CONFLICT_FEATURES:-0}" == 1 ]]; then ARGS+=(--conflict-features); fi
if [[ "${ATC_MASK_CONFLICT_FEATURES:-0}" == 1 ]]; then ARGS+=(--mask-conflict-features); fi
"${PYTHON_BIN}" -m atc_rl.cluster verify
"${PYTHON_BIN}" -m atc_rl.train "${ARGS[@]}"
"${PYTHON_BIN}" -m atc_rl.audit --run "$RUN_DIR" --out "$RUN_DIR/checkpoint-audit.json"
