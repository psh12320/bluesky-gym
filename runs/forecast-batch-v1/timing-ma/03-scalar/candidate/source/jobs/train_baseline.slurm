#!/bin/bash
#SBATCH --job-name=atc-train
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
PYTHON_BIN="${ATC_PYTHON:-python}"
SEED="${ATC_SEED:-$((1000 + 100 * ${SLURM_ARRAY_TASK_ID:-0}))}"
ALGORITHM="${ATC_ALGORITHM:-sac}"
RECIPE="${ATC_RECIPE:-baseline}"
ENV_KIND="${ATC_ENV:-ma}"
WORKERS="${ATC_WORKERS:-8}"
if [[ "${ENV_KIND}" == ma ]]; then
    DEFAULT_GRADIENT_STEPS=$((4 * WORKERS))
else
    DEFAULT_GRADIENT_STEPS=${WORKERS}
fi
RUN_DIR="runs/${ALGORITHM}-${ENV_KIND}-${RECIPE}-seed${SEED}-${SLURM_JOB_ID}"
ARGS=(--env "${ENV_KIND}" --algorithm "${ALGORITHM}" --recipe "${RECIPE}"
      --workers "${WORKERS}" --steps "${ATC_STEPS:-3000000}"
      --seed "${SEED}" --device "${ATC_DEVICE:-cuda}" --gradient-steps "${ATC_GRADIENT_STEPS:-${DEFAULT_GRADIENT_STEPS}}"
      --max-wall-seconds "${ATC_MAX_WALL_SECONDS:-9000}" --run-dir "${RUN_DIR}"
      --buffer-size "${ATC_BUFFER_SIZE:-1000000}" --checkpoint-every "${ATC_CHECKPOINT_EVERY:-500000}")
# Omitted learner options preserve the trainer defaults or the loaded checkpoint.
for NAME in NET_ARCH LEARNING_RATE TAU BATCH_SIZE LEARNING_STARTS CRITIC_WARMUP_UPDATES; do
    VARIABLE="ATC_${NAME}"
    if [[ -n "${!VARIABLE:-}" ]]; then
        FLAG="${NAME,,}"
        ARGS+=("--${FLAG//_/-}" "${!VARIABLE}")
    fi
done
if [[ "${ATC_GUARD_STATIC:-0}" == 1 ]]; then
    ARGS+=(--guard-static)
fi
if [[ "${ATC_GUARD_TRAFFIC:-0}" == 1 ]]; then
    ARGS+=(--guard-traffic)
fi
if [[ "${ALGORITHM}" == sac ]]; then
    ARGS+=(--save-replay)
fi
if [[ -n "${ATC_RESUME:-}" ]]; then
    ARGS+=(--resume "${ATC_RESUME}")
fi
if [[ -n "${ATC_RESUME_REPLAY:-}" ]]; then
    ARGS+=(--resume-replay "${ATC_RESUME_REPLAY}")
fi
"${PYTHON_BIN}" -m atc.train "${ARGS[@]}"
echo "Saved training result in ${RUN_DIR}; evaluate checkpoints in a separate job."
