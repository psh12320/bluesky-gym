#!/bin/bash
#SBATCH --job-name=atc-rl-eval
#SBATCH --partition=normal
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=evaluate-rl-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted source root}"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
PYTHON_BIN="${ATC_PYTHON:-$SLURM_SUBMIT_DIR/.venv-cluster/bin/python}"
EVAL_SEED="${ATC_EVAL_SEED:-20260}"
OUT="${ATC_EVAL_ROOT:-runs}/evaluation-seed${EVAL_SEED}-${SLURM_JOB_ID}"
ARGS=(--episodes "${ATC_EPISODES:-20}" --seed "$EVAL_SEED" --out "$OUT")
if [[ "${ATC_CLASSICAL:-0}" == 1 ]]; then
    if [[ -n "${ATC_MODEL:-}" ]]; then
        echo 'Choose a classical evaluation or a checkpoint, not both.' >&2
        exit 2
    fi
    ARGS+=(--classical)
else
    ARGS+=(--model "${ATC_MODEL:?Set ATC_MODEL to the saved checkpoint path}")
fi
if [[ "${ATC_HELDOUT:-0}" == 1 ]]; then ARGS+=(--heldout); fi
"$PYTHON_BIN" -m atc_rl.cluster verify
"$PYTHON_BIN" -m atc_rl.evaluate "${ARGS[@]}"
"$PYTHON_BIN" -m atc_rl.cluster verify
