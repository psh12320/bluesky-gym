#!/bin/bash
#SBATCH --job-name=atc-rl-matrix
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200-141:1
#SBATCH --time=03:00:00
#SBATCH --array=0-11%1
#SBATCH --output=matrix-rl-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen source root}"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
PYTHON_BIN="${ATC_PYTHON:-$SLURM_SUBMIT_DIR/.venv-cluster/bin/python}"
"$PYTHON_BIN" -m atc_rl.cluster verify
"$PYTHON_BIN" -m atc_rl.matrix run --plan "${ATC_PLAN:?Set the selected matrix JSON path}" \
    --index "${SLURM_ARRAY_TASK_ID:?Use a Slurm array}" --device cuda \
    --max-wall-seconds "${ATC_MAX_WALL_SECONDS:-9900}"
