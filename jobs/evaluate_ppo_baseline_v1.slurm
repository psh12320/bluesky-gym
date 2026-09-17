#!/bin/bash
#SBATCH --job-name=atc-ppo-eval
#SBATCH --partition=normal
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=ppo-baseline-eval-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted source root}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
test -x "${ATC_PYTHON:?Set ATC_PYTHON to the existing cluster interpreter}"
"$ATC_PYTHON" jobs/ppo_baseline_v1.py evaluate --index "${SLURM_ARRAY_TASK_ID:?}"
