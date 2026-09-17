#!/bin/bash
#SBATCH --job-name=atc-ppo-v1
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200-141:1
#SBATCH --time=03:00:00
#SBATCH --output=ppo-baseline-train-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted source root}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
test -x "${ATC_PYTHON:?Set ATC_PYTHON to the existing cluster interpreter}"
if (( ${SLURM_CPUS_PER_TASK:-0} < 10 )); then
    echo 'Eight simulator worlds require CPUs for workers, learner and coordination.' >&2
    exit 2
fi
nvidia-smi
"$ATC_PYTHON" -c 'import torch; assert torch.cuda.is_available(); print("Torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.get_device_name(0))'
"$ATC_PYTHON" jobs/ppo_baseline_v1.py train --index "${SLURM_ARRAY_TASK_ID:?}"
