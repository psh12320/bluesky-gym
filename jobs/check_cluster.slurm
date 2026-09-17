#!/bin/bash
#SBATCH --job-name=atc-check
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-check-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
PYTHON_BIN="${ATC_PYTHON:-python}"
"${PYTHON_BIN}" -m atc.transfer verify
"${PYTHON_BIN}" -c 'import platform, torch; print(platform.platform()); print("Torch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable in this allocation"; print(torch.cuda.get_device_name(0))'
for ENV_KIND in sa ma; do
    RUN_DIR="runs/cluster-check-${ENV_KIND}-${SLURM_JOB_ID}"
    "${PYTHON_BIN}" -m atc.train --env "${ENV_KIND}" --algorithm sac --recipe public_weights \
        --workers 2 --steps 400 --learning-starts 100 --gradient-steps 4 --batch-size 64 \
        --buffer-size 2000 --checkpoint-every 400 --seed 1500 --device cuda \
        --save-replay --max-wall-seconds 180 --run-dir "${RUN_DIR}"
    "${PYTHON_BIN}" -m atc.evaluate --env "${ENV_KIND}" --algorithm sac --recipe public_weights \
        --model "${RUN_DIR}/model.zip" --episodes 2 --seed 2026 --out "${RUN_DIR}/smoke-evaluation"
done
echo "Both GPU training and original-simulator evaluation smoke checks completed."
