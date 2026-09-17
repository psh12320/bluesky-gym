#!/bin/bash
#SBATCH --job-name=atc-rl-check
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200-141:1
#SBATCH --time=03:00:00
#SBATCH --output=bootstrap-rl-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted source root}"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
python3 -m atc_rl.cluster verify
nvidia-smi
python3 --version
df -h .
if [[ -n "${ATC_PYTHON:-}" ]]; then
    PYTHON_BIN="$ATC_PYTHON"
    if [[ "$PYTHON_BIN" != /* || ! -x "$PYTHON_BIN" ]]; then
        echo 'ATC_PYTHON must name an existing executable using an absolute path.' >&2
        exit 2
    fi
    printf 'Reusing environment without installing packages: %s\n' "$PYTHON_BIN"
else
    if [[ -e .venv-cluster ]]; then
        echo 'Existing environment preserved. Set ATC_PYTHON to its bin/python to reuse it.' >&2
        exit 2
    fi
    python3 -m venv .venv-cluster
    PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv-cluster/bin/python"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
    "$PYTHON_BIN" -m pip install --prefer-binary -r jobs/requirements-onpolicy.txt
fi
"$PYTHON_BIN" -m pip check
CHECK_DIR="runs/cluster-check-${SLURM_JOB_ID}"
mkdir -p runs
mkdir "$CHECK_DIR"
"$PYTHON_BIN" -m pip freeze > "$CHECK_DIR/packages.txt"
"$PYTHON_BIN" -c 'import torch; print("Torch",torch.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(512,512,device="cuda",requires_grad=True); x.square().mean().backward(); torch.cuda.synchronize(); assert torch.isfinite(x.grad).all()'
"$PYTHON_BIN" -m pytest tests/rl -q -p no:cacheprovider --basetemp="$CHECK_DIR/tests" --junitxml="$CHECK_DIR/tests.xml"
"$PYTHON_BIN" -m atc_rl.train --algorithm ppo --workers 8 --live-steps 12800     --rollout-steps 64 --batch-size 1024 --epochs 2 --seed 49000 --device cuda     --checkpoint-live-steps 12800 --max-wall-seconds 600 --run-dir "$CHECK_DIR/training"
"$PYTHON_BIN" -m atc_rl.evaluate --classical --episodes 2 --seed 20260 --out "$CHECK_DIR/classical"
"$PYTHON_BIN" -m atc_rl.check_cluster_result "$CHECK_DIR/classical"
"$PYTHON_BIN" -m atc_rl.evaluate --model "$CHECK_DIR/training/model.zip" --episodes 2 --seed 20260 --out "$CHECK_DIR/reload"
"$PYTHON_BIN" -m atc_rl.cluster verify
printf 'GPU checks passed. Results: %s\n' "$CHECK_DIR"
