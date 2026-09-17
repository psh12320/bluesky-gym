#!/bin/bash
#SBATCH --job-name=atc-sac-check
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200-141:1
#SBATCH --time=00:30:00
#SBATCH --output=sac-check-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted bundle root}"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
PYTHON_BIN="${ATC_PYTHON:-$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python}"
RUN_DIR="runs/cluster-sac-check-v1/job-${SLURM_JOB_ID:?Run this through Slurm}"
if (( ${SLURM_CPUS_PER_TASK:-0} < 10 )); then
    echo 'Eight simulator workers require CPUs in addition to learner capacity.' >&2
    exit 2
fi
test -x "$PYTHON_BIN"
mkdir -p runs/cluster-sac-check-v1
mkdir "$RUN_DIR"
"$PYTHON_BIN" -m atc_rl.cluster verify
"$PYTHON_BIN" -m pytest tests/rl/test_sac_support.py -q -p no:cacheprovider --basetemp="$RUN_DIR/tests"
"$PYTHON_BIN" -u -m atc_rl.train_sac --run-dir "$RUN_DIR/train" \
    --workers 8 --device cuda --live-steps 20000 --seed 51900 \
    --collection-steps 128 --learning-starts 5000 --gradient-steps 8 \
    --buffer-size 200000 --batch-size 256 --checkpoint-live-steps 10000 \
    --guidance --static-filter --conflict-features --max-wall-seconds 1200
"$PYTHON_BIN" -m atc_rl.audit_sac --run "$RUN_DIR/train" --out "$RUN_DIR/audit.json"
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
from pathlib import Path
import json,sys,numpy as np,torch
from atc_rl.sac_support import load_sac_for_inference
root=Path(sys.argv[1])
summary=json.loads((root/'train/training_summary.json').read_text())
config=json.loads((root/'train/config.json').read_text())
assert summary['status']=='complete' and summary['live_transitions']>=20000
assert summary['aircraft_completed']>0 and summary['padded_transitions']>0
assert config['device']=='cuda' and torch.cuda.is_available()
model=load_sac_for_inference(root/'train/model.zip',config,device='cuda')
assert model.device.type=='cuda' and next(model.actor.parameters()).device.type=='cuda'
observation=np.zeros(model.observation_space.shape,dtype=np.float32)
observation[-1]=1.
action=model.predict(observation,deterministic=True)[0]
assert action.shape==(2,) and np.isfinite(action).all() and (np.abs(action)<=1).all()
result={'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':str(model.device),
        'gpu':torch.cuda.get_device_name(0),'loaded_actor_device':str(next(model.actor.parameters()).device),
        'reload_action_finite_and_bounded':True,'live_transitions':summary['live_transitions'],
        'padded_transitions':summary['padded_transitions'],'aircraft_completed':summary['aircraft_completed'],
        'gradient_rounds':summary['gradient_rounds'],'performance_evidence':False}
(root/'cuda-reload.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
PY
"$PYTHON_BIN" -u -m atc_rl.evaluate --model "$RUN_DIR/train/model.zip" \
    --episodes 2 --seed 20260 --out "$RUN_DIR/vector"
"$PYTHON_BIN" -u -m atc_rl.competition --env ma --model "$RUN_DIR/train/model.zip" \
    --episodes 2 --seed 20260 --out "$RUN_DIR/native"
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
from pathlib import Path
import csv,json,sys
from atc.metrics import METRICS
root=Path(sys.argv[1])
def rows(name):
    with (root/name/'aircraft.csv').open(newline='') as stream:
        return {(int(r['episode']),r['agent']):r for r in csv.DictReader(stream)}
a,b=rows('vector'),rows('native')
assert a.keys()==b.keys() and len(a)==20
for key,row in a.items():
    assert row['scenario_sha256']==b[key]['scenario_sha256']
    for metric in METRICS:assert float(row[metric])==float(b[key][metric]),(key,metric)
result={'status':'complete','matched_metric_values':len(a)*len(METRICS),
        'cuda_reload':json.loads((root/'cuda-reload.json').read_text()),
        'training':json.loads((root/'train/training_summary.json').read_text()),
        'performance_evidence':False}
(root/'complete.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
PY
"$PYTHON_BIN" -m atc_rl.cluster verify
