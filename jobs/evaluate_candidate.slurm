#!/bin/bash
#SBATCH --job-name=atc-score
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-score-%j.log

# Submit one job per track after confirming the cluster's CPU partition/account.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
PYTHON_BIN="${ATC_PYTHON:?Set ATC_PYTHON to the installed Python 3.12 executable}"
ARCHIVE="${ATC_CANDIDATE_ARCHIVE:?Set ATC_CANDIDATE_ARCHIVE to an absolute archive path}"
TRACK="${ATC_ENV:?Set ATC_ENV to sa or ma}"
case "${TRACK}" in sa|ma) ;; *) echo 'ATC_ENV must be sa or ma' >&2; exit 2 ;; esac
if [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -ne 1 ]]; then
    echo 'Submit one job per track; this template does not select array seeds or tracks.' >&2
    exit 2
fi
if [[ "${PYTHON_BIN}" != /* || "${ARCHIVE}" != /* ]]; then
    echo 'Python and archive paths must be absolute.' >&2
    exit 2
fi
EXPECTED_SHA=da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c
ACTUAL_SHA="$(sha256sum "${ARCHIVE}")"
ACTUAL_SHA="${ACTUAL_SHA%% *}"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo 'Candidate archive hash differs from the fixed decimal-heading-v1 release.' >&2
    exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1
export ATC_ENV="${TRACK}"
export ATC_JOB_RESULTS_DIR="${SLURM_SUBMIT_DIR}/runs/cluster-score-${TRACK}-${SLURM_JOB_ID}"
mkdir -p "${SLURM_SUBMIT_DIR}/runs"
mkdir "${ATC_JOB_RESULTS_DIR}"  # Refuse to reuse or overwrite an existing run.
cp -- "${BASH_SOURCE[0]}" "${ATC_JOB_RESULTS_DIR}/job.slurm"
printf '%s\n' "${EXPECTED_SHA}" > "${ATC_JOB_RESULTS_DIR}/archive.sha256"
"${PYTHON_BIN}" -m zipfile -e "${ARCHIVE}" "${ATC_JOB_RESULTS_DIR}/candidate"
cd "${ATC_JOB_RESULTS_DIR}/candidate"
"${PYTHON_BIN}" verify_contents.py
"${PYTHON_BIN}" -m pip freeze > "${ATC_JOB_RESULTS_DIR}/packages.txt"
# Each job owns its extraction, including simulator settings, caches and checks.
"${PYTHON_BIN}" check_development.py --env "${TRACK}"
cd source
"${PYTHON_BIN}" -m atc.deployment --env "${TRACK}" \
    --model "../models/${TRACK}/model.zip" \
    --out "${ATC_JOB_RESULTS_DIR}/metrics.csv"
"${PYTHON_BIN}" - <<'PY'
import csv, hashlib, json, os, platform
from datetime import datetime, timezone
from pathlib import Path
from atc.metrics import METRICS, summarize
out = Path(os.environ['ATC_JOB_RESULTS_DIR'])
kind = os.environ['ATC_ENV']
with (out / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
agents = 1 if kind == 'sa' else 10
assert [int(row['episode_index']) for row in rows] == list(range(1000 * agents))
records = [{'episode': i // agents, 'agent': f'completion-slot-{i % agents}',
            **{key: float(row[key]) for key in METRICS}}
           for i, row in enumerate(rows)]
result = summarize(records, 1000, agents)
manifest = json.loads((out / f'candidate/models/{kind}/deployment.json').read_text())
model = out / f'candidate/models/{kind}/model.zip'
assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest['model_sha256']
result.update(env=kind, seed=42, official_protocol=True, judge_verified=False,
              completed_at_utc=datetime.now(timezone.utc).isoformat(),
              model_sha256=manifest['model_sha256'],
              heading_transport=manifest['heading_transport'],
              csv_sha256=hashlib.sha256((out / 'metrics.csv').read_bytes()).hexdigest(),
              platform=platform.platform(), python=platform.python_version(),
              slurm_job_id=os.environ['SLURM_JOB_ID'],
              record_grouping='Consecutive fixed-count scenario groups; completion slots are not aircraft identities',
              interpretation='Reproduction of fixed candidates; do not use these scores for checkpoint selection')
(out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
PY
cd ..
"${PYTHON_BIN}" verify_contents.py
printf 'Completed fixed candidate evaluation: %s\n' "${ATC_JOB_RESULTS_DIR}"
