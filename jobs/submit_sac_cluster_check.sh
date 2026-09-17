#!/bin/bash
set -euo pipefail
PYTHON_BIN="${ATC_PYTHON:-$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python}"
test -x "$PYTHON_BIN"
"$PYTHON_BIN" -m atc_rl.cluster verify
mkdir -p runs/cluster-sac-check-v1
mkdir runs/cluster-sac-check-v1/submission-lock
submitted=$(sbatch --parsable jobs/sac_cluster_check.slurm)
job_id=${submitted%%;*}
case "$job_id" in
    ''|*[!0-9]*) echo "Unexpected Slurm response: $submitted" >&2; exit 2 ;;
esac
printf 'stage\tjob_id\ncheck\t%s\n' "$job_id" > runs/cluster-sac-check-v1/submissions.tsv
cat runs/cluster-sac-check-v1/submissions.tsv
