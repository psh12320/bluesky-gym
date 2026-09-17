#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ATC_PYTHON="${ATC_PYTHON:-$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python}"
test -x "$ATC_PYTHON"
python3 -m atc_rl.cluster verify
# Refuse duplicate launches, including after partially successful submission.
mkdir -p runs
mkdir runs/cluster-ppo-static-v1
receipt=runs/cluster-ppo-static-v1/submissions.tsv
printf 'stage\tjob_id\n' > "$receipt"
first=$(sbatch --parsable --export=ALL --array=0 jobs/train_ppo_static_v1.slurm)
first=${first%%;*}
printf 'train_row0\t%s\n' "$first" >> "$receipt"
printf 'First training job: %s\n' "$first"
rest=$(sbatch --parsable --export=ALL --array=1-5%1 --dependency="afterok:$first" --kill-on-invalid-dep=yes jobs/train_ppo_static_v1.slurm)
rest=${rest%%;*}
printf 'train_rows1_5\t%s\n' "$rest" >> "$receipt"
eval_first=$(sbatch --parsable --export=ALL --array=0 --dependency="afterok:$first" --kill-on-invalid-dep=yes jobs/evaluate_ppo_static_v1.slurm)
eval_first=${eval_first%%;*}
printf 'eval_row0\t%s\n' "$eval_first" >> "$receipt"
eval_rest=$(sbatch --parsable --export=ALL --array=1-5%1 --dependency="afterok:$rest" --kill-on-invalid-dep=yes jobs/evaluate_ppo_static_v1.slurm)
eval_rest=${eval_rest%%;*}
printf 'eval_rows1_5\t%s\n' "$eval_rest" >> "$receipt"
cat "$receipt"
printf 'Outputs: %s/runs/cluster-ppo-static-v1\n' "$PWD"
