#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ATC_PYTHON="${ATC_PYTHON:-$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python}"
test -x "$ATC_PYTHON"
python3 -m atc_rl.cluster verify
# Refuse duplicate launches, including after partially successful submission.
mkdir -p runs
mkdir runs/cluster-ppo-gsde-v1
submit_job() {
    local output job
    if ! output=$(sbatch "$@"); then
        printf 'Scheduler submission failed: %s\n' "$output" >&2
        return 1
    fi
    job=${output%%;*}
    case "$job" in ''|*[!0-9]*) printf 'Invalid scheduler reply: %s\n' "$output" >&2; return 2 ;; esac
    printf '%s\n' "$job"
}
receipt=runs/cluster-ppo-gsde-v1/submissions.tsv
printf 'stage\tjob_id\n' > "$receipt"
first=$(submit_job --parsable --export=ALL --array=0 jobs/train_ppo_gsde_v1.slurm)
first=${first%%;*}
printf 'train_row0\t%s\n' "$first" >> "$receipt"
printf 'First training job: %s\n' "$first"
rest=$(submit_job --parsable --export=ALL --array=1-5%1 --dependency="afterok:$first" --kill-on-invalid-dep=yes jobs/train_ppo_gsde_v1.slurm)
rest=${rest%%;*}
printf 'train_rows1_5\t%s\n' "$rest" >> "$receipt"
eval_first=$(submit_job --parsable --export=ALL --array=0 --dependency="afterok:$first" --kill-on-invalid-dep=yes jobs/evaluate_ppo_gsde_v1.slurm)
eval_first=${eval_first%%;*}
printf 'eval_row0\t%s\n' "$eval_first" >> "$receipt"
eval_rest=$(submit_job --parsable --export=ALL --array=1-5%1 --dependency="afterok:$rest" --kill-on-invalid-dep=yes jobs/evaluate_ppo_gsde_v1.slurm)
eval_rest=${eval_rest%%;*}
printf 'eval_rows1_5\t%s\n' "$eval_rest" >> "$receipt"
summary=$(submit_job --parsable --export=ALL --dependency="afterok:$first:$rest:$eval_first:$eval_rest" --kill-on-invalid-dep=yes jobs/analyze_ppo_gsde_v1.slurm)
summary=${summary%%;*}
printf 'analyze_all_rows\t%s\n' "$summary" >> "$receipt"
cat "$receipt"
printf 'Outputs: %s/runs/cluster-ppo-gsde-v1\n' "$PWD"
