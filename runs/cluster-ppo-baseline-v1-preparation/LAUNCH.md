# Launch the selected three-seed PPO baseline

Bootstrap job 851402 passed. This experiment reuses the existing environment without installing packages.
Original source directories, checkpoints and logs remain in place. Use this fresh extraction once.

## Windows PowerShell

```powershell
scp -J shri@sjump.comp.nus.edu.sg "C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym\runs\cluster-ppo-baseline-v1.zip" shri@xlogin.comp.nus.edu.sg:cs4246-rl/
```

## Existing xlogin session

```bash
(
set -euo pipefail
cd "$HOME/cs4246-rl"
printf '%s  %s\n' \
  '27bec8a905b56360225ab15db4ddc33d0f3235733cb0425d47c4263de00c3d8c' \
  'cluster-ppo-baseline-v1.zip' | sha256sum -c -
mkdir cluster-ppo-baseline-v1
python3 -m zipfile -e cluster-ppo-baseline-v1.zip cluster-ppo-baseline-v1
cd cluster-ppo-baseline-v1
export ATC_PYTHON="$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python"
bash jobs/submit_ppo_baseline_v1.sh
)
```

The launcher records four Slurm submissions: first training seed, remaining two training seeds, first-seed evaluation, and remaining two evaluations. Each array runs at most one task at a time. There is at most one GPU training task running; CPU evaluation can overlap training. If submission stops partway through, preserve submissions.tsv and report the error; do not rerun the launcher.

Each training allocation requests one H200, 16 CPUs, 64 GB RAM and three hours. Eight simulator worlds use CPU processes; the GPU trains the network. Seeds are 50100, 50200 and 50300, with 1,000,000 live transitions each. Padding is excluded from the budget. Training stops before the job deadline; a partial run is retained and rejected as incomplete.

Recipe: shared PPO, goal_offset action reference, neutral action mean, initial standard deviation 0.05, reward scale 0.01, no progress shaping, no route guidance, no conflict filter. The goal-relative action mapping itself supplies a navigation prior; only initial-versus-trained comparisons measure learning.

The first allocation runs the expanded RL test suite. Every completed run is audited. CPU jobs evaluate its own initial, first saved checkpoint at/above 100k and 300k, and final policy on the same twenty development worlds (stream 20260). They also evaluate the unchanged classical controller and save comparison figures and learning curves. Unseen streams remain reserved.

## Monitor

```bash
cd "$HOME/cs4246-rl/cluster-ppo-baseline-v1"
cat runs/cluster-ppo-baseline-v1/submissions.tsv
squeue -u "$USER"
jobs_csv=$(awk 'NR>1 {print $2}' runs/cluster-ppo-baseline-v1/submissions.tsv | paste -sd, -)
sacct -j "$jobs_csv" --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
first=$(awk '$1=="train_seed50100" {print $2}' runs/cluster-ppo-baseline-v1/submissions.tsv)
tail -n 60 "ppo-baseline-train-${first}_0.log"
```

Each seed's artifacts are under runs/cluster-ppo-baseline-v1/seed-SEED/: train/, evaluations/, comparison/ and curve/. Report all three seed outcomes, not only the strongest. The three-seed aggregate is a follow-up after all evaluations finish.

## Validation

127 manifest files verified; all 122 base files are byte-identical to onpolicy-learning-source-v2. The three new shell files passed syntax checks. Stubbed scheduler/driver checks verified dependency wiring, initial-policy evaluation, checkpoint selection delegation, budget gating and preservation on partial or duplicate submissions. These checks did not submit a real cluster job; expanded GPU-host tests run in the first allocation.
