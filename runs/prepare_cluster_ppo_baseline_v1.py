from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile
root=Path.cwd()
work=root/'runs/cluster-ppo-baseline-v1-preparation'
work.mkdir(exist_ok=False)
sha=lambda data: hashlib.sha256(data).hexdigest()
attachment=Path(r'C:\Users\Shricharan\.codex\attachments\4e4dc70c-84ab-4bef-8c64-f43ab89aadbf\pasted-text.txt')
receipt=root/'runs/cluster-receipt-851402'
receipt.mkdir(exist_ok=False)
(receipt/'user-transcript.txt').write_bytes(attachment.read_bytes())
record={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'job_id':851402,'state':'COMPLETED','exit_code':'0:0',
        'elapsed_seconds':62,'allocated_cpus':16,'max_rss_kib':4690400,
        'transcript_sha256':sha(attachment.read_bytes()),'environment':'/home/s/shri/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python',
        'source_files_verified':84,'smoke_live_transitions':15258,'smoke_optimizer_steps':30,
        'classical_parity':'Two scenarios and all 180 aircraft metrics match the Windows reference.',
        'scope':'Bootstrap integration passed. The tiny raw PPO smoke policy has zero arrivals; this is not trained-policy performance evidence.',
        'evidence':'User-supplied sacct output and log tail; full remote log and artifacts have not been downloaded.'}
(receipt/'receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
base=root/'runs/onpolicy-learning-source-v2.zip'
base_sha='33579f122d1fffd6bf6f9aefdabad846111afe20cbaadb56e390ae2d5a849b8c'
assert sha(base.read_bytes())==base_sha
with zipfile.ZipFile(base) as z:files={name:z.read(name) for name in z.namelist()}
manifest=json.loads(files.pop('cluster-manifest.json'))
assert set(files)==set(manifest['files'])
assert all(sha(files[name])==digest for name,digest in manifest['files'].items())
protocol={
 'registered_at_utc':datetime.now(timezone.utc).isoformat(),'experiment':'cluster-ppo-baseline-v1',
 'purpose':'Replicate the most promising unsupported PPO recipe across three fixed training seeds.',
 'seeds':[50100,50200,50300],
 'config':{'algorithm':'ppo','workers':8,'live_steps':1000000,'rollout_steps':256,'batch_size':1024,'epochs':10,
           'device':'cuda','checkpoint_live_steps':100000,'max_wall_seconds':9900,
           'initial_action_std':.05,'neutral_action_mean':True,'action_reference':'goal_offset',
           'guidance':False,'filter':False,'progress_scale':0.0,'reward_scale':.01},
 'evaluation':{'seed':20260,'worlds':20,'stages':['initial','100k','300k','final'],
               'checkpoint_selection':'First saved checkpoint at/above each threshold, within one rollout.',
               'reward_scale':1.0,'progress_scale':0.0,'classical':'Fixed controller, unchanged; rerun per seed.'},
 'advance_screen':{'arrival_at_least':.95,'arrival_change_at_least':-.02,'clean_completion_change_at_least':.05,
                   'safety_time_changes_at_most':0.0,'metrics':['intrusion_time','time_in_restricted_area','time_outside_sector']},
 'resources':{'training':{'partition':'gpu','gpu_request':'h200-141:1','cpus':16,'memory_gb':64,'hours':3},
              'evaluation':{'partition':'normal','cpus':2,'memory_gb':8,'hours':3}},
 'source_base_sha256':base_sha,'bootstrap_job':851402,
 'interpretation':['Goal-relative actions supply a navigation prior; compare with each seed own untrained policy.',
                   'Retain every seed and failure; no best-seed selection.',
                   'Eight worlds and CUDA differ from the two-world local pilot; this is not an exact trajectory replication.',
                   'Report the three seed effects and sample standard deviation; scenario bootstrap intervals do not measure seed uncertainty.',
                   'Only the paired initial/final comparison isolates learning within this recipe.',
                   'Competitive performance remains unproven.'],
 'remaining_experiments':['Separate trained guidance/filter ablations','Matched centralized-critic MAPPO if justified',
                          'Preserved SAC comparison','Unseen streams 20301 and 20302 after candidate selection'],
 'scheduling':'First seed gates the other two; one GPU task at a time. CPU evaluations depend on completed-budget training.',
 'failure_behavior':'Stop on failed checks or incomplete training. Never overwrite existing output.'
}
new={
'jobs/ppo_baseline_v1.json':json.dumps(protocol,indent=2)+'\n',
'jobs/ppo_baseline_v1.py':r'''"""Run the fixed cluster baseline using existing training and evaluation modules."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atc_rl.cluster import verify
from atc_rl.matrix_evaluate import select_checkpoint

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def run(module, *args):
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--index", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    verify(ROOT)
    plan = read(ROOT / "jobs/ppo_baseline_v1.json")
    seed = plan["seeds"][args.index]
    parent = ROOT / "runs/cluster-ppo-baseline-v1" / f"seed-{seed}"
    training = parent / "train"
    expected = {**plan["config"], "seed": seed}
    if args.action == "train":
        parent.mkdir(parents=True, exist_ok=False)
        if args.index == 0:
            run("pytest", "tests/rl", "-q", "-p", "no:cacheprovider",
                "--basetemp", parent / "tests", "--junitxml", parent / "tests.xml")
        command = []
        for key, value in expected.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.extend([flag, str(value)])
        run("atc_rl.train", *command, "--run-dir", training)
        run("atc_rl.audit", "--run", training, "--out", training / "checkpoint-audit.json")
    config = read(training / "config.json")
    summary = read(training / "training_summary.json")
    audit = read(training / "checkpoint-audit.json")
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Training differs from the registered recipe")
    maximum_rollout = expected["workers"] * 10 * expected["rollout_steps"]
    if summary["status"] != "complete" or not expected["live_steps"] <= summary["live_transitions"] < expected["live_steps"] + maximum_rollout:
        raise ValueError("Partial training preserved; requested live-transition budget not completed")
    for key in ("status", "live_transitions", "optimizer_steps", "model_sha256"):
        if audit[key] != summary[key]:
            raise ValueError("Checkpoint audit does not match training summary")
    manifest = read(ROOT / "cluster-manifest.json")
    source = {name: digest for name, digest in manifest["files"].items()
              if name == "pyproject.toml" or name.endswith(".py") and name.split("/")[0] in
              ("atc", "atc_rl", "core", "bluesky_gym", "bluesky_zoo")}
    if read(training / "provenance.json")["source_sha256"] != source:
        raise ValueError("Training source differs from the immutable bundle")
    if args.action == "evaluate":
        output = parent / "evaluations"
        output.mkdir(exist_ok=False)
        common = ("--episodes", plan["evaluation"]["worlds"], "--seed", plan["evaluation"]["seed"])
        classical = output / "classical"
        run("atc_rl.evaluate", "--classical", *common, "--out", classical)
        evaluations = {}
        for stage in plan["evaluation"]["stages"]:
            checkpoint, _ = select_checkpoint(training, stage)
            evaluations[stage] = output / stage
            run("atc_rl.evaluate", "--model", checkpoint, *common, "--out", evaluations[stage])
        run("atc_rl.compare", "--initial", evaluations["initial"], "--trained", evaluations["final"],
            "--classical", classical, "--out", parent / "comparison")
        curve_args = []
        for directory in evaluations.values():
            curve_args.extend(["--evaluation", directory])
        run("atc_rl.curves", "--training-run", training, "--classical", classical,
            "--out", parent / "curve", *curve_args)
    verify(ROOT)
    print(json.dumps({"completed": args.action, "seed": seed, "live_transitions": summary["live_transitions"]}), flush=True)

if __name__ == "__main__":
    main()
''',
'jobs/train_ppo_baseline_v1.slurm':r'''#!/bin/bash
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
''',
'jobs/evaluate_ppo_baseline_v1.slurm':r'''#!/bin/bash
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
''',
'jobs/submit_ppo_baseline_v1.sh':r'''#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ATC_PYTHON="${ATC_PYTHON:-$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python}"
test -x "$ATC_PYTHON"
python3 -m atc_rl.cluster verify
# Refuse duplicate launches, including after partially successful submission.
mkdir -p runs
mkdir runs/cluster-ppo-baseline-v1
receipt=runs/cluster-ppo-baseline-v1/submissions.tsv
printf 'stage\tjob_id\n' > "$receipt"
first=$(sbatch --parsable --export=ALL --array=0 jobs/train_ppo_baseline_v1.slurm)
first=${first%%;*}
printf 'train_seed50100\t%s\n' "$first" >> "$receipt"
printf 'First training job: %s\n' "$first"
rest=$(sbatch --parsable --export=ALL --array=1-2%1 --dependency="afterok:$first" --kill-on-invalid-dep=yes jobs/train_ppo_baseline_v1.slurm)
rest=${rest%%;*}
printf 'train_seed50200_50300\t%s\n' "$rest" >> "$receipt"
eval_first=$(sbatch --parsable --export=ALL --array=0 --dependency="afterok:$first" --kill-on-invalid-dep=yes jobs/evaluate_ppo_baseline_v1.slurm)
eval_first=${eval_first%%;*}
printf 'eval_seed50100\t%s\n' "$eval_first" >> "$receipt"
eval_rest=$(sbatch --parsable --export=ALL --array=1-2%1 --dependency="afterok:$rest" --kill-on-invalid-dep=yes jobs/evaluate_ppo_baseline_v1.slurm)
eval_rest=${eval_rest%%;*}
printf 'eval_seed50200_50300\t%s\n' "$eval_rest" >> "$receipt"
cat "$receipt"
printf 'Outputs: %s/runs/cluster-ppo-baseline-v1\n' "$PWD"
'''
}
for name,text in new.items():
    assert name not in files
    files[name]=text.replace('\r\n','\n').encode('utf-8')
    target=root/name
    if target.exists():raise FileExistsError(target)
    target.write_bytes(files[name])
manifest['files']={name:sha(data) for name,data in sorted(files.items())}
manifest['experiment']={'protocol':'jobs/ppo_baseline_v1.json','source_base_sha256':base_sha,
                        'unchanged_base_files':122,'added_files':sorted(new),'implementation':'Training and evaluation packages unchanged.'}
bundle=root/'runs/cluster-ppo-baseline-v1.zip'
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(files.items()):z.writestr(name,data)
    z.writestr('cluster-manifest.json',json.dumps(manifest,indent=2))
extracted=root/'runs/cluster-ppo-baseline-v1-verify'
extracted.mkdir(exist_ok=False)
with zipfile.ZipFile(bundle) as z:z.extractall(extracted)
metadata={'bundle':str(bundle),'sha256':sha(bundle.read_bytes()),'bytes':bundle.stat().st_size,
          'source_files':len(files),'base_sha256':base_sha,'base_files_unchanged':122,'added_files':sorted(new)}
bundle.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
(work/'protocol.json').write_bytes(files['jobs/ppo_baseline_v1.json'])
doc=root/'docs/competition/CLUSTER_RL.md'
text=doc.read_text(encoding='utf-8')
marker='## Transfer from Windows PowerShell'
prefix='''# Run the RL experiments on the university cluster

Bootstrap recovery job **851402 completed successfully** (62 seconds, exit 0:0),
reusing ~/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python. Its GPU training/reload
checks passed, and all 180 classical metrics across two scenarios matched Windows.
The user-supplied transcript is preserved under runs/cluster-receipt-851402/.
The full remote logs remain in their original cluster directories.

The next selected experiment is runs/cluster-ppo-baseline-v1.zip: three PPO seeds,
one million live transitions each, eight worlds, goal-relative actions, reward
scale 0.01, neutral mean, initial action standard deviation 0.05, no route guidance
and no conflict filter. Its fixed protocol is jobs/ppo_baseline_v1.json.
Use [the exact launch commands](../../runs/cluster-ppo-baseline-v1-preparation/LAUNCH.md).
The first training allocation checks the expanded RL test suite. Later training
and CPU evaluations depend on successful completed-budget training. All existing
source archives, environments, evaluations and logs are preserved. No reinstall
is needed.

The sections below retain the earlier setup and general experiment templates as
historical reference. The selected launch instructions above supersede their
raw-PPO commands. Keep passwords in the SSH/SCP prompt.

'''
assert marker in text
doc.write_text(prefix+text[text.index(marker):],encoding='utf-8')
print(json.dumps(metadata,indent=2))
