"""Create or verify a self-contained source bundle without changing Git state."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import zipfile


def verify(root):
    manifest=json.loads((root/'cluster-manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing or unsafe bundle path: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Bundle file changed: '+name)
    print(json.dumps({'verified_files':len(manifest['files']),'upstream_commit':manifest['upstream_commit']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['pack','verify'])
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.action=='verify':
        verify(root);return
    if args.out is None:parser.error('pack requires --out')
    destination=args.out.resolve()
    if not destination.is_relative_to(root/'runs') or destination.exists():
        parser.error('Choose a new archive under runs/')
    paths=[root/'pyproject.toml',root/'README.md',root/'LICENSE',root/'scripts/evaluate_competition.py',root/'docs/competition/NATIVE_POLICY.md',root/'docs/competition/EXPLORATION.md',root/'docs/competition/POSITION_SCALE.md',root/'docs/competition/CREDIT_ASSIGNMENT.md']
    for package in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo'):
        paths.extend((root/package).rglob('*.py'))
    paths.extend((root/'tests/rl').rglob('*.py'))
    paths.extend((root/'tests/rl/fixtures').rglob('*.json'))
    paths.extend(root/name for name in ('jobs/train_onpolicy.slurm','jobs/bootstrap_onpolicy.slurm',
        'jobs/requirements-onpolicy.txt','jobs/evaluate_onpolicy.slurm','jobs/train_onpolicy_matrix.slurm','jobs/evaluate_onpolicy_matrix.slurm',
        'docs/competition/REWARD_SCALE.md','docs/competition/PROGRESS_REWARD.md','docs/competition/INITIALIZATION.md',
        'docs/competition/GOAL_OFFSET.md','docs/competition/MATRIX_RL.md','docs/competition/STATIC_FILTER.md',
        'docs/competition/RL_PLAN.md','docs/competition/CLUSTER_RL.md'))
    commit=subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}',
        '-C',str(root),'rev-parse','HEAD'],text=True).strip()
    manifest={'upstream_commit':commit,'files':{},'execution':'Run modules from this root using PYTHONPATH; no project wheel install required.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            name=path.relative_to(root).as_posix();data=path.read_bytes()
            manifest['files'][name]=hashlib.sha256(data).hexdigest()
            archive.writestr(name,data)
        archive.writestr('cluster-manifest.json',json.dumps(manifest,indent=2))
    record={'bundle':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'bytes':destination.stat().st_size,'source_files':len(manifest['files']),'upstream_commit':commit}
    destination.with_suffix('.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
