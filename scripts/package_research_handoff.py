"""Publish an immutable inventory of local experiment evidence and byte-only archives.

Does not load models, replay buffers, or pickle objects; does not alter experiments.
Run once per publication directory, before staging the recorded Git paths.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEXT = {'.json', '.jsonl', '.csv', '.tsv', '.md', '.py', '.log', '.txt', '.xml', '.ps1', '.sh', '.slurm', '.sbatch', '.toml', '.cfg', '.ini', '.ris', '.html', '.yaml', '.yml'}
BINARY_SOFTWARE = {'.whl', '.pyc', '.pyo', '.so', '.dll', '.exe', '.lib', '.a', '.pyd'}
# Deliberately report only filenames and rule labels, never potential credential values.
SECRET_RULES = [
    ('github-token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b')),
    ('github-fine-token', re.compile(r'github_pat_[A-Za-z0-9_]{40,}')),
    ('private-key', re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')),
    ('provider-token', re.compile(r'\b(?:sk-proj-|hf_)[A-Za-z0-9_-]{25,}')),
    ('aws-access-key', re.compile(r'\bAKIA[A-Z0-9]{16}\b')),
]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def classify(path):
    parts = path.relative_to(ROOT).parts
    folder = parts[1] if len(parts) > 1 else ''
    if folder.startswith('publication-'):
        return 'exclude', 'publication scratch/output (manifest is stored separately)'
    if folder.startswith('pytest-') or any(part.startswith(('test_', 'submission-')) or part == 'tmp' or part.startswith('tmp-') for part in parts[2:-1]):
        return 'exclude', 'generated test fixtures; test code and validation logs retained'
    if path.suffix == '.p':
        return 'exclude', 'regenerable simulator navigation cache'
    if folder == 'course-guidelines-review-v1':
        return 'exclude', 'course document reference, not project work'
    if folder == 'public-source-a458870c' and path.name.endswith('.txt'):
        return 'exclude', 'third-party source mirror; attribution and upstream links retained in PUBLIC_WORK.md'
    if folder.endswith('-verify'):
        source = ROOT / 'runs' / (folder[:-7] + '.zip')
        if source.is_file():
            return 'exclude', 'duplicate expanded source; original zip retained'
    if any(p in {'__pycache__', '.pytest_cache', '.venv', '.git', 'site-packages', 'workers', 'simulator', 'cache', 'wheelhouse', 'wheelhouse-v2', 'download-tmp', 'download-v2-tmp', 'zmq-build'} for p in parts[1:]):
        return 'exclude', 'generated runtime, dependency, simulator cache or worker directory'
    if folder == 'linux-runtime-v1' and ('python' in parts[2:] or path.name.startswith('cpython-')):
        return 'exclude', 'downloaded Python runtime'
    if path.suffix.lower() in BINARY_SOFTWARE or path.name == '.DS_Store':
        return 'exclude', 'compiled cache or downloaded dependency'
    if path.name == '.env' or path.suffix in {'.pem', '.key'}:
        return 'exclude', 'local credential configuration'
    if path.suffix.lower() in TEXT or path.name in {'LICENSE', 'Makefile'}:
        return 'git', 'experiment source, protocol, measurements or log'
    return 'release', 'historical replay bytes' if path.suffix == '.pkl' else 'model, dataset, frozen bundle or media'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='publication-20260917')
    parser.add_argument('--tag', default='research-handoff-20260917')
    args = parser.parse_args()
    if not re.fullmatch(r'publication-[a-zA-Z0-9-]+', args.name):
        raise ValueError('Use a simple publication directory name')
    out = ROOT / 'runs' / args.name
    out.mkdir(exist_ok=False)
    published = ROOT / 'docs' / 'competition' / 'artifacts'
    published.mkdir(exist_ok=True)
    manifest_path = published / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    files = sorted((p for p in (ROOT / 'runs').rglob('*') if stat.S_ISREG(p.lstat().st_mode)), key=lambda p: p.as_posix())
    entries, excluded, records, suspicious = [], [], [], []
    for path in files:
        route, reason = classify(path)
        rel = path.relative_to(ROOT).as_posix()
        row = {'path': rel, 'bytes': path.lstat().st_size, 'storage': route, 'reason': reason}
        if route == 'exclude':
            excluded.append(row)
            continue
        row['sha256'] = digest(path)
        if route == 'git':
            for rule, expression in SECRET_RULES:
                if expression.search(path.read_text(encoding='utf-8', errors='replace')):
                    suspicious.append({'path': rel, 'rule': rule})
            records.append(rel)
        entries.append(row)
    if suspicious:
        (out / 'credential-review.json').write_text(json.dumps(suspicious, indent=2))
        raise RuntimeError('Potential credentials found; see filename-only review, no publication performed')
    (out / 'git-records.txt').write_text('\n'.join(records) + '\n', encoding='utf-8')
    print(json.dumps({'records': len(records), 'binary_files': sum(e['storage'] == 'release' for e in entries), 'excluded': len(excluded)}), flush=True)
    (out / 'inventory.json').write_text(json.dumps({'files': entries, 'excluded': excluded}, indent=2), encoding='utf-8')
    assets = []
    for category in ('artifacts', 'replay'):
        selected = [e for e in entries if e['storage'] == 'release' and (e['path'].endswith('.pkl')) == (category == 'replay')]
        batches = []
        for item in selected:
            if not batches or sum(e['bytes'] for e in batches[-1]) + item['bytes'] > 1_100_000_000:
                batches.append([])
            batches[-1].append(item)
        for number, batch in enumerate(batches, 1):
            archive_name = f'{category}-{number:02d}.zip'
            archive_path = out / archive_name
            with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for item in batch:
                    path = ROOT / item['path']
                    archive.write(path, item['path'])
                    item['asset'] = archive_name
                    if digest(path) != item['sha256']:
                        raise RuntimeError('Input changed while archiving: ' + item['path'])
            assets.append({'name': archive_name, 'bytes': archive_path.stat().st_size, 'sha256': digest(archive_path), 'files': len(batch), 'uncompressed_bytes': sum(e['bytes'] for e in batch)})
            print(json.dumps(assets[-1]), flush=True)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'release_tag': args.tag,
                'repository': 'psh12320/bluesky-gym', 'scope': 'Local experiment records and artifacts; remote-only cluster artifacts are not present.',
                'files': entries, 'assets': assets, 'excluded': excluded}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (out / 'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets), encoding='utf-8')
    groups = {}
    for entry in entries:
        key = entry['path'].split('/')[1]
        groups.setdefault(key, []).append(entry)
    lines = ['# Experiment evidence index', '', 'Snapshot of retained local evidence. The handoff interprets the results; presence of a file is not a successful outcome. Binary artifacts are in the matching release and mapped by `artifacts/manifest.json`.', '', 'Historical paths and frozen source hashes are retained. Protocols may contain original absolute machine paths; adapt copies when rerunning, never edit the archived originals.', '', '| Run or record | Git files | Release files | Evidence entry points |', '| --- | ---: | ---: | --- |']
    priority = ['three-seed-results.json', 'paired-three-seed-results.json', 'paired-results.json', 'results.json', 'comparison.json', 'decision.json', 'receipt.json', 'complete.json', 'summary.json', 'protocol.json', 'state.json']
    for key, members in sorted(groups.items()):
        ranked = sorted((e for e in members if e['storage'] == 'git' and Path(e['path']).name in priority), key=lambda e: (len(Path(e['path']).parts), priority.index(Path(e['path']).name), e['path']))[:4]
        links = ', '.join(f"[{e['path'].removeprefix('runs/' + key + '/') if e['path'].startswith('runs/' + key + '/') else Path(e['path']).name}](../../{e['path']})" for e in ranked)
        lines.append(f"| `{key}` | {sum(e['storage']=='git' for e in members)} | {sum(e['storage']=='release' for e in members)} | {links} |")
    (published.parent / 'EXPERIMENT_INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(manifest_path), 'run_groups': len(groups), 'assets': len(assets), 'release_bytes': sum(a['bytes'] for a in assets)}), flush=True)

if __name__ == '__main__':
    main()