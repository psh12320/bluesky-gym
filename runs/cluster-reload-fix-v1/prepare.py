from pathlib import Path
import hashlib
import json
import zipfile

root = Path.cwd()
work = root / 'runs/cluster-reload-fix-v1'
work.mkdir(exist_ok=False)
original = root / 'runs/onpolicy-cluster-v1.zip'
original_sha = 'a1e578f3540edccddbf97506aff7eba5ac221b5a4946242cc154253b564dd293'
assert hashlib.sha256(original.read_bytes()).hexdigest() == original_sha
with zipfile.ZipFile(original) as archive:
    files = {name: archive.read(name) for name in archive.namelist()}
manifest = json.loads(files.pop('cluster-manifest.json'))
assert set(files) == set(manifest['files'])
for name, data in files.items():
    assert hashlib.sha256(data).hexdigest() == manifest['files'][name], name

name = 'tests/rl/test_policy_and_buffer.py'
before = b'AircraftPolicy.load(path)'
after = b'AircraftPolicy.load(path, device=model.device)'
assert files[name].count(before) == 1
files[name] = files[name].replace(before, after)
main_test = root / name
main_bytes = main_test.read_bytes()
assert main_bytes.count(before) == 1
main_test.write_bytes(main_bytes.replace(before, after))

name = 'jobs/bootstrap_onpolicy.slurm'
before = '''python3 -m venv .venv-cluster
PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv-cluster/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
"$PYTHON_BIN" -m pip install --prefer-binary -r jobs/requirements-onpolicy.txt'''
after = '''if [[ -n "${ATC_PYTHON:-}" ]]; then
    PYTHON_BIN="$ATC_PYTHON"
    if [[ "$PYTHON_BIN" != /* || ! -x "$PYTHON_BIN" ]]; then
        echo 'ATC_PYTHON must name an existing executable using an absolute path.' >&2
        exit 2
    fi
    printf 'Reusing environment without installing packages: %s\\n' "$PYTHON_BIN"
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
fi'''

def patch_job(data):
    text = data.decode('utf-8').replace('\r\n', '\n')
    assert text.count(before) == 1
    text = text.replace(before, after)
    text = text.replace('export PYTHONUNBUFFERED=1 ', 'export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ')
    assert text.count('mkdir -p "$CHECK_DIR"') == 1
    text = text.replace('mkdir -p "$CHECK_DIR"', 'mkdir -p runs\nmkdir "$CHECK_DIR"')
    return text.encode('utf-8')

files[name] = patch_job(files[name])
main_job = root / name
main_job.write_bytes(patch_job(main_job.read_bytes()))
changed = [name for name, data in files.items() if hashlib.sha256(data).hexdigest() != manifest['files'][name]]
assert sorted(changed) == ['jobs/bootstrap_onpolicy.slurm', 'tests/rl/test_policy_and_buffer.py']
manifest['files'] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
manifest['recovery'] = {'original_bundle': original.name, 'original_sha256': original_sha,
                        'failed_job': '851227', 'changed_files': sorted(changed),
                        'scope': 'Explicit test reload device; optional reuse of an existing environment; fresh check output.'}
archive_path = root / 'runs/onpolicy-cluster-reload-fix-v1.zip'
with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
    for name, data in files.items():
        archive.writestr(name, data)
    archive.writestr('cluster-manifest.json', json.dumps(manifest, indent=2))
verify_root = root / 'runs/onpolicy-cluster-reload-fix-v1-verify'
verify_root.mkdir(exist_ok=False)
with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(verify_root)
record = {'bundle': str(archive_path), 'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
          'bytes': archive_path.stat().st_size, 'source_files': len(files),
          'original_bundle_sha256': original_sha, 'changed_files': sorted(changed),
          'unchanged_source_files': len(files)-len(changed), 'failed_cluster_job': '851227',
          'environment_reuse_path': '$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python'}
archive_path.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
assert hashlib.sha256(original.read_bytes()).hexdigest() == original_sha
print(json.dumps(record, indent=2))