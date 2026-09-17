"""Restore checksum-verified research archives without loading models or replay objects."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def restore(archive_path, manifest, verify_only=False):
    assets = {item['name']: item for item in manifest['assets']}
    expected = assets[archive_path.name]
    if archive_path.stat().st_size != expected['bytes'] or sha256(archive_path) != expected['sha256']:
        raise ValueError('Archive checksum/size mismatch: ' + archive_path.name)
    records = {item['path']: item for item in manifest['files'] if item.get('asset') == archive_path.name}
    restored = identical = 0
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(records) or set(archive.namelist()) != set(records):
            raise ValueError('Archive members do not match manifest')
        for name, record in records.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'runs' or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
            destination = ROOT.joinpath(*relative.parts)
            if not destination.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Destination escapes repository')
            if archive.getinfo(name).file_size != record['bytes']:
                raise ValueError('Member size differs: ' + name)
            if verify_only:
                h = hashlib.sha256()
                with archive.open(name) as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        h.update(chunk)
                if h.hexdigest() != record['sha256']:
                    raise ValueError('Member checksum differs: ' + name)
                continue
            if destination.exists():
                if sha256(destination) != record['sha256']:
                    raise FileExistsError('Preserving different local file: ' + name)
                identical += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.restore-pending')
            try:
                with temporary.open('xb') as output, archive.open(name) as source:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                if sha256(temporary) != record['sha256']:
                    raise ValueError('Restored checksum differs: ' + name)
                if destination.exists():
                    raise FileExistsError('Destination appeared during restore: ' + name)
                temporary.rename(destination)
                restored += 1
            except BaseException:
                # Leave a partial file for inspection; never remove or replace old evidence.
                raise
    print(json.dumps({'asset': archive_path.name, 'verified': len(records), 'restored': restored, 'already_identical': identical}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'docs/competition/artifacts/manifest.json').read_text(encoding='utf-8'))
    for path in args.archives:
        restore(path.resolve(), manifest, args.verify_only)

if __name__ == '__main__':
    main()