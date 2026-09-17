from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    path=(ROOT/name).resolve()
    assert ROOT in path.parents, name
    assert path.is_file(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
print(f"Verified {len(manifest['files'])} packaged files")
