import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


def capture(directory):
    """Save source and dependency evidence, including uncommitted experiment code."""
    root = Path(__file__).resolve().parents[1]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    def command(*args):
        return subprocess.check_output(git + list(args), text=True, encoding="utf-8")
    metadata = {"git_commit": command("rev-parse", "HEAD").strip(),
                "git_dirty": bool(command("status", "--porcelain").strip()),
                "python": platform.python_version(), "platform": platform.platform()}
    paths = command("ls-files", "--cached", "--others", "--exclude-standard", "-z").split(chr(0))
    hashes = {}
    with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(paths)):
            path = root / name
            if not name or not path.is_file():
                continue
            if path.suffix not in {".py", ".toml", ".lock", ".slurm", ".yaml", ".yml", ".md", ".sh", ".ps1"} and name not in {"LICENSE", ".gitignore", ".gitattributes"}:
                continue
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    metadata["source_sha256"] = hashes
    (directory / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (directory / "packages.txt").write_text(freeze, encoding="utf-8")
    return metadata
