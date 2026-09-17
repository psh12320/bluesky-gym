"""Package the current experiment source for an interactive cluster transfer."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from atc.provenance import capture


def verify(root):
    manifest = root / "runs/cluster-transfer/provenance.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in metadata["source_sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    commit = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                      "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != metadata["git_commit"]:
        mismatches.append("Git HEAD (checkout the commit recorded in the manifest)")
    if mismatches:
        raise ValueError("Transferred source differs: " + ", ".join(mismatches))
    return len(metadata["source_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "verify"])
    parser.add_argument("--out", type=Path, help="New ZIP path under runs/")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "verify":
        try:
            count = verify(root)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
        print(f"Verified {count} transferred source files and the upstream Git revision.")
        return
    if args.out is None:
        parser.error("pack requires --out")
    output = args.out.resolve()
    if not output.is_relative_to(root / "runs"):
        parser.error("Bundle output must be under this repository's runs/ directory")
    if output.exists():
        parser.error("Bundle already exists; use a new output name")
    # The source overlay does not represent deletions from the base checkout.
    deleted = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                      "-C", str(root), "diff", "--name-only", "--diff-filter=D", "HEAD"], text=True).strip()
    if deleted:
        parser.error("Source overlay cannot package deleted tracked files: " + deleted)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="transfer-", dir=output.parent) as temporary:
        metadata = capture(temporary)
        directory = Path(temporary)
        with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as destination:
            with zipfile.ZipFile(directory / "source.zip") as source:
                for name in source.namelist():
                    destination.writestr(name, source.read(name))
            for name in ("provenance.json", "packages.txt"):
                destination.write(directory / name, "runs/cluster-transfer/" + name)
    print(json.dumps({"bundle": str(output), "git_commit": metadata["git_commit"],
                      "source_files": len(metadata["source_sha256"]),
                      "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
