import hashlib
import json

import pytest

from atc.transfer import verify


def test_transfer_verification_detects_changed_source_and_base_revision(tmp_path, monkeypatch):
    source = tmp_path / "controller.py"
    source.write_text("initial source", encoding="utf-8")
    manifest = tmp_path / "runs/cluster-transfer/provenance.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"git_commit": "abc123", "source_sha256": {
        "controller.py": hashlib.sha256(source.read_bytes()).hexdigest(),
    }}), encoding="utf-8")
    monkeypatch.setattr("atc.transfer.subprocess.check_output", lambda *a, **kw: "abc123\n")
    assert verify(tmp_path) == 1
    source.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="controller.py"):
        verify(tmp_path)
    source.write_text("initial source", encoding="utf-8")
    monkeypatch.setattr("atc.transfer.subprocess.check_output", lambda *a, **kw: "def456\n")
    with pytest.raises(ValueError, match="Git HEAD"):
        verify(tmp_path)
