import hashlib
import io
import zipfile
import pytest
import torch
from atc_rl.checkpoint_identity import policy_fingerprint,verified_checkpoint


def checkpoint(path,value,extra):
    data=io.BytesIO();torch.save({'actor.weight':torch.tensor([value])},data)
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('policy.pth',data.getvalue());archive.writestr('metadata',extra)
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_policy_identity_ignores_archive_metadata_but_detects_weight_changes(tmp_path):
    a=tmp_path/'a.zip';b=tmp_path/'b.zip';c=tmp_path/'c.zip'
    record=checkpoint(a,1.,'first');checkpoint(b,1.,'second');checkpoint(c,2.,'first')
    assert a.read_bytes()!=b.read_bytes()
    assert policy_fingerprint(a)==policy_fingerprint(b)
    assert policy_fingerprint(a)!=policy_fingerprint(c)
    assert verified_checkpoint(tmp_path,record)==a
    with pytest.raises(ValueError,match='integrity'):
        verified_checkpoint(tmp_path,{'file':b.name,'sha256':record['sha256']})


def test_checkpoint_paths_cannot_escape_the_run_directory(tmp_path):
    with pytest.raises(ValueError,match='integrity'):
        verified_checkpoint(tmp_path,{'file':'../outside.zip','sha256':'unused'})
