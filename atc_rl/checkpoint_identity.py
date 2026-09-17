"""Compare policy states independently of checkpoint archive metadata."""
import hashlib
import io
import json
from pathlib import Path
import zipfile


def policy_fingerprint(path):
    import torch
    with zipfile.ZipFile(path) as archive:
        state=torch.load(io.BytesIO(archive.read('policy.pth')),map_location='cpu',weights_only=True)
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if not isinstance(value,torch.Tensor):raise ValueError('Unexpected non-tensor policy state')
        value=value.detach().cpu().contiguous()
        header=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'big'));digest.update(header)
        data=value.numpy().tobytes();digest.update(len(data).to_bytes(8,'big'));digest.update(data)
    return digest.hexdigest()


def verified_checkpoint(directory,record):
    directory=Path(directory).resolve();path=(directory/record['file']).resolve()
    if path.parent!=directory or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Checkpoint does not match its integrity record')
    return path
