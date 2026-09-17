from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from stable_baselines3 import PPO
from atc_rl.initialization import neutral_action_mean
from atc_rl.checkpoint_identity import policy_fingerprint

def main():
    torch.set_num_threads(1)
    original=ROOT/'runs/ppo-direct-pilot-v1/initial-model.zip'
    destination=ROOT/'runs/ppo-initialization-v1/neutral-reference'
    destination.mkdir(exist_ok=False)
    model=PPO.load(original,device='cpu')
    assert model.num_timesteps==0 and model._n_updates==0
    before={name:p.detach().clone() for name,p in model.policy.named_parameters()}
    neutral_action_mean(model)
    with torch.no_grad():model.policy.log_std.fill_(math.log(.05))
    model.policy_kwargs['log_std_init']=math.log(.05)
    for name,p in model.policy.named_parameters():
        if name!='log_std' and not name.startswith('action_net.'):
            assert torch.equal(p,before[name])
    checkpoint=destination/'initial-model.zip';model.save(checkpoint)
    config=json.loads((original.parent/'config.json').read_text())
    config.update(run_dir=str(destination),initial_action_std=.05,neutral_action_mean=True,
        reward_scale=1.0,progress_scale=0.0,
        record_type='Untrained initialization reference; no simulator experience or optimization performed.')
    (destination/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    record={'file':checkpoint.name,'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'live_transitions':0,'counted_transitions':0,'padded_transitions':0,'optimizer_steps':0,'policy_epochs':0,'rollouts':0}
    (destination/'checkpoints.json').write_text(json.dumps([record],indent=2),encoding='utf-8')
    files=[Path(__file__),ROOT/'pyproject.toml']
    for package in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo'):files.extend((ROOT/package).rglob('*.py'))
    hashes={}
    with zipfile.ZipFile(destination/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name=path.relative_to(ROOT).as_posix();data=path.read_bytes()
            hashes[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
    provenance={'source_sha256':hashes,'original_checkpoint':str(original),
        'original_checkpoint_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'transformation':'Zero fresh action mean weights/bias; set log standard deviation to log(0.05). Preserve all hidden and critic weights.',
        'optimizer_updates':0,'simulator_transitions':0,'policy_fingerprint':policy_fingerprint(checkpoint)}
    (destination/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'reference':str(checkpoint),'policy_fingerprint':provenance['policy_fingerprint'],'training_performed':False}))

if __name__=='__main__':main()
