"""Check durable artifacts from the interrupted runs without declaring completion."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,io,json,zipfile
import torch

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def state(path):
    with zipfile.ZipFile(path) as z:
        policy=torch.load(io.BytesIO(z.read('policy.pth')),map_location='cpu',weights_only=True)
        optimizer=torch.load(io.BytesIO(z.read('policy.optimizer.pth')),map_location='cpu',weights_only=True)
        data=json.loads(z.read('data'))
    return policy,optimizer,data

def audit(directory):
    config=read(directory/'config.json');provenance=read(directory/'provenance.json')
    manifest=read(directory/'checkpoints.json')
    assert not (directory/'training_summary.json').exists()
    assert len({r['file'] for r in manifest})==len(manifest)
    with zipfile.ZipFile(directory/'source.zip') as z:
        assert set(z.namelist())==set(provenance['source_sha256'])
        for name,digest in provenance['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    with (directory/'learning.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    by_rollout={int(r['rollout']):r for r in rows}
    assert set(by_rollout)==set(range(1,len(rows)+1))
    for row in rows:assert int(row['live_transitions'])+int(row['padded_transitions'])==int(row['counted_transitions'])
    initial=[r for r in manifest if r['file']=='initial-model.zip'];assert len(initial)==1
    assert all(initial[0][k]==0 for k in ('live_transitions','counted_transitions','padded_transitions','optimizer_steps','policy_epochs','rollouts'))
    before,_,_=state(directory/'initial-model.zip');checks=[];previous=-1
    for record in manifest:
        path=(directory/record['file']).resolve();assert path.parent==directory.resolve() and sha(path)==record['sha256']
        assert record['live_transitions']>=previous;previous=record['live_transitions']
        assert record['live_transitions']+record['padded_transitions']==record['counted_transitions']
        policy,optimizer,data=state(path)
        assert set(policy)==set(before) and all(torch.isfinite(v).all() for v in policy.values())
        assert data['num_timesteps']==record['counted_transitions'] and data['_n_updates']==record['policy_epochs']
        steps={int(v['step'].item()) for v in optimizer['state'].values() if 'step' in v}
        if record['live_transitions']:
            row=by_rollout[record['rollouts']]
            for k in ('live_transitions','counted_transitions','padded_transitions','optimizer_steps','policy_epochs'):assert int(row[k])==record[k]
            assert steps=={record['optimizer_steps']}
            changed=[k for k in policy if not torch.equal(policy[k],before[k])]
            actor=lambda k:k=='log_std' or k.startswith(('action_net.','mlp_extractor.policy_net.'))
            assert any(actor(k) for k in changed) and any(not actor(k) for k in changed)
        else:assert not steps
        checks.append({**record,'serialized_counts_match':True,'finite_weights':True,'optimizer_counters_match':True})
    return {'run':str(directory),'status':'interrupted_partial','requested_live_transitions':config['live_steps'],
            'last_logged_live_transitions':int(rows[-1]['live_transitions']),
            'last_durable_live_transitions':manifest[-1]['live_transitions'],
            'verified_source_files':len(provenance['source_sha256']),'checkpoints':checks,
            'limitations':['No final training summary or complete per-aircraft training history survived.',
                'This validates durable weights and logged accounting; it is not the completed-run audit.',
                'Policy quality requires separate physical evaluation; million-step budget not met.']}

if __name__=='__main__':
    torch.set_num_threads(1);parent=Path(__file__).resolve().parent
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'runs':[audit(parent/name) for name in ('native','scaled')]}
    with (parent/'partial-checkpoint-audit-v1.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps([{'run':r['run'],'checkpoints':len(r['checkpoints']),'last_durable_live_transitions':r['last_durable_live_transitions']} for r in result['runs']]))
