from pathlib import Path
import csv,hashlib,io,json,math,zipfile
import torch
from atc_rl.checkpoint_identity import policy_fingerprint
root=Path(__file__).resolve().parents[2];parent=Path(__file__).resolve().parent
torch.set_num_threads(1)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
new=parent/'new';old=parent/'reference'

def equal(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def optimizer(path):
    with zipfile.ZipFile(path) as z:return torch.load(io.BytesIO(z.read('policy.optimizer.pth')),map_location='cpu',weights_only=True)

for filename in ('initial-model.zip','model.zip'):
    assert policy_fingerprint(new/filename)==policy_fingerprint(old/filename)
    assert equal(optimizer(new/filename),optimizer(old/filename))
for filename in ('training-aircraft.csv','training-returns.csv'):
    assert (new/filename).read_bytes()==(old/filename).read_bytes(),filename
summaries=[read(path/'training_summary.json') for path in (new,old)]
for key in ('live_transitions','padded_transitions','counted_transitions','optimizer_steps','policy_epochs','rollouts','aircraft_completed'):
    assert summaries[0][key]==summaries[1][key],key
rows=[]
for path in (new,old):
    with (path/'learning.csv').open(newline='',encoding='utf-8') as stream:rows.append(list(csv.DictReader(stream)))
assert len(rows[0])==len(rows[1])==1
new_row,old_row=rows[0][0],rows[1][0]
for key in old_row:
    if key not in ('wall_seconds','live_transitions_per_second'):assert new_row[key]==old_row[key],key
assert int(new_row['rollout_live_samples'])==summaries[0]['live_transitions']==2270
assert summaries[0]['padded_transitions']==290
for key in ('rollout_return_variance_live','rollout_value_mse_live','rollout_value_explained_variance_live','rollout_action_box_clip_fraction_live'):
    assert math.isfinite(float(new_row[key])),key
assert 0<=float(new_row['rollout_action_box_clip_fraction_live'])<=1
for path,source_name in ((new,'onpolicy-diagnostics-source-v1-verify'),(old,'onpolicy-durable-source-v1-verify')):
    provenance=read(path/'provenance.json')
    assert all(sha(root/'runs'/source_name/name)==digest for name,digest in provenance['source_sha256'].items())
    audit=read(path/'checkpoint-audit.json');assert audit['status']=='complete' and not audit['policy_quality_assessed']
record=dict(full_initial_and_final_policy_tensors_identical=True,full_initial_and_final_optimizer_states_identical=True,
    all_original_nontiming_learning_fields_identical=True,completed_training_aircraft_and_returns_byte_identical=True,
    live_transitions=2270,padded_transitions=290,optimizer_steps=5,
    initial_fingerprint=policy_fingerprint(new/'initial-model.zip'),final_fingerprint=policy_fingerprint(new/'model.zip'),
    newer_model_sha256=sha(new/'model.zip'),reference_model_sha256=sha(old/'model.zip'),
    live_diagnostics={key:value for key,value in new_row.items() if key.startswith('rollout_')},
    full_rl_unit_tests_passed=119,both_completed_training_audits_passed=True,source_snapshots_unchanged=True,
    performance_evidence=False,scope='One real PPO rollout verifies observational logging parity; it does not establish policy quality or multi-seed performance.')
with (parent/'numerical-parity.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))