"""Run physical parity checks before evaluating the preserved SAC reference."""
from pathlib import Path
import csv,hashlib,json,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
PARENT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(PARENT/'protocol.json')
assert sha(ROOT/'atc_rl/legacy_reference.py')==protocol['evaluator_sha256']
assert sha(Path(protocol['candidate'])/'manifest.json')==protocol['candidate_manifest_sha256']
reference=Path(protocol['classical_reference'])
assert sha(reference/'aircraft.csv')==protocol['classical_csv_sha256']
fixture=Path(protocol['original_harness_fixture'])
assert sha(fixture)==protocol['original_harness_fixture_sha256']
metrics=('waypoint_reached','flight_time','intrusion_events','intrusion_time','restricted_area_events',
         'time_in_restricted_area','sector_exit_events','time_outside_sector','total_reward')

def records(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def compare_rows(actual,expected):
    assert len(actual)==len(expected)==20
    deltas={key:max(abs(float(a[key])-float(b[key])) for a,b in zip(actual,expected)) for key in metrics}
    assert all(delta<=(1e-5 if key=='total_reward' else 0.) for key,delta in deltas.items()),deltas
    return deltas

checks=[]
for stage in protocol['stages']:
    directory=PARENT/stage['out']
    if directory.exists():raise RuntimeError('Preserve the existing stage; do not restart this runner: '+str(directory))
    print(json.dumps({'starting_stage':stage['out'],'episodes':stage['episodes'],'seed':stage['seed']}),flush=True)
    subprocess.run([sys.executable,'-m','atc_rl.legacy_reference','--candidate',protocol['candidate'],
        '--out',str(directory),'--mode',stage['mode'],'--episodes',str(stage['episodes']),'--seed',str(stage['seed'])],cwd=ROOT,check=True)
    actual=records(directory/'aircraft.csv')
    if stage['out']=='zero-reference-check':
        lookup={(r['episode'],r['agent']):r for r in records(reference/'aircraft.csv')}
        expected=[lookup[(r['episode'],r['agent'])] for r in actual]
        assert all(a['scenario_sha256']==b['scenario_sha256'] for a,b in zip(actual,expected))
        deltas=compare_rows(actual,expected)
    elif stage['out']=='harness-reproduction':
        deltas=compare_rows(actual,read(fixture)['actual_records'])
    else:
        break
    result={'stage':stage['out'],'metric_max_absolute_difference':deltas,'aircraft_verified':len(actual),
            'csv_sha256':sha(directory/'aircraft.csv'),'passed':True}
    checks.append(result)
    with (PARENT/(stage['out']+'-validation.json')).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)

sys.path.insert(0,str(ROOT))
from atc_rl.compare import load_evaluation
import numpy as np
sac=load_evaluation(PARENT/'eval-dev20');classical=load_evaluation(reference)
assert sac[2]==classical[2]
assert len(sac[2])==20
rng=np.random.default_rng(701);indices=rng.integers(0,20,size=(10000,20))
effects={}
for key,values in sac[3].items():
    delta=values.astype(float)-classical[3][key]
    effects[key]={'sac_minus_classical':float(delta.mean()),
        'paired_world_bootstrap_95_interval':np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist()}
result={'checks':checks,'worlds':20,'sac_csv_sha256':sac[0]['csv_sha256'],
    'classical_csv_sha256':classical[0]['csv_sha256'],'model_sha256':protocol['model_sha256'],
    'means':{name:{key:float(values.mean()) for key,values in data[3].items()} for name,data in [('sac',sac),('classical',classical)]},
    'system_effects':effects,'interpretation':protocol['interpretation'],
    'zero_reference_caveat':'Zero residual reproduces the controller reference; this is not a saved untrained neural policy comparison.',
    'training_seed_count':1,'unseen_scenarios_used':False}
with (PARENT/'comparison-classical-dev20.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps({'completed':True,'means':result['means']}),flush=True)