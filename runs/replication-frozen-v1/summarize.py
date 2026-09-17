"""Summarize every declared replica; refuse a final table with missing seeds."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from atc.compare import load_evaluation,paired_comparison
from atc.metrics import METRICS,summarize
OUT=Path(__file__).resolve().parent
protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for primary in protocol['primary_models'].values():
    assert sha(ROOT/primary['model'])==primary['model_sha256']
missing=[]
for track in protocol['tracks']:
    for seed in protocol['replication_training_seeds']:
        p=OUT/f'{track}-seed{seed}'
        if not (p/'state.json').exists() or json.loads((p/'state.json').read_text()).get('status')!='complete':
            missing.append(f'{track}-seed{seed}')
if missing:
    raise SystemExit('A final replication table requires every declared case. Missing or incomplete: '+', '.join(missing))
assert not (OUT/'summary.json').exists(), 'Preserve the completed replication summary'
keys=METRICS[:-1]+('clean_completion_rate','all_aircraft_clean_completion_rate')
tracks={};table=[]
for track in protocol['tracks']:
    original=ROOT/('runs/heldout-2027-ma-interval5-v1' if track=='ma' else 'runs/heldout-2027-sa-reach250-v1')
    refmeta,reference=load_evaluation(original/'classical')
    assert (refmeta['env'],refmeta['seed'],refmeta['episodes'])==(track,2027,200)
    seeds={}
    for seed in [protocol['primary_training_seed'],*protocol['replication_training_seeds']]:
        prefix=original/'learned' if seed==2900 else OUT/f'{track}-seed{seed}/replication-200'
        metadata,records=load_evaluation(prefix)
        assert (metadata['env'],metadata['seed'],metadata['episodes'])==(track,2027,200)
        expected=summarize(records,200,10 if track=='ma' else 1)
        assert all(metadata[k]==v for k,v in expected.items())
        if seed==2900:
            assert metadata['model_sha256']==protocol['primary_models'][track]['model_sha256']
        else:
            case=OUT/f'{track}-seed{seed}'
            state=json.loads((case/'state.json').read_text())
            assert state['protocol_sha256']==sha(OUT/'protocol.json')
            assert state['training_audit']['exact_25000_step_callback']
            assert metadata['model_sha256']==state['deployed_model_sha256']==sha(case/'candidate/model.zip')
        values={k:(expected['metrics'][k]['mean'] if k in METRICS else expected[k]) for k in keys}
        seeds[str(seed)]={'role':'primary' if seed==2900 else 'replication','model_sha256':metadata['model_sha256'],'csv_sha256':sha(prefix.with_suffix('.csv')),'values':values,'paired_classical_comparison':paired_comparison(reference,records,200)}
        table.append({'track':track,'training_seed':seed,'role':seeds[str(seed)]['role'],**values})
    distribution={}
    for metric in keys:
        values=np.array([s['values'][metric] for s in seeds.values()])
        distribution[metric]={'mean':float(values.mean()),'minimum':float(values.min()),'maximum':float(values.max()),'sample_standard_deviation':float(values.std(ddof=1))}
    tracks[track]={'classical_csv_sha256':sha((original/'classical').with_suffix('.csv')),'training_seeds':seeds,'descriptive_seed_distribution':distribution}
result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),'all_declared_seeds_included':True,'primary_candidates_unchanged':True,'scenario_stream':2027,'scope':'Three fixed training seeds per method on the same previously evaluated scenario stream; no replica selection','uncertainty':'Scenario bootstrap intervals are pointwise and conditional on a trained model. Training-seed mean/range/sample standard deviation are descriptive from n=3, with no claimed precise population interval.','tracks':tracks}
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (OUT/'per-seed.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
print(json.dumps({'all_declared_seeds_included':True,'per_seed':table},indent=2))
