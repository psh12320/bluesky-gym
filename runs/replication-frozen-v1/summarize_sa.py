"""Report the completed SA track while the declared MA replications remain open."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from atc.compare import load_evaluation,paired_comparison
from atc.metrics import METRICS,summarize
base=Path(__file__).resolve().parent
out=base/'sa-completed-track.json'
assert not out.exists()
protocol=json.loads((base/'protocol.json').read_text())
primary=ROOT/'runs/heldout-2027-sa-reach250-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
refmeta,reference=load_evaluation(primary/'classical')
assert (refmeta['env'],refmeta['seed'],refmeta['episodes'])==('sa',2027,200)
seeds={}
for seed in [protocol['primary_training_seed'],*protocol['replication_training_seeds']]:
    prefix=primary/'learned' if seed==2900 else base/f'sa-seed{seed}/replication-200'
    metadata,records=load_evaluation(prefix)
    assert (metadata['env'],metadata['seed'],metadata['episodes'])==('sa',2027,200)
    summary=summarize(records,200,1)
    assert all(metadata[k]==v for k,v in summary.items())
    if seed==2900:
        expected=protocol['primary_models']['sa']['model_sha256']
        assert sha(ROOT/protocol['primary_models']['sa']['model'])==expected
        comparison_path=primary/'comparison.json'
    else:
        state=json.loads((base/f'sa-seed{seed}/state.json').read_text())
        assert state['status']=='complete'
        assert state['protocol_sha256']==sha(base/'protocol.json')
        assert state['training_audit']['exact_25000_step_callback']
        assert state['evaluations']['replication-200']['csv_sha256']==sha(prefix.with_suffix('.csv'))
        expected=state['deployed_model_sha256']
        assert sha(base/f'sa-seed{seed}/candidate/model.zip')==expected
        comparison_path=base/f'sa-seed{seed}/comparison.json'
    assert metadata['model_sha256']==expected
    comparison=paired_comparison(reference,records,200)
    assert comparison==json.loads(comparison_path.read_text())['metrics']
    values={key:summary['metrics'][key]['mean'] for key in METRICS}
    values.update({key:summary[key] for key in ('clean_completion_rate','all_aircraft_clean_completion_rate')})
    seeds[str(seed)]={'role':'primary' if seed==2900 else 'replication','model_sha256':expected,
                     'csv_sha256':sha(prefix.with_suffix('.csv')),'values':values,
                     'paired_classical_comparison':comparison,
                     'missed_aircraft':[r for r in records if not r['waypoint_reached']]}
stats={}
for metric in next(iter(seeds.values()))['values']:
    values=np.asarray([seed['values'][metric] for seed in seeds.values()])
    stats[metric]={'mean':float(values.mean()),'minimum':float(values.min()),'maximum':float(values.max()),
                   'sample_standard_deviation':float(values.std(ddof=1))}
result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'track':'sa','scenario_seed':2027,
        'protocol_sha256':sha(base/'protocol.json'),'heading_transport':'original formatter in every replication',
        'all_three_sa_training_seeds_included':True,'ma_replications_still_pending':True,
        'scope':'Completed SA track only on the same previously evaluated scenario stream. This is not the final combined replication summary and is not a new untouched test.',
        'interpretation':'Descriptive mean/range/sample standard deviation from three training seeds; no precise population-level training-uncertainty claim and no checkpoint reselection.',
        'training_seeds':seeds,'descriptive_seed_distribution':stats,
        'classical_csv_sha256':sha((primary/'classical').with_suffix('.csv'))}
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'per_seed':{seed:item['values'] for seed,item in seeds.items()},
                  'descriptive_seed_distribution':stats},indent=2))
