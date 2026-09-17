"""Independently verify a completed replica's saved records, model and source archives."""
import argparse,hashlib,json,math,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from atc.metrics import METRICS,summarize
from atc.compare import load_evaluation,paired_comparison
base=Path(__file__).resolve().parent
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--track',choices=['sa','ma'],required=True);parser.add_argument('--seed',type=int,choices=[2902,2904],required=True);args=parser.parse_args()
protocol=read(base/'protocol.json');folder=base/f'{args.track}-seed{args.seed}'
target=folder/'completion-audit-independent.json';assert not target.exists(),'Preserve the completed audit'
state=read(folder/'state.json');assert state['status']=='complete'
assert state['protocol_sha256']==sha(base/'protocol.json')
assert all(stage['exit_code']==0 for stage in state['stages'])
model=folder/'candidate/model.zip';assert sha(model)==state['deployed_model_sha256']==state['training_audit']['checkpoint_sha256']
with zipfile.ZipFile(model) as archive:
    assert archive.testzip() is None
    data=json.loads(archive.read('data'))
    assert data['num_timesteps']==25000 and data['_n_updates']==7996 and data['seed']==args.seed
execution={name:digest for name,digest in protocol['primary_source_sha256'].items() if Path(name).suffix in {'.py','.toml','.slurm','.sh','.ps1','.lock','.yaml','.yml'}}
for name,digest in execution.items():assert sha(ROOT/name)==digest,name
for item in protocol['primary_models'].values():assert sha(ROOT/item['model'])==item['model_sha256']
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'track':args.track,'training_seed':args.seed,
        'protocol_sha256':sha(base/'protocol.json'),'completed_state_sha256':sha(folder/'state.json'),
        'model_sha256':sha(model),'fixed_callback_steps':25000,'fixed_callback_updates':7996,
        'frozen_execution_files_unchanged':len(execution),'primary_models_unchanged':True,'evaluations':{}}
for label,seed,episodes in [('development-20',2026,20),('replication-200',2027,200)]:
    meta,rows=load_evaluation(folder/label);assert (meta['env'],meta['seed'],meta['episodes'])==(args.track,seed,episodes)
    expected=summarize(rows,episodes,10 if args.track=='ma' else 1)
    assert all(meta[key]==value for key,value in expected.items())
    assert expected==state['evaluations'][label]['summary']
    assert sha(folder/f'{label}.csv')==state['evaluations'][label]['csv_sha256']
    provenance=folder/f'{label}_provenance';record=read(provenance/'provenance.json')
    with zipfile.ZipFile(provenance/'source.zip') as archive:
        assert archive.testzip() is None and set(archive.namelist())==set(record['source_sha256'])
        for name,digest in record['source_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
    for name,digest in execution.items():assert record['source_sha256'].get(name)==digest,name
    result['evaluations'][label]={'summary':expected,'csv_sha256':sha(folder/f'{label}.csv'),
             'source_archive_sha256':sha(provenance/'source.zip'),'packages_sha256':sha(provenance/'packages.txt'),
             'all_archived_source_hashes_verified':True}
    if label=='replication-200':replica=rows
reference_folder=ROOT/('runs/heldout-2027-ma-interval5-v1' if args.track=='ma' else 'runs/heldout-2027-sa-reach250-v1')
_,reference=load_evaluation(reference_folder/'classical')
comparison=paired_comparison(reference,replica,200)
assert comparison==read(folder/'comparison.json')['metrics']
result['paired_classical_comparison']=comparison
result['classical_csv_sha256']=sha(reference_folder/'classical.csv')
result['missed_arrivals']=[row for row in replica if not row['waypoint_reached']]
result['missing_training_seeds_are_not_omitted']='This audits one declared replica only; the final combined table still requires all declared cases'
target.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'track':args.track,'training_seed':args.seed,'all_saved_summaries_recomputed':True,
                  'source_archives_verified':2,'frozen_execution_files_unchanged':len(execution),
                  'paired_comparison_recomputed':True,'missed_arrivals':len(result['missed_arrivals']),
                  'missed_scenarios':len({r['episode'] for r in result['missed_arrivals']}),
                  'summary':result['evaluations']['replication-200']['summary'],
                  'audit_sha256':sha(target)},indent=2))
