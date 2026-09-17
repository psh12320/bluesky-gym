from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parents[2];parent=Path(__file__).resolve().parent
initial=root/'runs/initial-support-ablation-v1/eval-g1-f0-dev20'
classical=root/'runs/ppo-direct-pilot-v1/eval-classical-dev20'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(parent/'protocol.json');gate=protocol['advance_screen']
identity=read(parent/'initial-control-identity.json');assert identity['exact_full_policy_tensor_identity']
for directory in (initial,parent/'eval-50k-dev20',parent/'eval-final-dev20'):
    summary=read(directory/'summary.json')
    assert summary['episodes']==20 and sha(directory/'aircraft.csv')==summary['csv_sha256']
results={}
for label in ('50k','final'):
    evaluation=parent/f'eval-{label}-dev20';out=parent/f'comparison-{label}-dev20'
    subprocess.run([sys.executable,'-m','atc_rl.compare','--initial',str(initial),'--trained',str(evaluation),
                    '--classical',str(classical),'--out',str(out)],cwd=root,check=True)
    comparison=read(out/'comparison.json');means=comparison['means']['trained']
    effects=comparison['comparisons']['learning_trained_minus_initial']
    checks={'arrival_at_least':means['waypoint_reached']>=gate['arrival_at_least'],
            'arrival_change_at_least':effects['waypoint_reached']['mean_difference']>=gate['arrival_change_at_least'],
            'clean_completion_change_at_least':effects['clean_completion']['mean_difference']>=gate['clean_completion_change_at_least'],
            **{metric:effects[metric]['mean_difference']<=gate['safety_time_changes_at_most'] for metric in gate['metrics']}}
    results[label]={'checkpoint':comparison['checkpoint'],'means':means,'learning_effects':effects,
                   'registered_screen_checks':checks,'screen_passed':all(checks.values()),
                   'comparison_sha256':sha(out/'comparison.json')}
subprocess.run([sys.executable,'-m','atc_rl.curves','--training-run',str(parent/'train'),
    '--evaluation',str(initial),'--evaluation',str(parent/'eval-50k-dev20'),
    '--evaluation',str(parent/'eval-final-dev20'),'--classical',str(classical),'--out',str(parent/'curve-dev20')],cwd=root,check=True)
record={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'training_seed':49900,'training_seeds_completed_and_evaluated':1,
        'registered_protocol_sha256':sha(parent/'protocol.json'),'exact_initial_control_identity':identity,
        'checkpoints':results,'unseen_scenarios_used':False,
        'additional_seed_protocol':'runs/ppo-guidance-replication-v1/protocol.json',
        'limitations':['One training seed; additional seeds are committed separately, not selected after this outcome.',
                      'Twenty development worlds; scenario bootstrap is not training-seed uncertainty.',
                      'Only trained-minus-initial with identical support settings is attributed to learning.']}
with (parent/'validated-results.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps({'means':{k:v['means'] for k,v in results.items()},'screen_passed':{k:v['screen_passed'] for k,v in results.items()}}))
