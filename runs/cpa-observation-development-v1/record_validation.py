from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile,xml.etree.ElementTree as ET
root=Path.cwd();work=root/'runs/cpa-observation-development-v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
done=read(work/'integration-complete.json')
assert done['reloaded_feature_flag'] and done['smoke_training']['status']=='complete'
with zipfile.ZipFile(root/'runs/onpolicy-learning-source-v2.zip') as old,zipfile.ZipFile(root/'runs/onpolicy-cpa-source-v1.zip') as new:
    names=[name for name in old.namelist() if name.startswith(('atc/','core/','bluesky_gym/','bluesky_zoo/')) and name.endswith('.py')]
    assert all(old.read(name)==new.read(name) for name in names)
manifest=read(root/'runs/onpolicy-cpa-source-v1-verify/cluster-manifest.json')
for name,digest in manifest['files'].items():
    if name.endswith('.py') or name=='jobs/train_onpolicy.slurm':
        assert sha(root/name)==digest,name
tests=ET.parse(work/'tests.xml').getroot().find('testsuite').attrib
assert int(tests['tests'])==128 and tests['failures']==tests['errors']=='0'
validation={'checked_at_utc':datetime.now(timezone.utc).isoformat(),
 'unit_tests':{'passed':128,'failed':0,'errors':0,'seconds':float(tests['time'])},
 'fixed_action_parity':done['parity'],'historical_reference_metrics_matched':360,
 'controller_and_simulator_source_files_unchanged':len(names),
 'smoke_live_transitions':done['smoke_training']['live_transitions'],
 'smoke_optimizer_steps':done['smoke_training']['optimizer_steps'],
 'smoke_model_sha256':done['smoke_training']['model_sha256'],
 'model_reload_passed':True,'feature_performance_evidence':False,
 'new_bundle':read(root/'runs/onpolicy-cpa-source-v1.json'),
 'cluster_baseline_bundle_unchanged':sha(root/'runs/cluster-ppo-baseline-v1.zip')=='27bec8a905b56360225ab15db4ddc33d0f3235733cb0425d47c4263de00c3d8c'}
(work/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
progress={}
for group in ('g0-f1','g1-f1'):
    import csv
    path=root/f'runs/ppo-filter-ablation-v1/{group}/seed-49900/train/learning.csv'
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    last=rows[-1]
    progress[group]={'seed':49900,'last_recorded_live_transitions':int(last['live_transitions']),
                     'training_wall_seconds':float(last['wall_seconds']),'coordinator_session':33654 if group=='g0-f1' else 28101}
status={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'previous_goal_turn_classification':'progress',
        'this_goal_turn_classification':'progress','full_goal_complete':False,'competitive_learned_performance_demonstrated':False,
        'guided_three_seed_analysis_complete':True,'guided_clean_completion_initial':.26,
        'guided_clean_completion_final_by_seed':{'49900':.235,'49920':.245,'49940':.25},
        'guided_all_three_registered_screens_failed':True,'support_control_analysis_complete':True,
        'filter_pilot_training':progress,'additional_filter_pilot_seeds_per_group':[49920,49940],
        'intermediate_learning_curve_evaluations_session':34290,'optional_cpa_features':validation,
        'cluster_baseline_training_submission':'Awaiting user job IDs; question pending, not resubmitted.',
        'cluster_bootstrap_job':851402,'cluster_bootstrap_status':'COMPLETED','unseen_scenarios_used':False,
        'remaining_work':['Finish filtered PPO cohorts and conditional learning-contribution analysis.',
                          'Finish all three guided-PPO physical learning curves.',
                          'Run the selected university baseline and evaluate every seed.',
                          'Train a controlled feature-ablation pilot if justified; integration success is not performance evidence.',
                          'Establish strong learned performance and evaluate reserved unseen scenarios.']}
path=root/'runs/rl-priority-status-20260916-v17.json'
with path.open('x',encoding='utf-8') as stream:json.dump(status,stream,indent=2)
print(json.dumps({'validation_passed':True,'unchanged_controller_simulator_files':len(names),
                  'filter_pilot_progress':progress,'goal_complete':False}))
