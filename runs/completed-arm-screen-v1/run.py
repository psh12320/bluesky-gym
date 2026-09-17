"""Apply registered screens to complete arms without ranking incomplete cohorts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.analyze_rl_cohort import screen,paired_effect
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
TRIALS=[
 ('sac_seed51800','runs/sac-ppo-matched-pilot-v1/seed-51800/sac','runs/sac-ppo-matched-pilot-v1/protocol.json','sac-comparison-source-v1-verify'),
 ('ppo_lambda099_seed51800','runs/ppo-credit-pilot-v1/seed-51800/lambda099','runs/ppo-credit-pilot-v1/protocol.json','sac-comparison-source-v1-verify'),
 ('ppo_gsde_every12_seed52200','runs/ppo-gsde-replication-v1/seed-52200/every12','runs/ppo-gsde-replication-v1/protocol.json','onpolicy-gsde-source-v1-verify'),
]

def main():
    sys.path.insert(0,str(ROOT/'runs/sac-comparison-source-v1-verify'))
    from atc_rl.compare import load_evaluation
    from atc_rl.cluster import verify
    result={}
    indices=np.random.default_rng(701).integers(0,20,size=(10000,20))
    for name,path,protocol_name,source_name in TRIALS:
        parent=ROOT/path;plan=read(ROOT/protocol_name);source=ROOT/'runs'/source_name
        verify(source)
        summary=read(parent/'train/training_summary.json')
        if summary['status']!='complete':raise ValueError('Training incomplete')
        model=parent/'train/model.zip'
        if sha(model)!=summary['model_sha256']:raise ValueError('Model identity mismatch')
        expected_source={k:v for k,v in read(source/'cluster-manifest.json')['files'].items()
                         if k.endswith('.py') and k.split('/')[0] in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo')}
        loaded={stage:load_evaluation(parent/f'eval-{stage}-dev20') for stage in ('initial','final')}
        for stage,(s,p,worlds,values) in loaded.items():
            if s['episodes']!=20 or s['agent_episodes']!=200 or set(worlds)!=set(range(20)):
                raise ValueError('Incomplete evaluation')
            if p['source_sha256']!=expected_source or p['seed']!=20260:raise ValueError('Evaluation recipe differs')
            if p['evaluation_progress_scale']!=0 or p['evaluation_reward_scale']!=1:raise ValueError('Native rewards changed')
        if loaded['initial'][2]!=loaded['final'][2]:raise ValueError('Worlds are not paired')
        if loaded['initial'][1]['checkpoint']['live_transitions']!=0:raise ValueError('Initial policy trained')
        if loaded['final'][1]['checkpoint']['sha256']!=summary['model_sha256']:raise ValueError('Evaluated policy differs')
        original=read(parent/'comparison/comparison.json')
        means={stage:{k:float(v.mean()) for k,v in item[3].items()} for stage,item in loaded.items()}
        if means['initial']!=original['means']['initial'] or means['final']!=original['means']['trained']:
            raise ValueError('Recomputed means differ from preserved comparison')
        checks=screen(means['initial'],means['final'],plan['advance_screen'])
        effects={k:paired_effect(v,loaded['initial'][3][k],indices) for k,v in loaded['final'][3].items()}
        result[name]={'training':summary,'means':means,'own_initial_learning':effects,'screen':plan['advance_screen'],
                      'screen_checks':checks,'screen_passed':all(checks.values()),'promoted_to_candidate':False,
                      'protocol_sha256':sha(ROOT/protocol_name),'comparison_sha256':sha(parent/'comparison/comparison.json'),
                      'paired_cohort_incomplete':True,'csv_sha256':{stage:item[0]['csv_sha256'] for stage,item in loaded.items()}}
    record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'completed_arms':result,
            'purpose':'Record candidate eligibility of complete arms; not an algorithm ranking, complete-cohort aggregate or early stopping decision.',
            'all_registered_runs_continue':True,'unseen_scenarios_used':False,
            'limitations':['One training seed per arm, 100k live transitions and twenty reused development worlds.',
                           'Incomplete paired cohorts cannot support algorithm or parameter superiority claims.',
                           'Clean completion is a development metric, not an official scalar competition score.',
                           'Existing registered screens are applied unchanged; no model is promoted and no running job is stopped.']}
    with (OUT/'results.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps({name:{'screen_passed':r['screen_passed'],'failed_checks':[k for k,v in r['screen_checks'].items() if not v],
                           'clean_change':r['own_initial_learning']['clean_completion']['mean_difference']} for name,r in result.items()}))
if __name__=='__main__':main()
