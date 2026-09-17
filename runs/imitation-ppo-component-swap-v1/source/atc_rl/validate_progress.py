"""Verify reward shaping changes learning feedback but not identical-action flights."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Choose a fresh validation directory')
    args.out.mkdir(parents=True)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    import numpy as np
    from atc_rl.world_pool import WorldPool
    from atc_rl.progress_reward import GAMMA
    records=[];decisions=0;padding=0;terminals=0;timeouts=0;nonzero_bonus=0
    started=time.perf_counter()
    for mode,world_count in [('goal',2),('circling',1)]:
        base=shaped=None
        try:
            base=WorldPool(1,args.out/mode/'base',guidance=mode=='goal',filter=mode=='goal')
            shaped=WorldPool(1,args.out/mode/'shaped',guidance=mode=='goal',filter=mode=='goal',progress_scale=100)
            base.seed(20260);shaped.seed(20260)
            original=base.reset();modified=shaped.reset()
            initial=np.array([info['initial_potential'] for info in shaped.reset_infos])
            scenario=base.reset_infos[0]['scenario_sha256']
            accumulated=np.zeros(10);steps=np.zeros(10,dtype=int);completed=0
            while completed<world_count:
                assert all(np.array_equal(original[key],modified[key]) for key in original)
                if mode=='goal':actions=base.goal_actions(original)
                else:actions=np.tile(np.array([1.,0.],dtype=np.float32),(10,1))
                original,base_reward,base_done,base_info=base.step(actions)
                modified,new_reward,new_done,new_info=shaped.step(actions)
                decisions+=1
                assert np.array_equal(base_done,new_done)
                assert all(np.array_equal(original[key],modified[key]) for key in original)
                for index,(plain,changed) in enumerate(zip(base_info,new_info)):
                    for key in ('inactive','aircraft_done','world_completed','TimeLimit.truncated'):
                        assert plain[key]==changed[key]
                    if plain['inactive']:
                        padding+=1
                        assert base_reward[index]==0 and new_reward[index]==0
                        continue
                    assert plain['native_reward']==changed['native_reward']
                    assert plain['shaping_reward']==0
                    bonus=changed['shaping_reward']
                    nonzero_bonus+=int(bonus!=0)
                    assert np.isclose(float(new_reward[index]),changed['native_reward']+bonus,rtol=1e-6,atol=1e-5)
                    accumulated[index]+=GAMMA**int(steps[index])*bonus
                    steps[index]+=1
                    if plain['aircraft_done']:
                        assert plain['metrics']==changed['metrics']
                        assert plain['aircraft_terminated']==changed['aircraft_terminated']
                        assert plain['aircraft_truncated']==changed['aircraft_truncated']
                        error=abs(accumulated[index]+initial[index])
                        assert error<1e-8,error
                        terminals+=int(plain['aircraft_terminated']);timeouts+=int(plain['aircraft_truncated'])
                        records.append({'mode':mode,'episode':completed,'scenario_sha256':scenario,'agent':f'KL00{index+1}',
                            'initial_potential':float(initial[index]),'discounted_shaping':float(accumulated[index]),
                            'telescoping_error':float(error),'learning_return':changed['learning_episode_return'],**plain['metrics']})
                if base_info[0]['world_completed']:
                    assert sum(r['mode']==mode and r['episode']==completed for r in records)==10
                    completed+=1
                    assert base.reset_infos[0]['scenario_sha256']==shaped.reset_infos[0]['scenario_sha256']
                    scenario=base.reset_infos[0]['scenario_sha256']
                    initial=np.array([info['initial_potential'] for info in shaped.reset_infos])
                    accumulated[:]=0;steps[:]=0
                    print(json.dumps({'mode':mode,'completed_worlds':completed}),flush=True)
        finally:
            if base is not None:base.close()
            if shaped is not None:shaped.close()
    assert terminals==20 and timeouts==10 and padding>0 and nonzero_bonus>0
    root=Path(__file__).resolve().parents[1]
    reference=json.loads((root/'tests/rl/fixtures/adapter-reference.json').read_text(encoding='utf-8'))
    baseline={(int(r['episode']),r['agent']):r for r in reference['aircraft']}
    for row in records:
        if row['mode']=='goal':
            expected=baseline[row['episode'],row['agent']]
            assert row['scenario_sha256']==expected['scenario_sha256']
            assert all(row[key]==float(expected[key]) for key in reference['metric_fields'])
    (args.out/'aircraft.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    result={'status':'passed','paired_worlds':3,'paired_aircraft':len(records),'identical_observations_metrics_terminals':True,
            'matches_original_adapter_fixture':True,'arrivals':terminals,'timeouts':timeouts,'padding_slots_checked':padding,
            'decision_pairs':decisions,'nonzero_shaping_steps':nonzero_bonus,
            'max_discounted_identity_error':max(r['telescoping_error'] for r in records),
            'wall_seconds':time.perf_counter()-started,
            'records_sha256':hashlib.sha256((args.out/'aircraft.json').read_bytes()).hexdigest()}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
