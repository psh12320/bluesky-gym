"""Verify goal-offset execution against the fixed bearing tracker in BlueSky."""
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
    root=Path(__file__).resolve().parents[1]
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for package in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo') for p in (root/package).rglob('*.py')}
    protocol={'seed':20260,'worlds_per_support_configuration':1,'guidance_filter_configurations':[[False,False],[False,True],[True,False],[True,True]],
              'action_reference':'goal_offset','latent_heading':0.,'speed_action':1.,'source_sha256':hashes,
              'purpose':'Execution equivalence only; this is an untrained control, not a learning result.'}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[];decisions=0;padding=0;arrivals=0;timeouts=0;start=time.perf_counter()
    for guidance,filtered in protocol['guidance_filter_configurations']:
        label=f'g{int(guidance)}f{int(filtered)}';base=modified=None
        try:
            base=WorldPool(1,args.out/label/'direct',guidance=guidance,filter=filtered)
            modified=WorldPool(1,args.out/label/'goal-offset',guidance=guidance,filter=filtered,action_reference='goal_offset')
            base.seed(20260);modified.seed(20260)
            original=base.reset();observed=modified.reset()
            scenario=base.reset_infos[0]['scenario_sha256']
            assert scenario==modified.reset_infos[0]['scenario_sha256']
            while True:
                assert all(np.array_equal(original[k],observed[k]) for k in original)
                commands=base.goal_actions(original)
                latent=np.zeros((10,2),dtype=np.float32);latent[:,1]=1.
                original,rewards,dones,infos=base.step(commands)
                observed,new_rewards,new_dones,new_infos=modified.step(latent)
                decisions+=1
                assert np.array_equal(rewards,new_rewards) and np.array_equal(dones,new_dones)
                assert all(np.array_equal(original[k],observed[k]) for k in original)
                for index,(left,right) in enumerate(zip(infos,new_infos)):
                    for key in left:
                        if key=='terminal_observation':
                            assert all(np.array_equal(left[key][k],right[key][k]) for k in left[key])
                        else:assert left[key]==right[key],(key,left[key],right[key])
                    padding+=int(left['inactive'])
                    if left['aircraft_done']:
                        arrivals+=int(left['aircraft_terminated']);timeouts+=int(left['aircraft_truncated'])
                        records.append({'support':label,'guidance':guidance,'filter':filtered,'scenario_sha256':scenario,
                                        'agent':f'KL00{index+1}',**left['metrics']})
                if infos[0]['world_completed']:
                    assert sum(r['support']==label for r in records)==10
                    assert base.reset_infos==modified.reset_infos
                    break
            print(json.dumps({'support':label,'equivalent':True,'aircraft':10}),flush=True)
        finally:
            if base is not None:base.close()
            if modified is not None:modified.close()
    fixture=json.loads((root/'tests/rl/fixtures/adapter-reference.json').read_text(encoding='utf-8'))
    for record in (r for r in records if r['support']=='g1f1'):
        reference=next(r for r in fixture['aircraft'] if int(r['episode'])==0 and r['agent']==record['agent'])
        assert reference['scenario_sha256']==record['scenario_sha256']
        assert all(float(reference[key])==record[key] for key in fixture['metric_fields'])
    for name,digest in hashes.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    (args.out/'aircraft.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    result={'status':'passed','paired_worlds':4,'paired_aircraft':len(records),'decision_pairs':decisions,
            'padding_slots':padding,'arrivals':arrivals,'timeouts':timeouts,'all_observations_rewards_terminals_metrics_identical':True,
            'supported_guidance_filter_combinations':4,'guided_filtered_world_matches_preserved_fixture':True,
            'source_unchanged':True,'wall_seconds':time.perf_counter()-start,
            'records_sha256':hashlib.sha256((args.out/'aircraft.json').read_bytes()).hexdigest()}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
