"""Evaluate a PPO/MAPPO/fresh-SAC checkpoint or fixed benchmark on one seeded world stream."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--classical', action='store_true')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--heldout', action='store_true')
    args = parser.parse_args()
    if bool(args.model) == bool(args.classical):
        parser.error('Choose a checkpoint or the frozen classical benchmark')
    if args.episodes < 1 or args.seed < 0 or args.seed == 42:
        parser.error('Positive episode count and a non-official stream are required')
    if args.seed in (20301,20302) and not args.heldout:
        parser.error('Reserved unseen stream; use only after candidate selection with --heldout')
    directory = args.out.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error('Choose an empty output directory')
    directory.mkdir(parents=True, exist_ok=True)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from atc.metrics import METRICS, summarize
    from atc_rl.world_pool import WorldPool
    torch.set_num_threads(1)
    config = {'guidance':True,'filter':True,'algorithm':'classical'}
    model = None
    if args.model:
        args.model = args.model.resolve()
        config = json.loads((args.model.parent/'config.json').read_text(encoding='utf-8-sig'))
        if config['algorithm']=='sac':
            from atc_rl.sac_support import load_sac_for_inference
            model = load_sac_for_inference(args.model, config)
        else:
            model = PPO.load(args.model, device='cpu')
            from atc_rl.exploration import validate_model
            validate_model(model,config)
            if model.policy.centralized != (config['algorithm']=='mappo'):
                raise ValueError('Checkpoint critic architecture differs from configuration')
        manifest = json.loads((args.model.parent/'checkpoints.json').read_text(encoding='utf-8-sig'))
        entries = [item for item in manifest if item['file']==args.model.name]
        if len(entries)!=1 or entries[0]['sha256']!=hashlib.sha256(args.model.read_bytes()).hexdigest():
            raise ValueError('Checkpoint not found in its original integrity manifest')
    root = Path(__file__).resolve().parents[1]
    sources = {str(path.relative_to(root)).replace(chr(92),'/'):hashlib.sha256(path.read_bytes()).hexdigest()
               for package in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo')
               for path in (root/package).rglob('*.py')}
    metadata = {'seed':args.seed,'episodes':args.episodes,'algorithm':config['algorithm'],
                'guidance':config['guidance'],'filter':config['filter'],'static_filter':config.get('static_filter',False),'action_reference':config.get('action_reference','direct'),'conflict_features':config.get('conflict_features',False),'mask_conflict_features':config.get('mask_conflict_features',False),
                'training_progress_scale':config.get('progress_scale',0.0),'evaluation_progress_scale':0.0,
                'training_reward_scale':config.get('reward_scale',1.0),'evaluation_reward_scale':1.0,
                'initial_action_std':config.get('initial_action_std',.6065306597126334),
                'neutral_action_mean':config.get('neutral_action_mean',False),
                'checkpoint':entries[0] if model else None,'model_path':str(args.model) if model else None,
                'source_sha256':sources,'inference':'deterministic, per aircraft, critic context zeroed',
                'scenario_protocol':'seed once, then continue the generator stream'}
    from atc_rl.exploration import configuration as exploration_configuration
    metadata.update(exploration_configuration(config))
    if config.get('exploration')=='categorical':metadata['maneuver_mapping']=config['maneuver_mapping']
    if config['algorithm']=='sac':
        metadata.update(inference='deterministic, per aircraft, local inputs only',
                        action_distribution=config['action_distribution'])
    from atc_rl.traffic_scaling import position_scale
    metadata['traffic_position_scale']=position_scale(config)
    (directory/'protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    records=[];scenarios=[];live=0;padded=0;decisions=0;terminated=0;truncated=0
    start=time.perf_counter()
    environment=WorldPool(1,directory/'workers',guidance=config['guidance'],filter=config['filter'],
                          action_reference=config.get('action_reference','direct'),conflict_features=config.get('conflict_features',False),mask_conflict_features=config.get('mask_conflict_features',False),static_filter=config.get('static_filter',False),traffic_position_scale=position_scale(config))
    try:
        if config.get('exploration')=='categorical':
            from atc_rl.maneuvers import ManeuverMapping,ManeuverVecEnv
            mapping=ManeuverMapping.from_schema(environment.actor_schema())
            if mapping!=ManeuverMapping.from_specification(config['maneuver_mapping']):
                raise ValueError('Evaluation maneuver layout differs from the checkpoint')
            environment=ManeuverVecEnv(environment,mapping)
        expected_space=environment.observation_space['actor'] if config['algorithm']=='sac' else environment.observation_space
        if model and model.observation_space!=expected_space:
            raise ValueError('Checkpoint observation space differs from evaluation')
        environment.seed(args.seed)
        observation=environment.reset()
        scenario_hash=environment.reset_infos[0]['scenario_sha256']
        episode=0
        fields=['episode','scenario_sha256','agent',*METRICS]
        with (directory/'aircraft.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            while episode<args.episodes:
                if model:
                    # The deployed actor receives no other-aircraft critic context.
                    actions=[]
                    for index in range(10):
                        if config['algorithm']=='sac':
                            local=observation['actor'][index]
                        else:
                            local={key:observation[key][index] for key in observation}
                            local['critic']=np.zeros_like(local['critic'])
                        actions.append(model.predict(local,deterministic=True)[0])
                    actions=np.asarray(actions,dtype=np.int64 if config.get('exploration')=='categorical' else np.float32)
                else:
                    actions=environment.goal_actions(observation)
                observation,rewards,dones,infos=environment.step(actions)
                decisions+=1
                for index,info in enumerate(infos):
                    if info['inactive']:
                        padded+=1
                        assert dones[index] and rewards[index]==0
                    else:live+=1
                    if info['aircraft_done']:
                        assert not info['inactive'] and dones[index]
                        assert not info['TimeLimit.truncated']
                        terminated+=int(info['aircraft_terminated'])
                        truncated+=int(info['aircraft_truncated'])
                        row={'episode':episode,'scenario_sha256':scenario_hash,'agent':f'KL00{index+1}',**info['metrics']}
                        writer.writerow(row);records.append(row);stream.flush()
                if infos[0]['world_completed']:
                    completed=[r for r in records if r['episode']==episode]
                    assert len(completed)==10
                    scenarios.append({'episode':episode,'sha256':scenario_hash})
                    print(json.dumps({'episode':episode,'arrival_rate':float(np.mean([r['waypoint_reached'] for r in completed]))}),flush=True)
                    episode+=1
                    scenario_hash=environment.reset_infos[0]['scenario_sha256']
        result=summarize(records,args.episodes,10)
        result.update(algorithm=config['algorithm'],guidance=config['guidance'],filter=config['filter'],static_filter=config.get('static_filter',False),
                      action_reference=config.get('action_reference','direct'),conflict_features=config.get('conflict_features',False),mask_conflict_features=config.get('mask_conflict_features',False),seed=args.seed,live_transitions=live,padded_transitions=padded,world_decisions=decisions,
                      aircraft_terminated=terminated,aircraft_truncated=truncated,wall_seconds=time.perf_counter()-start,
                      actor_inference_uses_joint_context=False,runtime=environment.runtime,
                      csv_sha256=hashlib.sha256((directory/'aircraft.csv').read_bytes()).hexdigest())
        assert live+padded==10*decisions
        for name,digest in sources.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        (directory/'scenarios.json').write_text(json.dumps(scenarios,indent=2),encoding='utf-8')
        (directory/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
    finally:
        environment.close()


if __name__=='__main__':
    main()
