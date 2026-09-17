"""Train shared PPO first, with optional centralized-critic MAPPO comparisons."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

# Keep the entry point light; the upstream environment imports Torch indirectly.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm',choices=['ppo','mappo'],default='ppo')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--live-steps',type=int,default=1000000)
    parser.add_argument('--rollout-steps',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--learning-rate',type=float,default=3e-4,help='Constant PPO optimizer learning rate')
    parser.add_argument('--gae-lambda',type=float,default=.95,help='GAE residual weighting parameter in [0,1]; leaves gamma and rewards unchanged')
    parser.add_argument('--seed',type=int,default=50100)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--initial-action-std',type=float,default=.6065306597126334)
    parser.add_argument('--neutral-action-mean',action='store_true')
    parser.add_argument('--exploration',choices=['gaussian','gsde','categorical'],default='gaussian')
    parser.add_argument('--sde-weight-std',type=float,help='Initial gSDE noise-matrix weight std, not marginal action std')
    parser.add_argument('--sde-sample-freq',type=int,help='Resample gSDE noise every N five-second decisions')
    parser.add_argument('--action-reference',choices=['direct','goal_offset'],default='direct')
    parser.add_argument('--guidance',action='store_true')
    parser.add_argument('--filter',action='store_true',help='Existing static-area and aircraft-conflict filters')
    parser.add_argument('--static-filter',action='store_true',help='Existing static-area filter only; aircraft conflict filtering stays off unless --filter is set')
    parser.add_argument('--conflict-features',action='store_true',help='Append current-state closest-approach features to each local observation')
    parser.add_argument('--mask-conflict-features',action='store_true',help='Same-size zero-input control; requires --conflict-features')
    parser.add_argument('--traffic-position-scale',type=float,default=1.0,help='Positive multiplier for existing x_r/y_r inputs only')
    parser.add_argument('--progress-scale',type=float,default=0.0)
    parser.add_argument('--reward-scale',type=float,default=1.0,help='Positive scale applied only to learning rewards')
    parser.add_argument('--actor-reference',type=Path,help='Zero-experience PPO checkpoint for a matched MAPPO actor')
    parser.add_argument('--pretrained-model',type=Path,help='Verified final behavior-cloned policy; supervised experience is recorded separately from PPO')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint-live-steps',type=int,default=100000)
    parser.add_argument('--max-wall-seconds',type=float,default=0)
    args=parser.parse_args()
    from atc_rl.exploration import configuration as exploration_configuration, ppo_options, log_std_initialization, require_matching
    from atc_rl.traffic_scaling import position_scale
    from atc_rl.learning_rate import constant_learning_rate
    try:
        constant_learning_rate(vars(args))
        position_scale(vars(args))
        exploration_configuration(vars(args))
    except ValueError as error:
        parser.error(str(error))
    if args.mask_conflict_features and not args.conflict_features:parser.error('Masking requires --conflict-features')
    if args.actor_reference is not None and args.algorithm!='mappo':parser.error('Actor reference is reserved for the MAPPO comparison')
    if args.pretrained_model is not None and (args.actor_reference is not None or args.neutral_action_mean):
        parser.error('Pretraining cannot be combined with another actor initialization')
    import math
    if not math.isfinite(args.gae_lambda) or not 0<=args.gae_lambda<=1:parser.error('GAE lambda must be finite and in [0, 1]')
    if not math.isfinite(args.initial_action_std) or not 0<args.initial_action_std<=1:
        parser.error('Initial normalized action standard deviation must be in (0, 1]')
    if args.actor_reference is not None:
        reference_config=json.loads((args.actor_reference.parent/'config.json').read_text(encoding='utf-8-sig'))
        if constant_learning_rate(reference_config)!=args.learning_rate:parser.error('Learning rate must match the PPO reference')
        if reference_config.get('gae_lambda',.95)!=args.gae_lambda:parser.error('GAE lambda must match the PPO reference')
        try:
            require_matching(reference_config, vars(args), allow_unbuilt=True)
            if position_scale(reference_config)!=args.traffic_position_scale:
                raise ValueError('Traffic position scaling must match the PPO reference')
        except ValueError as error:
            parser.error(str(error))
        if ((args.exploration=='gaussian' and reference_config.get('initial_action_std',.6065306597126334)!=args.initial_action_std) or
                reference_config.get('neutral_action_mean',False)!=args.neutral_action_mean):
            parser.error('Initial actor settings must match the PPO reference configuration')
        if reference_config.get('action_reference','direct')!=args.action_reference:
            parser.error('Action reference must match the PPO comparison')
        if reference_config.get('conflict_features',False)!=args.conflict_features:
            parser.error('Conflict observation features must match the PPO reference')
        if reference_config.get('static_filter',False)!=args.static_filter:
            parser.error('Static filter must match the PPO reference')
        if reference_config.get('mask_conflict_features',False)!=args.mask_conflict_features:
            parser.error('Conflict feature masking must match the PPO reference')
    if not math.isfinite(args.reward_scale) or args.reward_scale<=0:parser.error('Learning reward scale must be finite and positive')
    if not math.isfinite(args.progress_scale) or args.progress_scale<0:parser.error('Progress scale must be finite and nonnegative')
    if min(args.workers,args.live_steps,args.rollout_steps,args.batch_size,args.epochs,args.checkpoint_live_steps)<1:
        parser.error('Training budgets and worker counts must be positive')
    if args.rollout_steps*args.workers*10<2:parser.error('PPO needs at least two rollout slots')
    seeds=[args.seed+10*i for i in range(args.workers)]
    if args.seed<0 or set(seeds)&{42,2026,2027,20260,20301,20302}:parser.error('Reserved evaluation seed or invalid seed')
    if args.max_wall_seconds<0:parser.error('Wall limit cannot be negative')
    directory=args.run_dir.resolve()
    if directory.exists() and any(directory.iterdir()):parser.error('Choose an empty run directory')
    directory.mkdir(parents=True,exist_ok=True)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    os.environ.update(SDL_VIDEODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1')
    import csv,hashlib,importlib.metadata,platform,zipfile
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from atc_rl.policy import AircraftPolicy
    from atc_rl.buffer import ActiveRolloutBuffer,RolloutAccounting
    from atc_rl.world_pool import WorldPool
    from atc_rl.rollout_diagnostics import FIELDS as DIAGNOSTIC_FIELDS,rollout_diagnostics
    from atc.metrics import METRICS
    torch.set_num_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    root=Path(__file__).resolve().parents[1]
    hashes={}
    with zipfile.ZipFile(directory/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        files=[root/'pyproject.toml']
        for package in ('atc','atc_rl','core','bluesky_gym','bluesky_zoo'):
            files.extend((root/package).rglob('*.py'))
        for path in sorted(files):
            data=path.read_bytes();name=path.relative_to(root).as_posix()
            hashes[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
    provenance={'source_sha256':hashes,'python':platform.python_version(),'platform':platform.platform(),
        'packages':{name:importlib.metadata.version(name) for name in ('torch','stable-baselines3','numpy','gymnasium','pettingzoo','bluesky-simulator')}}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(world_seeds=seeds,actor_information=('own aircraft observation with zeroed conflict-feature channels plus remaining-time fraction' if args.mask_conflict_features else 'own aircraft observation with current-state closest-approach features plus remaining-time fraction' if args.conflict_features else 'own standard aircraft observation plus remaining-time fraction'),
        critic_information='joint aircraft observations, alive flags and target identity' if args.algorithm=='mappo' else 'same local observation as actor',
        actor_widths=[128,128],critic_widths=[256,256],shared_actor=True,shared_critic=True,
        gamma=.996508469331006,gae_lambda=args.gae_lambda,learning_rate=args.learning_rate,clip_range=.2,entropy_coefficient=.01,
        value_coefficient=.5,max_grad_norm=.5,target_kl=.03,decision_interval_seconds=5,
        reward_recipe='public_weights',automatic_residual_controller=args.action_reference=='goal_offset',
        navigation_prior=('Goal/reference-bearing tracking before learned heading offset' if args.action_reference=='goal_offset' else 'None added to the action'),
        learning_reward_formula='reward_scale * (native_reward + potential_shaping)',
        progress_shaping={'scale':args.progress_scale,'distance_norm_km':150.0,'distance_cap_km':1500.0,
            'potential':'-scale * clipped_distance_to_goal_region / 150 * remaining_time_fraction',
            'formula':'gamma * next_potential - current_potential; terminal potential is zero',
            'original_scoring_info_unchanged':True},
        finite_horizon='Do not bootstrap beyond goal completion or the actual 3000-second competition deadline; bootstrap at rollout cuts.',
        action_distribution='SB3 diagonal Gaussian; clip sampled latent actions to [-1,1] before the recorded action mapping',
        action_mapping=('Direct heading/speed increments' if args.action_reference=='direct' else
            'Heading: observed reference bearing plus learned +/-90 degree offset, wrapped and clipped to +/-45 degree turn; speed increment unchanged'),
        active_mask='Policy, value, entropy and advantage-normalization samples exclude inactive aircraft slots',
        critic_fit_diagnostic='Live rollout samples only; collected value predictions versus GAE targets, before optimization',
        budget_unit='Live aircraft transitions; finish the current rollout, recording exact overshoot and padded slot count')
    config['traffic_position_normalization_m']=1000000.0/args.traffic_position_scale
    if args.exploration=='gsde':
        config.update(initial_action_std=None,
            initial_noise_interpretation='sde_weight_std is the noise-matrix weight std; marginal action std depends on local actor features',
            action_distribution='SB3 generalized state-dependent exploration; local actor features only; latent actions clipped to [-1,1]',
            noise_resampling_seconds=5*args.sde_sample_freq,
            state_dependent_noise_parameterization='SB3 full_std=True, learn_features=False, use_expln=False, squash_output=False')
    (directory/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    accounting=RolloutAccounting();environment=None;start=time.perf_counter();optimizer_steps=[0];checkpoints=[]
    def save(model,label):
        path=directory/(label+'.zip');temporary=directory/(label+'.pending.zip')
        model.save(temporary);temporary.replace(path)
        record={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'live_transitions':accounting.live_transitions,'counted_transitions':model.num_timesteps,
                'padded_transitions':accounting.padded_transitions,'optimizer_steps':optimizer_steps[0],
                'policy_epochs':model._n_updates,'rollouts':accounting.rollouts}
        if config.get('pretraining') is not None:
            record['pretraining']=config['pretraining']
            record['live_transition_scope']='Additional PPO experience after supervised initialization'
        checkpoints.append(record)
        manifest=directory/'checkpoints.pending.json'
        manifest.write_text(json.dumps(checkpoints,indent=2),encoding='utf-8')
        manifest.replace(directory/'checkpoints.json')
    try:
        environment=WorldPool(args.workers,directory/'workers',guidance=args.guidance,filter=args.filter,progress_scale=args.progress_scale,action_reference=args.action_reference,conflict_features=args.conflict_features,mask_conflict_features=args.mask_conflict_features,static_filter=args.static_filter,traffic_position_scale=args.traffic_position_scale)
        if args.exploration=='categorical':
            from atc_rl.maneuvers import ManeuverMapping, ManeuverVecEnv, initialize_route_choice
            schema=environment.actor_schema()
            mapping=ManeuverMapping.from_schema(schema)
            environment=ManeuverVecEnv(environment,mapping)
            config.update(maneuver_mapping=mapping.specification(),maneuver_observation_schema=schema,
                initial_action_std=None,route_choice_initial_probability=.9,
                navigation_prior='Initial category probabilities prefer observed route bearing and speed +1 with probability .9 each; all later logits are learned.',
                action_distribution='Independent categorical heading and speed choices; no continuous action clipping',
                action_mapping='Twenty headings: local observed bearing or a fixed five-degree turn grid; three speed increments.')
            if args.actor_reference is not None:
                require_matching(reference_config,config)
            (directory/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        (directory/'runtime.json').write_text(json.dumps(environment.runtime,indent=2),encoding='utf-8')
        if args.reward_scale!=1.0:
            from atc_rl.reward_scale import ScaledLearningRewards
            environment=ScaledLearningRewards(environment,args.reward_scale)
        model=PPO(AircraftPolicy,environment,seed=args.seed,device=args.device,n_steps=args.rollout_steps,
            batch_size=args.batch_size,n_epochs=args.epochs,learning_rate=args.learning_rate,gamma=config['gamma'],gae_lambda=args.gae_lambda,
            clip_range=.2,ent_coef=.01,vf_coef=.5,max_grad_norm=.5,target_kl=.03,
            rollout_buffer_class=ActiveRolloutBuffer,policy_kwargs={'centralized':args.algorithm=='mappo','log_std_init':log_std_initialization(config)},verbose=0,**ppo_options(config))
        if args.exploration=='categorical':
            initialize_route_choice(model,config['route_choice_initial_probability'])
        if args.neutral_action_mean:
            from atc_rl.initialization import neutral_action_mean
            neutral_action_mean(model)
        if args.actor_reference is not None:
            from atc_rl.actor_reference import match_initial_actor
            match_initial_actor(model,args.actor_reference,directory/'actor-reference')
        if args.pretrained_model is not None:
            from atc_rl.pretrained import initialize_from_pretrained
            config['pretraining']=initialize_from_pretrained(model,args.pretrained_model,config,directory)
            config['training_method']=args.algorithm+'_after_behavior_cloning'
            (directory/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        model.set_logger(configure(str(directory),['csv']))
        hook=model.policy.optimizer.register_step_post_hook(lambda *_:optimizer_steps.__setitem__(0,optimizer_steps[0]+1))
        initial_parameters={name:parameter.detach().clone() for name,parameter in model.policy.named_parameters()}
        save(model,'initial-model')
        next_checkpoint=args.checkpoint_live_steps
        fields=['rollout','live_transitions','padded_transitions','counted_transitions','worlds_completed','aircraft_completed',
                'optimizer_steps','policy_epochs','wall_seconds','live_transitions_per_second','training_return_last100','native_return_last100',
                'training_arrival_last100','policy_loss','value_loss','approx_kl','clip_fraction',*DIAGNOSTIC_FIELDS]
        from atc_rl.training_records import TrainingRecords
        with TrainingRecords(directory,args.reward_scale) as training_records, (directory/'learning.csv').open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            while accounting.live_transitions<args.live_steps:
                before=accounting.live_transitions
                model.learn(total_timesteps=args.rollout_steps*environment.num_envs,reset_num_timesteps=False,callback=accounting)
                assert accounting.live_transitions>before
                assert model.num_timesteps==accounting.live_transitions+accounting.padded_transitions
                elapsed=time.perf_counter()-start
                completed=accounting.completed_aircraft[-100:]
                row=dict(rollout=accounting.rollouts,live_transitions=accounting.live_transitions,
                    padded_transitions=accounting.padded_transitions,counted_transitions=model.num_timesteps,
                    worlds_completed=accounting.completed_worlds,aircraft_completed=len(accounting.completed_aircraft),
                    optimizer_steps=optimizer_steps[0],policy_epochs=model._n_updates,wall_seconds=elapsed,
                    live_transitions_per_second=accounting.live_transitions/max(elapsed,1e-9),
                    training_return_last100=float(np.mean(accounting.completed_learning_returns[-100:])) if completed else None,
                    native_return_last100=float(np.mean([r['total_reward'] for r in completed])) if completed else None,
                    training_arrival_last100=float(np.mean([r['waypoint_reached'] for r in completed])) if completed else None)
                latest=model.logger.name_to_value
                for column,key in [('policy_loss','policy_gradient_loss'),('value_loss','value_loss'),
                                   ('approx_kl','approx_kl'),('clip_fraction','clip_fraction')]:
                    value=latest.get('train/'+key)
                    row[column]=float(value) if value is not None else None
                row.update(rollout_diagnostics(model.rollout_buffer))
                training_records.append(accounting.completed_aircraft,accounting.completed_learning_returns)
                writer.writerow(row);stream.flush();os.fsync(stream.fileno())
                print(json.dumps(row),flush=True)
                if accounting.live_transitions>=next_checkpoint:
                    save(model,f'policy-live-{accounting.live_transitions}')
                    next_checkpoint=(accounting.live_transitions//args.checkpoint_live_steps+1)*args.checkpoint_live_steps
                if args.max_wall_seconds and elapsed>=args.max_wall_seconds:break
        save(model,'model');hook.remove()
        changed=[name for name,parameter in model.policy.named_parameters() if not torch.equal(initial_parameters[name],parameter)]
        assert any(name.startswith('mlp_extractor.policy_net') or name.startswith('action_net') for name in changed)
        assert any(name.startswith('mlp_extractor.value_net') or name.startswith('value_net') for name in changed)
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        for name,digest in hashes.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        summary={'status':'complete' if accounting.live_transitions>=args.live_steps else 'wall_limit_before_budget',
            'algorithm':args.algorithm,'live_transitions':accounting.live_transitions,'padded_transitions':accounting.padded_transitions,
            'counted_transitions':model.num_timesteps,'optimizer_steps':optimizer_steps[0],'policy_epochs':model._n_updates,
            'rollouts':accounting.rollouts,'worlds_completed':accounting.completed_worlds,
            'aircraft_completed':len(accounting.completed_aircraft),'wall_seconds':time.perf_counter()-start,
            'actor_and_critic_parameters_changed':True,'source_unchanged':True,'model_sha256':checkpoints[-1]['sha256'],
            'training_seed':args.seed,'performance_evaluation':'Not performed by this training command; use separate fixed-scenario evaluations.'}
        if config.get('pretraining') is not None:
            summary['pretraining']=config['pretraining']
            summary['training_method']='ppo_after_behavior_cloning'
        (directory/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        (directory/'failure.json').write_text(json.dumps({'error':repr(error),'live_transitions':accounting.live_transitions,
            'counted_or_padding_records_retained':True},indent=2),encoding='utf-8')
        raise
    finally:
        if environment is not None:environment.close()

if __name__=='__main__':main()
