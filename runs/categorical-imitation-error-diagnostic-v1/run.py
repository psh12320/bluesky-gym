"""Compare frozen imitation actors on the same teacher-visited validation states."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'runs/categorical-imitation-source-v1-verify'
sys.path.insert(0,str(SOURCE))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from atc_rl.cluster import verify
    from atc_rl.demonstrations import load_demonstrations,require_disjoint
    from atc_rl.checkpoint_identity import verified_checkpoint
    from atc_rl.imitation import actor_mean,action_errors,categorical_labels
    from atc_rl.maneuvers import ManeuverMapping
    torch.set_num_threads(1)
    verify(SOURCE)
    parents={'continuous':ROOT/'runs/imitation-ppo-pilot-v1/bc',
             'categorical':ROOT/'runs/categorical-imitation-ppo-pilot-v1/bc'}
    protected={str(folder/name):sha(folder/name) for folder in parents.values()
               for name in ('model.zip','config.json','training_summary.json','checkpoints.json','provenance.json')}
    datasets={name:load_demonstrations(ROOT/'runs/imitation-data-v1'/name,name) for name in ('train','validation')}
    require_disjoint(datasets['train'],datasets['validation'])
    models={};summaries={};configs={};mapping=None
    for name,parent in parents.items():
        record=next(r for r in read(parent/'checkpoints.json') if r['file']=='model.zip')
        model=PPO.load(verified_checkpoint(parent,record),device='cpu')
        summary=read(parent/'training_summary.json');config=read(parent/'config.json')
        assert summary['epochs']==40 and summary['supervised_optimizer_steps']==2480
        assert summary['supervised_examples_seen']==2509760 and summary['reinforcement_learning_updates']==0
        assert model.num_timesteps==0 and model._n_updates==0 and not model.policy.optimizer.state
        assert not model.policy.centralized
        model.policy.set_training_mode(False)
        models[name]=model;summaries[name]=summary;configs[name]=config
        if name=='categorical':mapping=ManeuverMapping.from_specification(config['maneuver_mapping'])
    assert configs['continuous']['dataset_provenance']==configs['categorical']['dataset_provenance']
    result={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'training_updates_performed':0,
            'script_sha256':sha(__file__),'protected_artifacts':protected,'datasets':{},'results':{},
            'policy_configuration':{name:{key:config.get(key) for key in
                ('seed','exploration','epochs','batch_size','learning_rate','pretraining_initialization')}
                for name,config in configs.items()},
            'limitations':['One supervised training seed per recipe; initialization, output distribution and loss differ.',
                'The datasets and optimization budget match; this is not a pure causal distribution comparison.',
                'Teacher-visited states only. Command accuracy does not establish closed-loop safety or PPO improvement.',
                'Intervention includes the full teacher filter; it does not isolate traffic and static avoidance.',
                'World-balanced descriptive errors accompany transition averages; correlated transitions are not independent trials.',
                'This diagnostic does not change the registered fixed-epoch checkpoint or select an alternative epoch.']}

    def statistics(predicted,teacher,mask,world_index):
        if not mask.any():return {'rows':0}
        error=(predicted-teacher)[mask];heading=np.abs(error[:,0])*45
        worlds=world_index[mask]
        target=teacher[mask,0]*45;estimate=predicted[mask,0]*45
        return {'rows':int(mask.sum()),'worlds':int(len(np.unique(worlds))),
            'heading_mae_degrees':float(heading.mean()),'speed_action_mae':float(np.abs(error[:,1]).mean()),
            'action_mse':float(np.square(error.astype(np.float64)).mean()),
            'exact_heading_match_rate':float((error[:,0]==0).mean()),
            'exact_command_match_rate':float(np.all(error==0,axis=1).mean()),
            'teacher_absolute_turn_mean_degrees':float(np.abs(target).mean()),
            'predicted_absolute_turn_mean_degrees':float(np.abs(estimate).mean()),
            'same_turn_sign':float((np.sign(target)==np.sign(estimate)).mean()),
            'heading_error_above_10_degrees':float((heading>10).mean()),
            'equal_world_heading_mae_degrees':float(np.mean([heading[worlds==w].mean() for w in np.unique(worlds)])),
            'per_world_heading_mae_degrees':{str(w):float(heading[worlds==w].mean()) for w in np.unique(worlds)}}

    for role,data in datasets.items():
        arrays=data['arrays'];local=arrays['actor'];teacher=arrays['teacher_action'];n=len(local)
        labels=categorical_labels(data,mapping)
        predictions={'nominal_navigation':arrays['nominal_action']}
        categorical_choices=None
        for name,model in models.items():
            chunks=[];choices=[]
            with torch.no_grad():
                for start in range(0,n,4096):
                    batch=local[start:start+4096]
                    output=actor_mean(model.policy,torch.as_tensor(batch))
                    if name=='continuous':chunks.append(output.clamp(-1,1).numpy())
                    else:
                        decision=torch.stack((output[:,:20].argmax(1),output[:,20:].argmax(1)),dim=1).numpy()
                        choices.append(decision);chunks.append(mapping.decode(batch,decision))
            prediction=np.concatenate(chunks);predictions[name]=prediction
            if name=='categorical':categorical_choices=np.concatenate(choices)
            recomputed=action_errors(model.policy,data,mapping=mapping if name=='categorical' else None)
            for key,expected in summaries[name]['final_errors'][role].items():
                if key in recomputed:
                    np.testing.assert_allclose(recomputed[key],expected,rtol=0,atol=1e-12)
            sample={'actor':local[:32], 'critic':np.zeros((32,*model.observation_space['critic'].shape),dtype=np.float32)}
            ordinary=model.predict(sample,deterministic=True)[0]
            ordinary=mapping.decode(local[:32],ordinary) if name=='categorical' else ordinary
            np.testing.assert_allclose(prediction[:32],ordinary,rtol=0,atol=2e-6)
        turns=np.abs(teacher[:,0])*45
        masks={'all':np.ones(n,dtype=bool),'teacher_intervention':arrays['intervened'],
               'no_teacher_intervention':~arrays['intervened'],'turn_at_most_2_degrees':turns<=2,
               'turn_2_to_10_degrees':(turns>2)&(turns<=10),'turn_above_10_degrees':turns>10,
               'turn_above_20_degrees':turns>20}
        grouped={group:{name:statistics(p,teacher,mask,data['world_index']) for name,p in predictions.items()}
                 for group,mask in masks.items()}
        confusion=np.zeros((20,20),dtype=np.int64)
        np.add.at(confusion,(labels[:,0],categorical_choices[:,0]),1)
        grouped['categorical_label_confusion']={'rows':int(confusion.sum()),'teacher_by_predicted':confusion.tolist(),
            'label_accuracy':float((labels[:,0]==categorical_choices[:,0]).mean()),
            'exact_command_accuracy':float(np.all(predictions['categorical']==teacher,axis=1).mean()),
            'interpretation':'Some labels can express the same turn; command accuracy is the action-level measure.'}
        result['results'][role]=grouped
        result['datasets'][role]={'rows':n,'worlds':len(data['scenarios']),
            'protocol_sha256':data['completion']['protocol_sha256'],
            'world_manifest_sha256':data['completion']['world_manifest_sha256']}
    for name,digest in protected.items():assert sha(name)==digest,name
    verify(SOURCE)
    with (OUT/'diagnostic.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':'complete','validation':{group:result['results']['validation'][group]
                     for group in ('all','teacher_intervention','turn_above_10_degrees')}}))


if __name__=='__main__':main()
