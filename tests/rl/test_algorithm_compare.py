import io
import zipfile
import pytest
import torch
from atc_rl.algorithm_compare import validate_recipes,actor_match


def config(algorithm):
    return dict(algorithm=algorithm,seed=49900,world_seeds=[49900,49910],device='cpu',reward_recipe='public_weights',workers=2,rollout_steps=256,batch_size=1024,epochs=10,
        guidance=False,filter=False,action_reference='goal_offset',reward_scale=.01,progress_scale=0.,
        initial_action_std=.05,neutral_action_mean=True,gamma=.9965,gae_lambda=.95,learning_rate=.0003,
        clip_range=.2,entropy_coefficient=.01,value_coefficient=.5,max_grad_norm=.5,target_kl=.03,
        actor_widths=[128,128],critic_widths=[256,256],decision_interval_seconds=5)


def test_matched_algorithms_allow_only_rollout_boundary_experience_difference():
    assert validate_recipes(config('ppo'),config('mappo'),{'live_transitions':100699},{'live_transitions':102000})==5120
    with pytest.raises(ValueError,match='experience'):validate_recipes(config('ppo'),config('mappo'),{'live_transitions':100699},{'live_transitions':300000})


@pytest.mark.parametrize('key,value',[('seed',50100),('world_seeds',[49900,49911]),('device','cuda'),('reward_recipe','modified'),('reward_scale',1.),('filter',True),('static_filter',True),('workers',8),('action_reference','direct'),('entropy_coefficient',0.),('conflict_features',True),('mask_conflict_features',True)])
def test_algorithm_effect_cannot_hide_changed_recipe(key,value):
    ppo=config('ppo');mappo=config('mappo');mappo[key]=value
    with pytest.raises(ValueError,match=key):validate_recipes(ppo,mappo,{'live_transitions':100699},{'live_transitions':102000})


def save(path,actor,critic):
    data=io.BytesIO();torch.save({'action_net.weight':torch.tensor([actor]),'value_net.weight':torch.ones(critic)},data)
    with zipfile.ZipFile(path,'w') as z:z.writestr('policy.pth',data.getvalue())


def test_actor_matching_records_larger_critic_but_rejects_changed_actor(tmp_path):
    ppo=tmp_path/'ppo.zip';mappo=tmp_path/'mappo.zip'
    save(ppo,1.,2);save(mappo,1.,5)
    result=actor_match(ppo,mappo)
    assert result['initial_actor_tensors_identical'] and result['ppo_critic_parameters']==2 and result['mappo_critic_parameters']==5
    save(mappo,2.,5)
    with pytest.raises(ValueError,match='initial actors'):actor_match(ppo,mappo)
