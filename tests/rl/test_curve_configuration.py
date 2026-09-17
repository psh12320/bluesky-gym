import pytest
from atc_rl.curves import validate_configuration


def settings():
    config=dict(algorithm='ppo',seed=49900,workers=2,guidance=False,filter=False,
                initial_action_std=.05,neutral_action_mean=True,action_reference='goal_offset',
                reward_scale=.01,progress_scale=0.)
    protocol={k:config[k] for k in ('algorithm','guidance','filter','action_reference')}
    return config,protocol


def test_untrained_identical_policy_can_reuse_native_evaluation_across_reward_scales():
    config,protocol=settings();source={**config,'reward_scale':1.}
    validate_configuration(source,config,protocol,0)
    with pytest.raises(ValueError,match='reward_scale'):validate_configuration(source,config,protocol,100000)


@pytest.mark.parametrize('change',['guidance','filter','static_filter','conflict_features','mask_conflict_features','action_reference','evaluation_reward_scale','evaluation_progress_scale'])
def test_curve_rejects_changed_deployment_or_scoring_even_for_untrained_policy(change):
    config,protocol=settings()
    if change in ('guidance','filter','static_filter','conflict_features','mask_conflict_features'):protocol[change]=True
    elif change=='action_reference':protocol[change]='direct'
    elif change=='evaluation_reward_scale':protocol[change]=.01
    else:protocol[change]=100.
    with pytest.raises(ValueError):validate_configuration(config,config,protocol,0)


def test_curve_requires_same_learning_rate_after_initialization():
    config, protocol = settings()
    config["learning_rate"] = 3e-5
    source = {**config, "learning_rate": 3e-4}
    validate_configuration(source, config, protocol, 0)
    with pytest.raises(ValueError, match="learning_rate"):
        validate_configuration(source, config, protocol, 100000)
