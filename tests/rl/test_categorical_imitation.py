import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from atc_rl.imitation import (ObservationOnlyEnvironment, actor_mean, actor_parameters, action_errors,
                             categorical_labels, fit_epoch, verify_teacher_dependencies)
from atc_rl.maneuvers import ManeuverMapping, initialize_route_choice
from atc_rl.policy import AircraftPolicy


def example():
    mapping=ManeuverMapping(3,0,1)
    x=np.tile(np.array([1.,0.,.5],np.float32),(16,1))
    nominal=mapping.decode(x,np.tile([0,2],(16,1)))
    teacher=mapping.decode(x,np.tile([19,0],(16,1)))
    data={"arrays":{"actor":x,"teacher_action":teacher,"nominal_action":nominal,"intervened":np.ones(16,dtype=bool)},
          "world_index":np.tile([0,1],8),"scenarios":[{},{}]}
    return mapping,data


def test_categorical_labels_reconstruct_commands_and_reject_inexact_or_wrong_bearings():
    mapping,data=example()
    labels=categorical_labels(data,mapping)
    np.testing.assert_array_equal(labels,np.tile([19,0],(16,1)))
    data["arrays"]["teacher_action"][0,0]=.123
    with pytest.raises(ValueError,match="exactly express"):categorical_labels(data,mapping)
    mapping,data=example()
    data["arrays"]["nominal_action"][0,0]=.2
    with pytest.raises(ValueError,match="nominal"):categorical_labels(data,mapping)


def test_categorical_supervised_fit_changes_only_actor_and_never_claims_rl_experience(tmp_path):
    torch.set_num_threads(1)
    mapping,data=example()
    env=ObservationOnlyEnvironment(np.array([-1,-1,0]),np.ones(3),mapping)
    with pytest.raises(RuntimeError):env.reset()
    with pytest.raises(RuntimeError):env.step([0,2])
    model=PPO(AircraftPolicy,env,seed=119,device="cpu",n_steps=2,batch_size=2,n_epochs=1,
              policy_kwargs={"actor_width":8,"critic_width":8})
    initialize_route_choice(model)
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    x=torch.as_tensor(data["arrays"]["actor"])
    y=torch.as_tensor(categorical_labels(data,mapping))
    weights=torch.ones(len(x))
    optimizer=torch.optim.Adam(actor_parameters(model.policy),lr=.03)
    losses=[]
    for _ in range(16):
        loss,steps=fit_epoch(model.policy,x,y,weights,optimizer,np.arange(len(x)),len(x),mapping)
        losses.append(loss)
        assert steps==1
    assert losses[-1]<losses[0]*.5
    changed=[k for k,v in model.policy.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith(("mlp_extractor.policy_net.","action_net.")) for k in changed)
    assert model.num_timesteps==0 and model._n_updates==0 and not model.policy.optimizer.state
    path=tmp_path/"model.zip";model.save(path)
    restored=PPO.load(path,device=model.device)
    assert restored.num_timesteps==0 and not restored.policy.optimizer.state
    errors=action_errors(restored.policy,data,mapping=mapping)
    assert errors["exact_command_match_rate"]==1. and errors["action_mse"]==0
    env.close()


def teacher_fixture(tmp_path):
    root=tmp_path/"current";original=tmp_path/"original"
    files={"atc/simulation.py":"FIXED=1\n", "atc_rl/demonstrations.py":"from atc_rl.world_worker import observation_recipe\n",
           "atc_rl/action_support.py":"FIXED=2\n",
           "atc_rl/world_worker.py":"from pathlib import Path\ndef observation_recipe(guidance=False,conflict_features=False):\n    return guidance, conflict_features\ndef worker():\n    pass\n"}
    for directory in (root,original):
        for name,text in files.items():
            path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    training={"protocol":{"source_sha256":{k:hashlib.sha256((original/k).read_bytes()).hexdigest() for k in files}}}
    path=root/"atc_rl/world_worker.py"
    path.write_text(files["atc_rl/world_worker.py"]+"def local_observation_schema(dictionary):\n    return dictionary\n",encoding="utf-8")
    return root,original,training


def test_teacher_compatibility_requires_verified_original_and_unchanged_used_recipe(tmp_path):
    root,original,training=teacher_fixture(tmp_path)
    with pytest.raises(ValueError,match="implementation differs"):verify_teacher_dependencies(training,root)
    result=verify_teacher_dependencies(training,root,original)
    assert len(result["exact_dependency_files"])==3
    assert result["unused_worker_definition_change"]["excluded_definitions"]==["worker","local_observation_schema"]


@pytest.mark.parametrize("change",["recipe","import_setup","collector","original"])
def test_teacher_compatibility_does_not_accept_changed_teacher_behavior(tmp_path,change):
    root,original,training=teacher_fixture(tmp_path)
    if change=="recipe":
        path=root/"atc_rl/world_worker.py";path.write_text(path.read_text().replace("return guidance, conflict_features","return False, False"))
    elif change=="import_setup":
        path=root/"atc_rl/world_worker.py";path.write_text(path.read_text()+"CHANGED=1\n")
    elif change=="collector":
        path=root/"atc_rl/demonstrations.py";path.write_text(path.read_text()+"CHANGED=1\n")
    else:
        path=original/"atc_rl/world_worker.py";path.write_text(path.read_text()+"CHANGED=1\n")
    with pytest.raises(ValueError):verify_teacher_dependencies(training,root,original)
