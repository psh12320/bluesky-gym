from dataclasses import asdict
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from gymnasium import spaces
from atc.conflicts import ConflictPrediction, FEATURE_NAMES
from atc.recipes import RECIPES
from atc_rl.world_worker import observation_recipe

@pytest.mark.parametrize("guidance", [False, True])
def test_predictive_observations_change_no_reward_action_or_guidance_setting(guidance):
    original=observation_recipe(guidance,False)
    augmented=observation_recipe(guidance,True)
    before,after=asdict(original),asdict(augmented)
    assert {key for key in before if before[key]!=after[key]}=={"conflict_features"}
    reference=RECIPES["public_route_choice_interval5" if guidance else "public_weights"]
    assert original.reward_kwargs()==reference.reward_kwargs()
    assert original.decision_interval_seconds==5
    assert not reference.conflict_features

class ToyWorld:
    def __init__(self):
        self.intruder_obs=SimpleNamespace(n=3,pos_norm=10000.,spd_norm=100.)
        self.intrusion_distance=5
        self.observation_spaces={"own":spaces.Dict({name:spaces.Box(-np.inf,np.inf,(3,),dtype=np.float64)
                                                   for name in ("x_r","y_r","vx_r","vy_r")})}
    def _get_obs(self,ac_id):
        return {"x_r":np.array([3.,0.,0.]),"y_r":np.zeros(3),
                "vx_r":np.array([-3.,0.,0.]),"vy_r":np.zeros(3)}

class PredictiveToy(ConflictPrediction,ToyWorld):
    pass

def test_features_use_relative_vector_units_and_preserve_original_slots(monkeypatch):
    monkeypatch.setitem(sys.modules,"bluesky",SimpleNamespace(traf=SimpleNamespace(ntraf=2)))
    base=ToyWorld();env=PredictiveToy()
    original=base._get_obs("own");observed=env._get_obs("own")
    for name,value in original.items():np.testing.assert_array_equal(observed[name],value)
    assert observed["traffic_tcpa"][0]==pytest.approx(100/180)
    assert observed["traffic_entry_time"][0]==pytest.approx((30000-9260)/300/180)
    assert observed["traffic_dcpa"][0]==0
    assert observed["traffic_predicted_conflict"][0]==1
    for name in FEATURE_NAMES:
        assert np.all(observed[name][1:]==0)
        assert env.observation_spaces["own"][name].contains(observed[name])
    assert set(observed)==set(original)|set(FEATURE_NAMES)

def test_legacy_matrix_cannot_silently_accept_a_new_observation_experiment(tmp_path):
    import json
    from atc_rl.matrix_evaluate import validate_training
    folder=tmp_path/"runs/row";folder.mkdir(parents=True)
    (folder/"config.json").write_text(json.dumps({"conflict_features":True}))
    with pytest.raises(ValueError,match="conflict_features"):
        validate_training(tmp_path,{},{"run_dir":"runs/row"})
