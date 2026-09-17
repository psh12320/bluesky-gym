"""Protect the paired input-information claim from support/configuration confounds."""
import json
import pytest
import scripts.analyze_ppo_static as analysis


@pytest.fixture
def pair(tmp_path, monkeypatch):
    paths = [tmp_path / arm for arm in ("features", "zeros")]
    for path, masked in zip(paths, (False, True)):
        (path / "train").mkdir(parents=True)
        config = {"seed":50800, "guidance":True, "filter":False, "static_filter":True,
                  "mask_conflict_features":masked, "actor_information":str(masked),
                  "run_dir":str(path / "train"), "workers":8}
        (path / "train/config.json").write_text(json.dumps(config))
        runtime = {"static_area_filter":True, "traffic_conflict_filter":False}
        (path / "train/runtime.json").write_text(json.dumps(runtime))
        for stage in ("initial","100k","300k","final"):
            folder = path / "evaluations" / stage
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps({"runtime":runtime}))
    monkeypatch.setattr(analysis, "policy_fingerprint", lambda path:"identical-tensors")
    return paths


def test_accepts_only_intended_input_mask_difference(pair):
    assert analysis.matched_pair(*pair) == "identical-tensors"


@pytest.mark.parametrize("key,value", [("seed",50900),("workers",1),("static_filter",False)])
def test_rejects_changed_training_recipe(pair, key, value):
    path = pair[1] / "train/config.json"
    config = json.loads(path.read_text());config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="more than the input mask"):
        analysis.matched_pair(*pair)


@pytest.mark.parametrize("stage", ["train","evaluations/final"])
def test_rejects_silent_traffic_filter_in_training_or_evaluation(pair, stage):
    path = pair[1] / stage / ("runtime.json" if stage == "train" else "summary.json")
    data = json.loads(path.read_text())
    runtime = data if stage == "train" else data["runtime"]
    runtime["traffic_conflict_filter"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="effective action filters"):
        analysis.matched_pair(*pair)


def test_rejects_unequal_initial_weights_or_capacity(pair, monkeypatch):
    monkeypatch.setattr(analysis, "policy_fingerprint", lambda path:str(path))
    with pytest.raises(ValueError, match="tensors or network capacity"):
        analysis.matched_pair(*pair)
