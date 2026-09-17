import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC

from atc.extend_observations import SpaceOnlyEnv, extend_model, feature_indices


def test_added_features_preserve_initial_actor_critics_and_checkpoint_loading(tmp_path):
    torch.set_num_threads(1)
    original = spaces.Dict({"a": spaces.Box(-10, 10, (2,)), "z": spaces.Box(-10, 10, (1,))})
    extended = spaces.Dict({**original.spaces, "prediction": spaces.Box(-10, 10, (2,))})
    retained, added = feature_indices(original, extended)
    source = SAC("MlpPolicy", SpaceOnlyEnv(spaces.flatten_space(original), spaces.Box(-1, 1, (2,))),
                 policy_kwargs={"net_arch": [16, 16]}, buffer_size=16, seed=3, device="cpu")
    model = extend_model(source, spaces.flatten_space(extended), retained, added, buffer_size=128)
    rng = np.random.default_rng(17)
    observations = rng.normal(size=(32, 5)).astype(np.float32)
    baseline = observations[:, retained]
    for old, new in zip(baseline, observations):
        assert np.array_equal(source.predict(old, deterministic=True)[0], model.predict(new, deterministic=True)[0])
    with torch.no_grad():
        actions = torch.zeros((32, 2))
        for critic_name in ("critic", "critic_target"):
            old_q = getattr(source, critic_name)(torch.tensor(baseline), actions)
            new_q = getattr(model, critic_name)(torch.tensor(observations), actions)
            assert all(torch.equal(a, b) for a, b in zip(old_q, new_q))
    source_parameters = {name: value.clone() for name, value in source.policy.state_dict().items()}
    model.actor(torch.tensor(observations)).sum().backward()
    assert model.actor.features_extractor.projection.weight.grad.abs().sum() > 0
    model.actor.optimizer.step()
    assert all(torch.equal(value, source_parameters[name]) for name, value in source.policy.state_dict().items())
    path = tmp_path / "extended-model"
    model.save(path)
    restored = SAC.load(path, device="cpu")
    assert restored.replay_buffer.buffer_size == 128
    assert np.array_equal(model.predict(observations[0], deterministic=True)[0], restored.predict(observations[0], deterministic=True)[0])
    assert restored.actor.features_extractor.projection.weight.requires_grad


def test_control_projection_stays_frozen_after_save_and_load(tmp_path):
    torch.set_num_threads(1)
    source = SAC("MlpPolicy", SpaceOnlyEnv(spaces.Box(-10, 10, (3,)), spaces.Box(-1, 1, (2,))),
                 policy_kwargs={"net_arch": [16, 16]}, buffer_size=16, seed=3, device="cpu")
    control = extend_model(source, spaces.Box(-10, 10, (5,)), [0, 2, 4], [1, 3], enabled=False)
    path = tmp_path / "control"
    control.save(path)
    restored = SAC.load(path, device="cpu", buffer_size=1)
    assert not restored.actor.features_extractor.projection.weight.requires_grad
    assert not restored.critic.features_extractor.projection.weight.requires_grad
    assert torch.count_nonzero(restored.actor.features_extractor.projection.weight) == 0
