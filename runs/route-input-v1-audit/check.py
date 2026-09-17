import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from gymnasium import spaces
from atc.envs import make_env
from atc.routes import RouteGuidance
from atc.route_input import configuration
from atc.submission import _validate_configuration
from atc.provenance import capture

root = Path.cwd()
source = root / 'runs/public-joint-v1-391k'
target = root / 'runs/public-route-input-v1-joint-391k'
assert not target.exists()
model_hash = hashlib.sha256((source / 'model.zip').read_bytes()).hexdigest()
assert model_hash == 'b4c7a671dbd0fae8146522fa8278f19f4119c859caf2e1b07fe0bb56eae45093'
audit = {}
for kind, recipe in [('ma', 'public_route_input'), ('sa', 'public_route_input_sa_transfer')]:
    env = make_env(kind, recipe, guard_traffic=True)
    obs, _ = env.reset(seed=2026)
    world = env.unwrapped
    checked = redirected = 0
    for _ in range(20):
        agents = list(env.agents) if kind == 'ma' else ['KL001']
        for agent in agents:
            raw = super(RouteGuidance, world)._get_obs(agent)
            guided = world._get_obs(agent)
            assert raw.keys() == guided.keys()
            for key in raw:
                if key not in ('cos_drift', 'sin_drift'):
                    np.testing.assert_array_equal(raw[key], guided[key])
            redirected += int(any(not np.array_equal(raw[k], guided[k]) for k in ('cos_drift', 'sin_drift')))
            checked += 1
        if kind == 'ma':
            obs, _, terminated, truncated, _ = env.step({agent: np.zeros(2) for agent in env.agents})
            shape = env.observation_space('KL001').shape
        else:
            obs, _, terminated, truncated, _ = env.step(np.zeros(2))
            shape = env.observation_space.shape
        assert shape == (124,)
    audit[kind] = dict(checked_observations=checked, redirected=redirected, shape=shape,
        original_fields_unchanged_except_bearing=True, controlled_aircraft=len(world.scenario.agents))
    if kind == 'sa':
        assert world.n_intruders == 10 and world.intruder_obs.n == 9
        audit[kind].update(actual_intruders=10, observed_intruders=9)
    env.close()
config = json.loads((source / 'config.json').read_text())
config.update(configuration())
config.update(recipe='public_route_input', training_recipe='public_weights', training_route_input=False,
    deployment_variant='Static-route bearing plus joint action correction; original weights, no additional training',
    source_configuration=str(source / 'config.json'), source_model_sha256=model_hash)
_validate_configuration('ma', config)
target.mkdir()
capture(target)
shutil.copy2(source / 'model.zip', target / 'model.zip')
(target / 'config.json').write_text(json.dumps(config, indent=2))
audit['model_sha256'] = model_hash
(root / 'runs/route-input-v1-audit/result.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
