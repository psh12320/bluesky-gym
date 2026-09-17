"""Load a versioned model deployment with explicit heading transport."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key, value in {'SDL_VIDEODRIVER': 'dummy', 'PYGAME_HIDE_SUPPORT_PROMPT': '1',
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
    os.environ.setdefault(key, value)

from atc import submission as legacy

FORMAT = 'airtrafficcontrol.deployment.v1'
_CONFIGURATIONS = {}
_MODELS = {}
_MANIFESTS = {}


def validate_manifest(kind, manifest):
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT:
        raise ValueError('Unknown deployment format')
    transport = manifest.get('heading_transport', {})
    if (not isinstance(transport, dict) or type(transport.get('revision')) is not int or transport['revision'] != 1
            or transport.get('encoding') != 'plain_decimal'):
        raise ValueError('Unsupported heading transport')
    digest = manifest.get('model_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Deployment requires a SHA-256 model hash')
    if manifest.get('expected_geography_backend') not in {'bluesky.tools.geo._cgeo', 'bluesky.tools.geo._geo'}:
        raise ValueError('Deployment requires an explicit geography backend')
    configuration = manifest.get('environment_configuration')
    if not isinstance(configuration, dict):
        raise ValueError('Deployment requires an environment configuration')
    legacy._validate_configuration(kind, configuration)
    return configuration


def load_policy(kind, model_path=None):
    for registry in (_CONFIGURATIONS, _MODELS, _MANIFESTS):
        registry.pop(kind, None)
    if model_path is None:
        raise ValueError('A versioned deployment requires an explicit model')
    path = Path(model_path).resolve()
    if not path.is_file() and Path(str(path) + '.zip').is_file():
        path = Path(str(path) + '.zip')
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads((path.parent / 'deployment.json').read_text(encoding='utf-8'))
    configuration = validate_manifest(kind, manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['model_sha256']:
        raise ValueError('Model hash differs from deployment manifest')
    import torch
    from stable_baselines3 import PPO, SAC
    torch.set_num_threads(1)
    algorithm = configuration['algorithm']
    model = {'ppo': PPO, 'sac': SAC}[algorithm].load(
        path, device='cpu', **({'buffer_size': 1} if algorithm == 'sac' else {}))
    _CONFIGURATIONS[kind] = configuration
    _MODELS[kind] = model
    _MANIFESTS[kind] = manifest
    print(json.dumps({'deployment_model': str(path), 'model_sha256': digest,
                      'env': kind, 'heading_transport': manifest['heading_transport']}), flush=True)
    return lambda observation: model.predict(observation, deterministic=True)[0]


def make_env(kind, n_agents=10):
    if n_agents != 10:
        raise ValueError('The competition requires ten multi-agent aircraft')
    if kind not in _MODELS:
        raise ValueError('Load the versioned deployment before creating its environment')
    configuration = _CONFIGURATIONS[kind]
    env = legacy.build_environment(kind, configuration['recipe'],
                                   bool(configuration.get('guard_static', False)),
                                   bool(configuration.get('guard_traffic', False)))
    try:
        from atc.heading_transport import attach_decimal_heading
        from bluesky.tools import geo
        from stable_baselines3.common.utils import check_for_correct_spaces
        if geo.qdrdist.__module__ != _MANIFESTS[kind]['expected_geography_backend']:
            raise ValueError('Geography backend differs from the recorded deployment')
        attach_decimal_heading(env)
        if kind == 'sa':
            observation_space, action_space = env.observation_space, env.action_space
        else:
            agent = env.possible_agents[0]
            observation_space, action_space = env.observation_space(agent), env.action_space(agent)
        model = _MODELS[kind]
        check_for_correct_spaces(legacy._SpaceView(observation_space, action_space),
                                 model.observation_space, model.action_space)
        return env
    except BaseException:
        env.close()
        raise


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--episodes', type=int, default=1000)
    known, _ = parser.parse_known_args()
    if known.episodes != 1000:
        parser.error('This entry runs the full official protocol; use a development check for smaller runs')
    from scripts import evaluate_competition as harness
    if harness.SEED != 42 or harness.N_EPISODES != 1000:
        raise ValueError('Official protocol constants were modified')
    # Replace only the two integration hooks explicitly permitted by the harness.
    harness.make_env = make_env
    harness.load_policy = load_policy
    harness.main()


if __name__ == '__main__':
    main()
