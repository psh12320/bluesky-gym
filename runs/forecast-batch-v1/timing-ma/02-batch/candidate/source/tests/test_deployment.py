import json

import pytest

from atc import deployment, submission


def manifest():
    return {'format': deployment.FORMAT,
            'heading_transport': {'revision': 1, 'encoding': 'plain_decimal'},
            'model_sha256': '0'*64,
            'expected_geography_backend': 'bluesky.tools.geo._cgeo',
            'environment_configuration': {'env': 'sa', 'algorithm': 'sac', 'recipe': 'baseline'}}


@pytest.mark.parametrize('field,value,message', [
    ('format', 'old', 'format'),
    ('heading_transport', None, 'transport'),
    ('heading_transport', {'revision': True, 'encoding': 'plain_decimal'}, 'transport'),
    ('heading_transport', {'revision': 2, 'encoding': 'plain_decimal'}, 'transport'),
    ('heading_transport', {'revision': 1, 'encoding': 'scientific'}, 'transport'),
    ('model_sha256', 'not-a-hash', 'hash'),
    ('expected_geography_backend', 'automatic', 'backend'),
])
def test_deployment_rejects_ambiguous_or_unsupported_metadata(field, value, message):
    data = manifest()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        deployment.validate_manifest('sa', data)


def test_deployment_rejects_wrong_track():
    with pytest.raises(ValueError, match='track'):
        deployment.validate_manifest('ma', manifest())


def test_changed_model_is_rejected_before_loading(tmp_path):
    model = tmp_path/'model.zip'
    model.write_bytes(b'different model')
    (tmp_path/'deployment.json').write_text(json.dumps(manifest()))
    with pytest.raises(ValueError, match='Model hash differs'):
        deployment.load_policy('sa', model)


def test_legacy_loader_cannot_silently_ignore_new_transport_manifest(tmp_path):
    model = tmp_path/'model.zip'
    model.write_bytes(b'model placeholder')
    (tmp_path/'deployment.json').write_text(json.dumps(manifest()))
    with pytest.raises(ValueError, match='config.json'):
        submission.load_policy('sa', model)


def test_environment_requires_loaded_model_and_fixed_agent_count(monkeypatch):
    monkeypatch.delitem(deployment._MODELS, 'ma', raising=False)
    with pytest.raises(ValueError, match='ten'):
        deployment.make_env('ma', n_agents=9)
    with pytest.raises(ValueError, match='Load'):
        deployment.make_env('ma')


def test_official_entry_rejects_a_changed_seed(monkeypatch):
    import sys
    from scripts import evaluate_competition as harness
    monkeypatch.setattr(sys, 'argv', ['deployment'])
    monkeypatch.setattr(harness, 'SEED', 2026)
    with pytest.raises(ValueError, match='constants'):
        deployment.main()
