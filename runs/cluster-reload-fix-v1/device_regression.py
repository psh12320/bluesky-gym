from pathlib import Path
import importlib.util
import json
from unittest.mock import patch
import zipfile
import stable_baselines3.common.policies as sb3_policies

root = Path.cwd()
repo = root.parents[1]
work = repo / 'runs/cluster-reload-fix-v1'
with zipfile.ZipFile(repo / 'runs/onpolicy-cluster-v1.zip') as archive:
    original_source = archive.read('tests/rl/test_policy_and_buffer.py')
original_namespace = {'__name__': 'original_serialization_tests'}
exec(compile(original_source, 'original_serialization_tests', 'exec'), original_namespace)
spec = importlib.util.spec_from_file_location('fixed_serialization_tests', root / 'tests/rl/test_policy_and_buffer.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
actual_get_device = sb3_policies.get_device
calls = []

def require_explicit_device(device='auto'):
    calls.append(str(device))
    if str(device) == 'auto':
        raise AssertionError('Implicit auto device would choose CUDA on the cluster')
    return actual_get_device(device)

results = []
for centralized in (False, True):
    original_dir = work / f'regression-original-{centralized}'
    original_dir.mkdir(exist_ok=False)
    fixed_dir = work / f'regression-fixed-{centralized}'
    fixed_dir.mkdir(exist_ok=False)
    with patch.object(sb3_policies, 'get_device', require_explicit_device):
        try:
            original_namespace['test_policy_serialization_retains_information_boundary'](original_dir, centralized)
        except AssertionError as error:
            assert 'Implicit auto device' in str(error)
        else:
            raise AssertionError('Original test did not reproduce implicit device selection')
        fixed.test_policy_serialization_retains_information_boundary(fixed_dir, centralized)
    results.append({'centralized': centralized, 'original_implicit_device_rejected': True,
                    'fixed_explicit_device_passed': True})
record = {'results': results, 'device_requests': calls,
          'scope': 'CPU regression that rejects implicit auto selection; actual CUDA rerun required on cluster.'}
(work / 'device-regression.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))