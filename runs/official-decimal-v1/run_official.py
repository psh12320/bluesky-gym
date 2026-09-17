"""Run and audit the versioned deployment through the original full harness."""
import argparse
import ast
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
for key, value in {'SDL_VIDEODRIVER': 'dummy', 'PYGAME_HIDE_SUPPORT_PROMPT': '1',
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
    os.environ.setdefault(key, value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def outside_hooks(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name in {'make_env', 'load_policy'})]
    return ast.dump(tree, include_attributes=False)


def preflight(kind):
    base = ROOT / 'runs/decimal-transport-candidates-v1'
    package = read(base / 'package-audit.json')
    archive_path = Path(package['archive'])
    assert sha(archive_path) == package['sha256']
    checks = package['extracted_runtime_check']['checks']
    assert set(checks) == {'sa', 'ma'}
    for track, check in checks.items():
        assert check['all_metrics_match'] and check['extracted_source_used']
        assert sha(base / f'extracted-v1/checks/{track}.json') == check['check_sha256']
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('manifest.json'))
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
            if name.startswith('source/') and Path(name).suffix in {
                    '.py', '.toml', '.lock', '.yaml', '.yml', '.ps1', '.sh', '.slurm'}:
                assert sha(ROOT / name.removeprefix('source/')) == digest, name
    candidate = base / kind
    manifest = read(candidate / 'deployment.json')
    assert sha(candidate / 'model.zip') == manifest['model_sha256']
    assert manifest['heading_transport'] == {'revision': 1, 'encoding': 'plain_decimal'}
    from atc.deployment import validate_manifest
    validate_manifest(kind, manifest)
    harness_path = ROOT / 'scripts/evaluate_competition.py'
    original = subprocess.check_output([
        'git', '-c', f'safe.directory={ROOT.as_posix()}', 'show',
        'HEAD:scripts/evaluate_competition.py'], text=True, encoding='utf-8')
    assert outside_hooks(original) == outside_hooks(harness_path.read_text(encoding='utf-8'))
    from scripts import evaluate_competition as harness
    assert (harness.SEED, harness.N_EPISODES, harness.N_AGENTS_MA) == (42, 1000, 10)
    return candidate, manifest, package, harness_path


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=['sa', 'ma'], required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    candidate, manifest, package, harness_path = preflight(args.env)
    if args.preflight_only:
        print(json.dumps({'env': args.env, 'preflight_passed': True,
                          'model_sha256': manifest['model_sha256'],
                          'package_sha256': package['sha256'],
                          'simulation_started': False}, indent=2))
        return
    out = Path(__file__).resolve().parent / args.env
    out.mkdir()  # Never overwrite a previous full scoring run.
    from atc.provenance import capture
    source = capture(out / 'source')
    shutil.copytree(candidate, out / 'candidate')
    model_path = out / 'candidate/model.zip'
    deployment_path = out / 'candidate/deployment.json'
    command = ['atc.deployment', '--env', args.env, '--model', str(model_path),
               '--out', str(out / 'metrics.csv')]
    protocol = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'env': args.env, 'episodes': 1000, 'seed': 42,
        'seed_once_then_continue': True, 'official_protocol': True,
        'judge_verified': False, 'heading_transport': manifest['heading_transport'],
        'model_sha256': sha(model_path), 'deployment_sha256': sha(deployment_path),
        'package_sha256': package['sha256'], 'harness_sha256': sha(harness_path),
        'runner_sha256': sha(Path(__file__)),
        'source_sha256': source['source_sha256'],
        'outside_two_allowed_hooks_ast_matches_head': True,
        'additional_training_transitions': 0,
        'no_tuning_or_model_selection_on_scoring_outcomes': True,
        'interpretation': ('Corrected execution of the already selected model. '
                           'Retain prior original-format outcomes. A repeated seed-42 '
                           'run is a revision comparison, not another untouched test.'),
        'command': [sys.executable, '-u', '-m', *command],
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    state = {'status': 'running', 'pid': os.getpid(),
             'protocol_sha256': sha(out / 'protocol.json')}
    save = lambda: (out / 'state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    save()
    print(json.dumps({key: protocol[key] for key in
                      ('started_at_utc', 'env', 'seed', 'episodes', 'model_sha256')}, indent=2), flush=True)
    started = time.perf_counter()
    try:
        from atc.deployment import main as run_harness
        previous_argv = sys.argv
        try:
            sys.argv = command
            with (out / 'harness.log').open('x', encoding='utf-8') as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)), \
                     contextlib.redirect_stderr(Tee(sys.stderr, log)):
                    run_harness()
        finally:
            sys.argv = previous_argv
        from atc.metrics import METRICS, summarize
        with (out / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        agents = 1 if args.env == 'sa' else 10
        assert [int(row['episode_index']) for row in rows] == list(range(1000 * agents))
        records = [{'episode': i // agents, 'agent': f'completion-slot-{i % agents}',
                    **{key: float(row[key]) for key in METRICS}}
                   for i, row in enumerate(rows)]
        summary = summarize(records, 1000, agents)
        assert sha(model_path) == protocol['model_sha256']
        assert sha(deployment_path) == protocol['deployment_sha256']
        assert sha(harness_path) == protocol['harness_sha256']
        for name, digest in protocol['source_sha256'].items():
            if Path(name).suffix in {'.py', '.toml', '.lock', '.yaml', '.yml', '.ps1', '.sh', '.slurm'}:
                assert sha(ROOT / name) == digest, name
        log_text = (out / 'harness.log').read_text(encoding='utf-8')
        summary.update({
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': time.perf_counter() - started,
            'env': args.env, 'seed': 42, 'official_protocol': True, 'judge_verified': False,
            'model_sha256': protocol['model_sha256'],
            'deployment_sha256': protocol['deployment_sha256'],
            'heading_transport': manifest['heading_transport'],
            'csv_sha256': sha(out / 'metrics.csv'),
            'protocol_sha256': sha(out / 'protocol.json'),
            'all_metrics_recomputed': True, 'execution_sources_unchanged': True,
            'record_grouping': ('Original CSV indexes final records, not scenario IDs. '
                                'Consecutive groups contain exactly the fixed aircraft '
                                'count per scenario; completion slots are not aircraft identities.'),
            'parser_diagnostic_lines': [line for line in log_text.splitlines()
                                        if 'ArgumentError' in line or 'SyntaxError' in line],
        })
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        state.update(status='complete', completed_at_utc=summary['completed_at_utc'],
                     summary_sha256=sha(out / 'summary.json'))
        save()
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error),
                     stopped_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
