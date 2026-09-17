"""Audit completed frozen held-out runs without changing controller selection."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import math
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atc.compare import load_evaluation, paired_comparison
from atc.metrics import METRICS, SAFETY, summarize

out = ROOT / sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent
protocol = json.loads((out/'protocol.json').read_text())
assert protocol['seed'] == 2027 and protocol['episodes'] == 200
assert protocol['no_further_tuning_on_2027'] is True
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
model = ROOT / protocol['model']
assert sha(model) == protocol['model_sha256']
assert json.loads((model.parent/'config.json').read_text()) == protocol['configuration']
frozen = protocol['source_sha256']
executable = {n:v for n,v in frozen.items() if Path(n).suffix in {'.py','.toml','.slurm','.sh','.ps1','.yaml','.yml','.lock'}}
assert all(sha(ROOT/n) == v for n,v in executable.items())
archives = {}
for prefix in ('frozen-source', 'learned_provenance', 'classical_provenance'):
    directory = out/prefix
    provenance = json.loads((directory/'provenance.json').read_text())
    assert provenance['git_commit'] == protocol['source_commit']
    assert all(provenance['source_sha256'].get(n) == v for n,v in executable.items())
    with zipfile.ZipFile(directory/'source.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(provenance['source_sha256'])
        assert all(hashlib.sha256(archive.read(n)).hexdigest() == v for n,v in provenance['source_sha256'].items())
    archives[prefix] = {'source_archive_sha256':sha(directory/'source.zip'), 'files':len(provenance['source_sha256']), 'executable_files_match_freeze':len(executable), 'packages_sha256':sha(directory/'packages.txt')}
assert len({a['packages_sha256'] for a in archives.values()}) == 1

rows, result = {}, {}
clean = lambda r: bool(r['waypoint_reached'] and all(r[k] == 0 for k in SAFETY))
for identity in ('classical','learned'):
    meta, records = load_evaluation(out/identity)
    assert (meta['env'],meta['seed'],meta['episodes']) == (protocol['env'],2027,200)
    assert meta['official_protocol'] is False and meta['guard_traffic'] and meta['guard_static']
    assert meta['model_sha256'] == (protocol['model_sha256'] if identity == 'learned' else None)
    assert meta['recipe'] == (protocol['configuration']['recipe'] if identity == 'learned' else protocol['classical_reference']['recipe'])
    assert meta['decision_interval_seconds'] == 5 if protocol['env']=='ma' else meta.get('decision_interval_seconds',10) == 10
    if identity == 'classical':
        assert meta['inference_mode'] == 'goal' and meta['goal_speed_action'] == 1
    else:
        assert meta['inference_mode'] == 'per_aircraft'
    summary = summarize(records,200,10 if protocol['env']=='ma' else 1)
    assert all(meta[k] == v for k,v in summary.items())
    for metric in METRICS:
        assert math.isclose(math.fsum(r[metric] for r in records)/len(records),meta['metrics'][metric]['mean'],abs_tol=1e-9,rel_tol=0)
    assert all(all(r[k] >= 0 for k in METRICS[:-1]) and r['flight_time'] <= 3000 for r in records)
    rows[identity] = {(r['episode'],r['agent']):r for r in records}
    stats = {}
    for metric in ('flight_time','intrusion_time','time_in_restricted_area','time_outside_sector'):
        values = np.asarray([r[metric] for r in records])
        stats[metric] = {'mean':float(values.mean()), 'median':float(np.median(values)), 'p95':float(np.quantile(values,.95)), 'p99':float(np.quantile(values,.99)), 'maximum':float(values.max()), 'affected_count':int(np.count_nonzero(values)), 'mean_if_positive':float(values[values>0].mean()) if np.any(values>0) else None}
    result[identity] = {'records':len(records),'csv_sha256':sha(out/(identity+'.csv')),'json_sha256':sha(out/(identity+'.json')),'summary':summary,'tails':stats,'missed_arrivals':[r for r in records if not r['waypoint_reached']], 'restricted_cases':[r for r in records if r['time_in_restricted_area']>0], 'outside_cases':[r for r in records if r['time_outside_sector']>0]}
assert rows['learned'].keys() == rows['classical'].keys()
comparison = json.loads((out/'comparison.json').read_text())
assert comparison['metrics'] == paired_comparison(list(rows['classical'].values()),list(rows['learned'].values()),200)
old,new=rows['classical'],rows['learned']
fixes=[{'classical':old[k],'learned':new[k]} for k in old if not old[k]['waypoint_reached'] and new[k]['waypoint_reached']]
losses=[{'classical':old[k],'learned':new[k]} for k in old if old[k]['waypoint_reached'] and not new[k]['waypoint_reached']]
fixed_clean=[k for k in old if not clean(old[k]) and clean(new[k])]
lost_clean=[k for k in old if clean(old[k]) and not clean(new[k])]
audit={'audited_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Frozen candidates, first seed-2027 comparison; no checkpoint selection or tuning on held-out outcomes','protocol_sha256':sha(out/'protocol.json'),'model_sha256':sha(model),'configuration_unchanged':True,'archives':archives,'all_summaries_recomputed':True,'paired_comparison_recomputed':True,'matching_complete_scenario_and_aircraft_ids':True,'evaluations':result,'arrival_fixes':fixes,'arrival_losses':losses,'clean_fixes':fixed_clean,'clean_losses':lost_clean,'limitations':['One held-out scenario stream and one selected training seed; pointwise bootstrap intervals omit training-seed uncertainty','Matching IDs and unchanged seeded scenario-generation code support pairing; no scenario geometry log was recorded in these runs','Safety tails are descriptive and include all aircraft, including timeouts; no post-hoc exclusions']}
(out/'completion-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({'env':protocol['env'],'archives':archives,'tails':{k:v['tails'] for k,v in result.items()},'arrival_fixes':fixes,'arrival_losses':losses,'clean_fixed':len(fixed_clean),'clean_lost':len(lost_clean)},indent=2))
