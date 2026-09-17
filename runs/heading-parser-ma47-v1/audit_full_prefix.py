"""Compare the preserved heading diagnostics with the completed uninterrupted run."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from atc.compare import load_evaluation
from atc.metrics import METRICS
base=Path(__file__).resolve().parent
out=base/'full-prefix-audit.json'
assert not out.exists()
full=ROOT/'runs/heldout-2027-ma-interval5-v1/learned'
metadata,records=load_evaluation(full)
assert (metadata['env'],metadata['seed'],metadata['episodes'])==('ma',2027,200)
assert metadata['inference_mode']=='per_aircraft'
expected={(r['episode'],r['agent']):r for r in records if r['episode']==46}
assert len(expected)==10
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results={}
for label,path in [('original',base/'diagnostic.json'),
                   ('corrected',ROOT/'runs/heading-parser-ma47-decimal-v1/diagnostic.json')]:
    diagnostic=json.loads(path.read_text())
    assert diagnostic['model_sha256']==metadata['model_sha256']
    assert diagnostic['geography_backend']=='bluesky.tools.geo._cgeo'
    actual={(r['episode'],r['agent']):r for r in diagnostic['final_records']}
    assert actual.keys()==expected.keys()
    per_aircraft={agent:{key:actual[(episode,agent)][key]-expected[(episode,agent)][key]
                         for key in METRICS} for episode,agent in expected}
    maximum={key:max(abs(row[key]) for row in per_aircraft.values()) for key in METRICS}
    results[label]={'diagnostic_sha256':sha(path),'signed_metric_changes_by_aircraft':per_aircraft,
                    'metric_max_absolute_difference':maximum,
                    'all_nine_metrics_exact':all(value==0 for value in maximum.values()),
                    'all_eight_physical_metrics_exact':all(maximum[key]==0 for key in METRICS[:-1]),
                    'heading_commands_checked':diagnostic['heading_commands_checked'],
                    'invalid_heading_commands':len(diagnostic['invalid_commands'])}
result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'scenario_index':46,
        'full_csv_sha256':sha(full.with_suffix('.csv')),'model_sha256':metadata['model_sha256'],
        'comparisons':results,'original_skipped_prefix_parity_verified':results['original']['all_nine_metrics_exact'],
        'scope':'Check the previously inspected case against the complete uninterrupted original-format evaluation. Corrected comparison is a bug-fix check, not fresh held-out evidence.'}
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({label:{k:v for k,v in value.items() if k!='signed_metric_changes_by_aircraft'} for label,value in results.items()},indent=2))
assert result['original_skipped_prefix_parity_verified'], 'Skipped-prefix diagnostic differs from the full rollout; inspect the saved audit'
