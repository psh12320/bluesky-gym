"""Audit the completed original-format full SA scoring run without changing its evidence."""
import hashlib,importlib.util,json,math,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('report_inputs',root/'runs/report-final-v1/collect_evidence.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
package=root/'output/candidates/decimal-heading-v1.zip'
assert module.sha(package)=='da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c'
with zipfile.ZipFile(package) as archive:
    selected=json.loads(archive.read('models/sa/deployment.json'))
result,rows=module.full_run('sa',False,selected,{})
protocol=module.read(out/'protocol.json')
execution={name:digest for name,digest in protocol['source_sha256'].items() if Path(name).suffix in {'.py','.toml','.slurm','.sh','.ps1','.lock','.yaml','.yml'}}
for name,digest in execution.items():assert module.sha(root/name)==digest,name
result.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),frozen_execution_files_unchanged=len(execution),
              all_1000_rows_retained=True,all_nine_summary_metrics_recomputed=True,
              model_identical_to_selected_sa_candidate=True,
              scope='Local original 1000-scenario seed-42 harness with original heading formatting. Not judge-verified; corrected deployment score remains separate.')
result['missed_arrivals']=[row for row in rows if row['waypoint_reached']==0]
result['sector_exit_records']=[row for row in rows if row['sector_exit_events']>0]
result['tails']={}
for key in ('flight_time','intrusion_time','time_in_restricted_area','time_outside_sector'):
    values=np.array([row[key] for row in rows]);positive=values[values>0]
    result['tails'][key]={'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),
                          'p99':float(np.quantile(values,.99)),'maximum':float(values.max()),
                          'positive_records':len(positive),'conditional_positive_mean':float(positive.mean()) if len(positive) else None}
def wilson(successes,n):
    z=1.959963984540054;p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [center-radius,center+radius]
result['arrival_wilson_ci95']=wilson(sum(row['waypoint_reached'] for row in rows),len(rows))
result['uncertainty_scope']='Pointwise scenario-sampling interval conditional on this trained model, assuming the procedural scenarios follow the same sampling distribution; excludes training variability.'
file=out/'completion-audit.json';assert not file.exists(),'Preserve the completed audit'
file.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'all_nine_summary_metrics_recomputed':True,'frozen_execution_files_unchanged':len(execution),
                  'missed_arrivals':len(result['missed_arrivals']),'arrival_wilson_ci95':result['arrival_wilson_ci95'],
                  'parser_diagnostic_lines':result['parser_diagnostic_lines'],'sector_exit_records':result['sector_exit_records'],
                  'tails':result['tails'],'completion_audit_sha256':module.sha(file)},indent=2))
