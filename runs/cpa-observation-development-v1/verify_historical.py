from pathlib import Path
import csv,hashlib,json,xml.etree.ElementTree as ET
root=Path.cwd();work=root/'runs/cpa-observation-development-v1'
references={0:root/'runs/goal-offset-v1/eval-initial-dev20',1:root/'runs/initial-support-ablation-v1/eval-g1-f0-dev20'}
metrics=('waypoint_reached','flight_time','intrusion_events','intrusion_time','restricted_area_events','time_in_restricted_area','sector_exit_events','time_outside_sector','total_reward')
records=[]
for guidance,directory in references.items():
    observed=json.loads((work/f'g{guidance}-cpa0.json').read_text())
    with (directory/'aircraft.csv').open(newline='',encoding='utf-8') as stream:
        reference=[row for row in csv.DictReader(stream) if int(row['episode'])<2]
    observed_by={(int(row['episode']),f'KL00{int(row["agent"])+1}'):row for row in observed['rows']}
    assert len(reference)==len(observed_by)==20
    for row in reference:
        actual=observed_by[int(row['episode']),row['agent']]
        for metric in metrics:assert float(row[metric])==actual[metric],(guidance,row['agent'],metric)
    scenario=json.loads((directory/'scenarios.json').read_text())[:2]
    assert observed['scenarios']==[row['sha256'] for row in scenario]
    records.append({'guidance':bool(guidance),'worlds':2,'exact_aircraft_metrics':180,'reference':str(directory),
                    'reference_csv_sha256':hashlib.sha256((directory/'aircraft.csv').read_bytes()).hexdigest()})
suite=ET.parse(work/'tests.xml').getroot().find('testsuite')
archives={'runs/cluster-ppo-baseline-v1.zip':'27bec8a905b56360225ab15db4ddc33d0f3235733cb0425d47c4263de00c3d8c',
          'runs/onpolicy-learning-source-v2.zip':'33579f122d1fffd6bf6f9aefdabad846111afe20cbaadb56e390ae2d5a849b8c'}
for path,digest in archives.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
result={'default_observation_historical_parity':records,'tests':dict(suite.attrib),
        'previous_bundles_unchanged':archives,'new_feature_integration':'Still running separately; see integration-complete.json when complete.',
        'test_runtime_note':'The test process emitted Windows WMI exception diagnostics while collecting system information, then passed all 128 tests with exit code zero.'}
(work/'historical-parity-and-tests.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'historical_reference_metrics_matched':360,'tests':suite.attrib['tests'],'failures':suite.attrib['failures'],'old_bundles_unchanged':True}))
