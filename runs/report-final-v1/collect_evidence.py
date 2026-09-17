"""Validate report inputs; never publish a final evidence snapshot with missing runs."""
import argparse,csv,hashlib,json,math,statistics,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from atc.compare import load_evaluation,paired_comparison
from atc.metrics import METRICS,summarize
BASE=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda path:json.loads(path.read_text(encoding='utf-8'))


def close_summary(actual,expected):
    for metric in METRICS:
        for statistic in ('mean','std'):
            assert math.isclose(actual['metrics'][metric][statistic],expected['metrics'][metric][statistic],rel_tol=0,abs_tol=1e-9)
    for key in ('episodes','agent_episodes','clean_completion_rate','all_aircraft_clean_completion_rate'):
        assert actual[key]==expected[key],key


def archive_check(folder,expected):
    provenance=read(folder/'provenance.json')
    assert provenance['source_sha256']==expected
    with zipfile.ZipFile(folder/'source.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())==set(expected)
        for name,digest in expected.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
    return {'source_archive_sha256':sha(folder/'source.zip'),'packages_sha256':sha(folder/'packages.txt')}


def heldout(kind):
    folder=ROOT/('runs/heldout-2027-ma-interval5-v1' if kind=='ma' else 'runs/heldout-2027-sa-reach250-v1')
    protocol=read(folder/'protocol.json')
    audit=read(folder/'completion-audit.json')
    assert protocol['env']==kind and protocol['seed']==2027 and protocol['episodes']==200
    assert protocol['no_further_tuning_on_2027'] and audit['protocol_sha256']==sha(folder/'protocol.json')
    result={'protocol_sha256':sha(folder/'protocol.json'),'completion_audit_sha256':sha(folder/'completion-audit.json')}
    data={}
    for name in ('classical','learned'):
        meta,rows=load_evaluation(folder/name)
        assert (meta['env'],meta['seed'],meta['episodes'])==(kind,2027,200)
        expected=summarize(rows,200,10 if kind=='ma' else 1)
        close_summary(meta,expected)
        assert sha(folder/f'{name}.csv')==audit['evaluations'][name]['csv_sha256']
        if name=='learned':assert meta['model_sha256']==protocol['model_sha256']
        data[name]=rows
        result[name]={'summary':expected,'csv_sha256':sha(folder/f'{name}.csv')}
    comparison=paired_comparison(data['classical'],data['learned'],200)
    assert comparison==read(folder/'comparison.json')['metrics']
    result['paired_comparison']=comparison
    return result


def full_run(kind,corrected,selected,packaged_execution):
    folder=ROOT/f'runs/official-decimal-v1/{kind}' if corrected else ROOT/'runs/official-sa-reach250-v1'
    summary=read(folder/'summary.json')
    protocol=read(folder/'protocol.json')
    assert (protocol['env'],protocol['seed'],protocol['episodes'])==(kind,42,1000)
    assert protocol['seed_once_then_continue'] and protocol['outside_two_allowed_hooks_ast_matches_head']
    assert summary['official_protocol'] and summary['judge_verified'] is False
    assert (summary['env'],summary['seed'])==(kind,42)
    assert summary['model_sha256']==selected['model_sha256']
    assert summary['model_sha256']==protocol['model_sha256']==sha(folder/'candidate/model.zip')
    assert summary['protocol_sha256']==sha(folder/'protocol.json')
    assert summary['csv_sha256']==sha(folder/'metrics.csv')
    if corrected:
        state=read(folder/'state.json')
        assert state['status']=='complete' and state['summary_sha256']==sha(folder/'summary.json')
        manifest=read(folder/'candidate/deployment.json')
        assert manifest==selected
        for name,digest in packaged_execution.items():assert protocol['source_sha256'].get(name)==digest,name
        assert manifest['heading_transport']=={'revision':1,'encoding':'plain_decimal'}
        assert summary['deployment_sha256']==sha(folder/'candidate/deployment.json')
    else:
        assert protocol['configuration']==selected['environment_configuration']
    with (folder/'metrics.csv').open(newline='',encoding='utf-8') as stream:raw=list(csv.DictReader(stream))
    count=1 if kind=='sa' else 10
    assert [int(row['episode_index']) for row in raw]==list(range(1000*count))
    rows=[{'episode':i//count,'agent':f'completion-slot-{i%count}',**{key:float(row[key]) for key in METRICS}}
          for i,row in enumerate(raw)]
    assert all(0<=row['flight_time']<=3000 and all(row[k]>=0 for k in METRICS[2:-1]) for row in rows)
    expected=summarize(rows,1000,count)
    close_summary(summary,expected)
    source=archive_check(folder/'source',protocol['source_sha256'])
    log=(folder/'harness.log').read_text(encoding='utf-8')
    diagnostics=[line for line in log.splitlines() if 'ArgumentError' in line or 'SyntaxError' in line]
    result={'summary':expected,'model_sha256':summary['model_sha256'],'protocol_sha256':sha(folder/'protocol.json'),
            'csv_sha256':sha(folder/'metrics.csv'),'summary_sha256':sha(folder/'summary.json'),
            'source':source,'parser_diagnostic_lines':diagnostics,'corrected_transport':corrected,
            'wall_seconds':summary['wall_seconds']}
    return result,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true',help='Check available evidence and list missing inputs; do not write final evidence')
    args=parser.parse_args()
    needed={
        'corrected_sa_full_score':ROOT/'runs/official-decimal-v1/sa/summary.json',
        'corrected_ma_full_score':ROOT/'runs/official-decimal-v1/ma/summary.json',
        'original_sa_full_score':ROOT/'runs/official-sa-reach250-v1/summary.json',
        'all_declared_replications':ROOT/'runs/replication-frozen-v1/summary.json',
    }
    missing=[name for name,path in needed.items() if not path.is_file()]
    evidence={'scope':'Feasibility report inputs; course enrollment, team review and external submissions are separate decisions',
              'collected_at_utc':datetime.now(timezone.utc).isoformat(),
              'heldout':{kind:heldout(kind) for kind in ('sa','ma')}}
    regression=read(ROOT/'runs/decimal-transport-candidates-v1/regression-audit.json')
    assert (regression['tests'],regression['failures'],regression['errors'],regression['skipped'])==(108,0,0,0)
    assert regression['junit_xml_sha256']==sha(ROOT/'runs/decimal-transport-candidates-v1/regression-results.xml')
    package=ROOT/'output/candidates/decimal-heading-v1.zip'
    assert regression['candidate_archive_sha256']==sha(package)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('manifest.json'))
        for name,digest in manifest['files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
        selected={kind:json.loads(archive.read(f'models/{kind}/deployment.json')) for kind in ('sa','ma')}
        packaged_execution={name.removeprefix('source/'):digest for name,digest in manifest['files'].items()
                            if name.startswith('source/') and Path(name).suffix in {'.py','.toml','.slurm','.sh','.ps1','.lock','.yaml','.yml'}}
    evidence['candidate_archive_sha256']=sha(package)
    evidence['regression_audit_sha256']=sha(ROOT/'runs/decimal-transport-candidates-v1/regression-audit.json')
    evidence['corrected_previews']={}
    for kind in ('sa','ma'):
        folder=ROOT/'output/videos/decimal-heading-v1'
        audit=read(folder/f'{kind}-audit.json')
        assert audit['all_nine_metrics_exact'] and audit['full_stream_decodes']
        assert audit['model_sha256']==selected[kind]['model_sha256']
        assert isinstance(audit['visual_inspection'],dict)
        assert audit['mp4_sha256']==sha(folder/audit['file'])
        evidence['corrected_previews'][kind]={'audit_sha256':sha(folder/f'{kind}-audit.json'),
                                             'video_sha256':audit['mp4_sha256'],'model_sha256':audit['model_sha256']}
    completed_full={}
    for label,kind,corrected in [('original_sa_full_score','sa',False),('corrected_sa_full_score','sa',True),('corrected_ma_full_score','ma',True)]:
        if label not in missing:
            value,rows=full_run(kind,corrected,selected[kind],packaged_execution)
            evidence[label]=value
            completed_full[label]=rows
    if 'all_declared_replications' not in missing:
        result=read(needed['all_declared_replications'])
        assert result['all_declared_seeds_included'] and result['primary_candidates_unchanged']
        protocol=read(ROOT/'runs/replication-frozen-v1/protocol.json')
        assert result['protocol_sha256']==sha(ROOT/'runs/replication-frozen-v1/protocol.json')
        for kind in ('sa','ma'):
            assert set(result['tracks'][kind]['training_seeds'])=={'2900','2902','2904'}
            assert result['tracks'][kind]['training_seeds']['2900']['model_sha256']==selected[kind]['model_sha256']
            reference_folder=ROOT/('runs/heldout-2027-ma-interval5-v1' if kind=='ma' else 'runs/heldout-2027-sa-reach250-v1')
            _,reference=load_evaluation(reference_folder/'classical')
            primary=result['tracks'][kind]['training_seeds']['2900']
            audited=evidence['heldout'][kind]
            assert primary['role']=='primary'
            assert result['tracks'][kind]['classical_csv_sha256']==audited['classical']['csv_sha256']
            assert primary['csv_sha256']==audited['learned']['csv_sha256']
            assert primary['paired_classical_comparison']==audited['paired_comparison']
            for metric,value in primary['values'].items():
                actual=(audited['learned']['summary']['metrics'][metric]['mean']
                        if metric in METRICS else audited['learned']['summary'][metric])
                assert math.isclose(actual,value,rel_tol=0,abs_tol=1e-9)
            for seed in (2902,2904):
                folder=ROOT/f'runs/replication-frozen-v1/{kind}-seed{seed}'
                state=read(folder/'state.json')
                assert state['status']=='complete'
                meta,rows=load_evaluation(folder/'replication-200')
                assert (meta['env'],meta['seed'],meta['episodes'])==(kind,2027,200)
                close_summary(meta,summarize(rows,200,10 if kind=='ma' else 1))
                expected=result['tracks'][kind]['training_seeds'][str(seed)]
                assert expected['model_sha256']==sha(folder/'candidate/model.zip')==meta['model_sha256']
                assert expected['csv_sha256']==sha(folder/'replication-200.csv')
                for metric,value in expected['values'].items():
                    actual=meta['metrics'][metric]['mean'] if metric in METRICS else meta[metric]
                    assert math.isclose(actual,value,rel_tol=0,abs_tol=1e-9)
                assert expected['paired_classical_comparison']==paired_comparison(reference,rows,200)
            for metric,summary in result['tracks'][kind]['descriptive_seed_distribution'].items():
                values=[item['values'][metric] for item in result['tracks'][kind]['training_seeds'].values()]
                recomputed={'mean':statistics.mean(values),'minimum':min(values),'maximum':max(values),'sample_standard_deviation':statistics.stdev(values)}
                assert all(math.isclose(summary[key],value,rel_tol=0,abs_tol=1e-9) for key,value in recomputed.items())
        evidence['replications']=result
        evidence['replication_summary_sha256']=sha(needed['all_declared_replications'])
    if {'original_sa_full_score','corrected_sa_full_score'}<=completed_full.keys():
        old,new=completed_full['original_sa_full_score'],completed_full['corrected_sa_full_score']
        evidence['sa_transport_revision_comparison']={key:{
            'maximum_absolute_change':max(abs(a[key]-b[key]) for a,b in zip(old,new)),
            'changed_aircraft_records':sum(a[key]!=b[key] for a,b in zip(old,new))} for key in METRICS}
    evidence['platform_scope']='Local Windows CPU evaluations. Linux wheel availability was inspected, but Linux native runtime parity has not been verified.'
    status={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'ready_for_final_report':not missing,
            'missing_completed_inputs':missing,'heldout_tracks_audited':['sa','ma'],
            'passing_regression_tests':108,'corrected_video_tracks_verified':['sa','ma'],
            'process_status':'Not inferred from files; inspect live tool handles separately',
            'final_pdf_written':False}
    (BASE/'readiness.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))
    if args.status:return
    if missing:raise SystemExit('Final evidence requires all completed inputs listed above')
    target=BASE/'evidence.json'
    assert not target.exists(),'Preserve the completed evidence snapshot'
    target.write_text(json.dumps(evidence,indent=2),encoding='utf-8')


if __name__=='__main__':main()
