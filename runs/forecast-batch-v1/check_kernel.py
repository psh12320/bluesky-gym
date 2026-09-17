"""Check and time batched unchanged-motion forecasts against the frozen scalar calls."""
import hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import numpy as np
import atc.traffic_projection as projection
out=Path(__file__).resolve().parent
source=Path(projection.__file__)
assert hashlib.sha256(source.read_bytes()).hexdigest()=='bdbbb4b4519a9808f5545cd816f77348afa9ad332bec83d15e760c13e44f1765'
protocol={'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'synthetic_rng_seed':8032,'fleet_sizes':[1,2,3,5,9,10,11],'batches_per_size':150,
          'test_scope':'Pure deterministic forecast kernel with zero commanded turns and unchanged target speeds; no simulator scenarios, training or policy selection',
          'requirement':'Exact coordinates and byte equality to independent scalar calls before any controller integration',
          'timing_scope':'Interleaved local kernel timings under concurrent work; no end-to-end or GPU speedup claim'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
state={'status':'running'}
def save():(out/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
def scalar(position,heading,speed):
    return np.stack([projection.predict_commands(p,float(h),float(v),[0.0],[float(v)])[0]
                     for p,h,v in zip(position,heading,speed)])
def batched(position,heading,speed):
    return projection.predict_commands(position,heading,speed,np.zeros(len(speed)),speed)
save();rng=np.random.default_rng(8032);failures=[];tested=0;aircraft=0;byte_failures=0;max_error=0.0
for size in protocol['fleet_sizes']:
    for case in range(protocol['batches_per_size']):
        position=rng.uniform(-500,500,(size,2));heading=rng.uniform(-720,1080,size);speed=rng.uniform(1,500,size)
        if case<6:
            edge=[0.0,-0.0,360.0,np.nextafter(360.0,0.0),1e-14,359.99999999999994][case]
            heading[:]=edge
            if case==0:position[:]=0;speed[:]=0
        expected=scalar(position,heading,speed);actual=batched(position,heading,speed)
        assert expected.shape==actual.shape==(size,19,2)
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        error=float(np.max(np.abs(expected-actual)));max_error=max(max_error,error)
        same_bytes=expected.tobytes()==actual.tobytes();byte_failures+=not same_bytes
        if not np.array_equal(expected,actual) or not same_bytes:
            if len(failures)<10:failures.append({'size':size,'case':case,'maximum_absolute_error':error,'byte_equal':same_bytes,
                                               'position':position.tolist(),'heading':heading.tolist(),'speed':speed.tolist()})
        tested+=1;aircraft+=size
result={'batches_tested':tested,'aircraft_forecasts_tested':aircraft,'byte_mismatching_batches':byte_failures,
        'maximum_absolute_coordinate_error_km':max_error,'first_failure_cases':failures,
        'all_coordinates_and_bytes_exact':byte_failures==0,'synthetic_rng_seed':8032,
        'controller_integrated':False,'scope':protocol['test_scope']}
if byte_failures==0:
    position=rng.uniform(-200,200,(10,2));heading=rng.uniform(0,360,10);speed=rng.uniform(130,280,10)
    scalar(position,heading,speed);batched(position,heading,speed)
    samples=[]
    for repeat in range(40):
        timings={}
        order=[('scalar',scalar),('batched',batched)]
        if repeat%2:order.reverse()
        for name,function in order:
            start=time.perf_counter()
            for _ in range(10):function(position,heading,speed)
            timings[name]=(time.perf_counter()-start)/10
        samples.append(timings)
    result['timing']={'pairs':len(samples),'fleet_size':10,'calls_per_sample':10,
                      'scalar_median_seconds':float(np.median([x['scalar'] for x in samples])),
                      'batched_median_seconds':float(np.median([x['batched'] for x in samples])),
                      'median_paired_speedup':float(np.median([x['scalar']/x['batched'] for x in samples])),
                      'samples':samples,'scope':protocol['timing_scope']}
(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
state.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),
             outcome='kernel_parity_passed' if byte_failures==0 else 'kernel_parity_failed',
             summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest());save()
print(json.dumps({k:v for k,v in result.items() if k not in ('first_failure_cases','timing')},indent=2))
if 'timing' in result:print(json.dumps({k:v for k,v in result['timing'].items() if k!='samples'},indent=2))
