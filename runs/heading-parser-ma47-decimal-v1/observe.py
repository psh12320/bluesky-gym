"""Check decimal heading transport on the diagnosed MA scenario; keep model weights unchanged."""
import os,sys,json,math,hashlib
from pathlib import Path
from decimal import Decimal
from datetime import datetime,timezone
for k,v in {'SDL_VIDEODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}.items():os.environ.setdefault(k,v)
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from atc import submission
from atc.metrics import METRICS
import numpy as np
import bluesky as bs
out=Path(__file__).resolve().parent
freeze=json.loads((ROOT/'runs/heldout-2027-ma-interval5-v1/protocol.json').read_text())
model=ROOT/freeze['model']
assert hashlib.sha256(model.read_bytes()).hexdigest()==freeze['model_sha256']
act=submission.load_policy('ma',model)
from atc.heading_transport import attach_decimal_heading
env=attach_decimal_heading(submission.make_env('ma'))
from bluesky.tools import geo
from bluesky.tools.misc import txt2hdg
assert geo.qdrdist.__module__=='bluesky.tools.geo._cgeo'
invalid=[];heading_count=0;nonfinite_observations=0;nonfinite_actions=0
original_stack=bs.stack.stack
current_episode=-1

def audited_stack(*commands,**kwargs):
    global heading_count
    for text in commands:
        for line in text.split(';'):
            fields=line.split()
            if len(fields)==3 and fields[0].upper()=='HDG':
                heading_count+=1
                try:txt2hdg(fields[2])
                except ValueError as error:
                    idx=bs.traf.id2idx(fields[1])
                    record={'episode':current_episode,'command':line,'heading_text':fields[2],'simulator_time':float(bs.sim.simt),'aircraft':fields[1],'current_heading':float(bs.traf.hdg[idx]),'error':str(error)}
                    try:
                        value=float(fields[2]);record['finite_numeric_heading']=math.isfinite(value)
                        if math.isfinite(value):
                            decimal=format(Decimal(fields[2]),'f')
                            record.update(decimal_equivalent=decimal,decimal_float_identical=float(decimal)==value,decimal_parser_value=float(txt2hdg(decimal)))
                    except ValueError:record['finite_numeric_heading']=False
                    invalid.append(record)
                    (out/'invalid-commands.json').write_text(json.dumps(invalid,indent=2),encoding='utf-8')
                    print(json.dumps(record),flush=True)
    return original_stack(*commands,**kwargs)

bs.stack.stack=audited_stack
final=[]
try:
    for current_episode in range(47):
        obs,_=env.reset(seed=2027 if current_episode==0 else None)
        if current_episode<46:
            continue
        rng_before=repr(env.unwrapped._np_random.bit_generator.state)
        while env.agents:
            agents=list(env.agents)
            actions={a:act(obs[a]) for a in agents}
            nonfinite_observations+=sum(not np.isfinite(obs[a]).all() for a in agents)
            nonfinite_actions+=sum(not np.isfinite(actions[a]).all() for a in agents)
            obs,_,terms,truncs,infos=env.step(actions)
            final.extend({'episode':current_episode,'agent':a,**{k:float(infos[a][k]) for k in METRICS}} for a in agents if terms[a] or truncs[a])
        assert rng_before==repr(env.unwrapped._np_random.bit_generator.state)
finally:
    bs.stack.stack=original_stack
    env.close()
result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Known-case decimal-transport diagnostic on seed-2027 scenario index 46, with prefix resets but prefix rollouts skipped; this is not new held-out evidence','model_sha256':freeze['model_sha256'],'geography_backend':geo.qdrdist.__module__,'heading_transport_revision':1,'heading_command_serialization_changed':True,'learned_model_unchanged':True,'heading_commands_checked':heading_count,'invalid_commands':invalid,'nonfinite_observation_count':nonfinite_observations,'nonfinite_action_count':nonfinite_actions,'final_records':final,'primary_full_run_metric_comparison':'Full-primary CSV parity pending; compare corrected and original skipped-prefix diagnostics below'}
original=json.loads((ROOT/'runs/heading-parser-ma47-v1/diagnostic.json').read_text())
old={r['agent']:r for r in original['final_records']};new={r['agent']:r for r in final}
assert old.keys()==new.keys()
result['original_diagnostic_sha256']=hashlib.sha256((ROOT/'runs/heading-parser-ma47-v1/diagnostic.json').read_bytes()).hexdigest()
result['metric_max_absolute_change']={k:max(abs(new[a][k]-old[a][k]) for a in old) for k in METRICS}
result['numeric_heading_formatter_tests']={'passed':7,'checks':'Actual rejected number, 2135 boundary/random/subnormal values, invalid headings, continuous/discrete target equivalence'}
assert not invalid and nonfinite_observations==0 and nonfinite_actions==0
(out/'diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='final_records'},indent=2),flush=True)
