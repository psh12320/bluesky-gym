"""Check the actual simulator clock and selected conflict-resolution implementation."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
for k,v in {'SDL_VIDEODRIVER':'dummy','PYGAME_HIDE_SUPPORT_PROMPT':'1','OMP_NUM_THREADS':'1'}.items():os.environ.setdefault(k,v)
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from atc import submission
import bluesky as bs
from bluesky.core.entity import getproxied
from bluesky.traffic.asas.resolution import ConflictResolution
results={}
for track,folder,interval in [('sa','heldout-2027-sa-reach250-v1',10),('ma','heldout-2027-ma-interval5-v1',5)]:
    freeze=json.loads((ROOT/'runs'/folder/'protocol.json').read_text())
    model=ROOT/freeze['model']
    assert hashlib.sha256(model.read_bytes()).hexdigest()==freeze['model_sha256']
    act=submission.load_policy(track,model)
    env=submission.make_env(track)
    try:
        obs,_=env.reset(seed=2026)
        before=float(bs.sim.simt)
        if track=='sa':env.step(act(obs))
        else:env.step({a:act(obs[a]) for a in env.agents})
        elapsed=float(bs.sim.simt)-before
        assert elapsed==interval and bs.sim.simdt==1
        implementation=type(getproxied(bs.traf.cr))
        assert implementation is ConflictResolution
        results[track]={'model_sha256':freeze['model_sha256'],'simulator_elapsed_seconds':elapsed,'actual_simulator_step_seconds':float(bs.sim.simdt),'expected_decision_interval_seconds':interval,'selected_resolution_implementation':implementation.__module__+'.'+implementation.__qualname__,'bluesky_conflict_resolution_off':True}
    finally:env.close()
out={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'seed':2026,'decisions_per_track':1,'scope':'Read the actual BlueSky clock and unwrapped conflict-resolution implementation after each frozen controller executes its first development decision. No simulator-setting changes.','checks':results}
(Path(__file__).resolve().parent/'simulation-audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
