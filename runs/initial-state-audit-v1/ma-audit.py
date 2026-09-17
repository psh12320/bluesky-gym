"""Inspect initial events and goal geometry on the 200-scenario MA development set."""
from pathlib import Path
from dataclasses import asdict
import csv,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union
from atc.envs import make_env
from atc.compare import load_evaluation
from atc.metrics import METRICS,SAFETY
import bluesky as bs
root=Path(__file__).resolve().parent
_,finals=load_evaluation('runs/candidate-ma-route-residual-v1-25k/validation-200')
finals={(r['episode'],r['agent']):r for r in finals};rows=[];checks={}
(root/'ma-failed-scenarios').mkdir(exist_ok=True)
env=make_env('ma','baseline')
try:
    for episode in range(200):
        _,infos=env.reset(seed=2026 if episode==0 else None);world=env.unwrapped
        scenario=json.loads(json.dumps(asdict(world.scenario)))
        if episode==12:
            reference=ROOT/'runs/diagnose-residual-v1-ma12-sa-support-check/scenario.json'
            assert scenario==json.loads(reference.read_text())
            checks['12']=dict(exact_saved_scenario_match=True,scenario_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
        center=world.scenario.center
        def xy(points):
            a=np.asarray(points);return (a-center)[...,::-1]*[60*1.852*np.cos(np.deg2rad(center[0])),60*1.852]
        sector=Polygon(xy(world.scenario.sector))
        obstacles=[Polygon(xy(o.vertices)) for o in world.scenario.obstacles]
        union=unary_union(obstacles)
        free=sector.difference(union)
        margin_free=sector.buffer(-2).difference(unary_union([o.buffer(2) for o in obstacles]))
        failed=False
        for spec in world.scenario.agents:
            info=infos[spec.ac_id];final=finals[(episode,spec.ac_id)]
            for key in ('intrusion_events','restricted_area_events','sector_exit_events'):
                assert final[key]>=info[key],(episode,spec.ac_id,key)
            start,goal=map(Point,xy([spec.start,spec.goal]))
            goal_free_distance=float(free.distance(goal)) if not free.is_empty else None
            margin_distance=float(margin_free.distance(goal)) if not margin_free.is_empty else None
            rows.append(dict(episode=episode,agent=spec.ac_id,**{k:float(info[k]) for k in METRICS},
                final_arrival=final['waypoint_reached'],start_in_obstacle=bool(union.covers(start)),
                goal_in_obstacle=bool(union.covers(goal)),
                goal_to_free_space_km=goal_free_distance,goal_to_2km_clearance_km=margin_distance))
            failed |= not bool(final['waypoint_reached'])
        if failed:
            (root/'ma-failed-scenarios'/f'episode{episode}.json').write_text(json.dumps(scenario,indent=2),encoding='utf-8')
        bs.stack.process()
finally:env.close()
with (root/'ma-reset-metrics.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report=dict(seed=2026,episodes=200,scenario_checks=checks,
 scope='Reset-time scored events and approximate planar goal geometry; original evaluation scores retained',
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
for count in (20,200):
    selected=[r for r in rows if r['episode']<count]
    initial=[(r['episode'],r['agent']) for r in selected if any(r[k]>0 for k in SAFETY)]
    missed=[r for r in selected if not r['final_arrival']]
    report[str(count)]=dict(initial_scored_safety_aircraft=initial,maximum_possible_clean_completion_rate=1-len(initial)/len(selected),
       initial_metric_totals={k:sum(r[k] for r in selected) for k in SAFETY},
       missed_aircraft_count=len(missed),missed_with_initial_scored_safety=sum(any(r[k]>0 for k in SAFETY) for r in missed),
       goals_in_obstacles=sum(r['goal_in_obstacle'] for r in selected),
       missed_goal_geometry=[{k:r[k] for k in ('episode','agent','start_in_obstacle','goal_in_obstacle','goal_to_free_space_km','goal_to_2km_clearance_km')} for r in missed])
(root/'ma-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
