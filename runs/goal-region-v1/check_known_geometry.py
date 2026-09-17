"""Check the two previously identified obstructed development goals geometrically."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from shapely.geometry import Point,Polygon,LineString
from shapely.ops import unary_union
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/'reference/source'));sys.path.insert(0,str(BASE))
from atc.routes import VisibilityPlanner
from goal_region import path_to_region,valid_scored_endpoints,inverse_xy
from core.tools import kwikqdrdist
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
records=[]
for episode,agent in [(103,'KL006'),(107,'KL005')]:
    source=ROOT/f'runs/initial-state-audit-v1/ma-failed-scenarios/episode{episode}.json'
    scenario=json.loads(source.read_text());center=np.asarray(scenario['center'])
    def xy(values):
        values=np.asarray(values,dtype=float)
        north=(values[...,0]-center[0])*60*1.852
        east=(values[...,1]-center[1])*60*1.852*np.cos(np.deg2rad(center[0]))
        return np.stack((east,north),axis=-1)
    spec=next(a for a in scenario['agents'] if a['ac_id']==agent)
    start,goal=xy(spec['start']),xy(spec['goal'])
    sector=xy(scenario['sector']);obstacles=[xy(o['vertices']) for o in scenario['obstacles']]
    free=Polygon(sector).difference(unary_union([Polygon(o) for o in obstacles]))
    margin_free=Polygon(sector).buffer(-2).difference(unary_union([Polygon(o).buffer(2) for o in obstacles]))
    entries=[]
    for clearance in (6,3,2,1,0):
        planner=VisibilityPlanner(sector,obstacles,clearance,sector_clearance=min(clearance,6))
        path=path_to_region(planner,start,goal,endpoint_valid=lambda endpoints:valid_scored_endpoints(endpoints,goal,center))
        row={'clearance_km':clearance,'point_route_exists':planner.path(start,goal) is not None,'region_route_exists':path is not None}
        if path is not None:
            assert planner.visible_domain.covers(LineString(path))
            endpoint=inverse_xy(path[-1],center)
            _,distance=kwikqdrdist(endpoint[0],endpoint[1],spec['goal'][0],spec['goal'][1])
            assert distance*1.852<4.95
            row.update(endpoint_scored_distance_km=float(distance*1.852),geometric_path_length_km=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()))
        entries.append(row)
    records.append({'episode':episode,'agent':agent,'source_scenario_sha256':sha(source),
        'goal_distance_to_free_space_km':float(free.distance(Point(goal))),
        'goal_distance_to_2km_margin_space_km':float(margin_free.distance(Point(goal))),
        'routes':entries})
result={'seed':2026,'scope':'Two known development geometry diagnostics; no simulator execution, performance improvement or hidden-test evidence',
    'navigation_radius_km':4.75,'original_capture_radius_km':5,'script_sha256':sha(Path(__file__)),
    'extension_sha256':sha(BASE/'goal_region.py'),'cases':records}
with (BASE/'known-geometry-audit.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))
