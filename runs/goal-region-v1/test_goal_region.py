from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from shapely.geometry import LineString, Point
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'reference/source'))
sys.path.insert(0,str(BASE))
from atc.routes import VisibilityPlanner
from goal_region import path_to_region, choose_region_route, valid_scored_endpoints, GoalRegionInput

SECTOR=[(-40,-40),(40,-40),(40,40),(-40,40)]

def assert_legal(planner,path,start,goal,radius=4.75):
    assert path is not None
    np.testing.assert_array_equal(path[0],start)
    assert np.linalg.norm(path[-1]-goal)<=radius+1e-8
    assert planner.visible_domain.covers(LineString(path))


def test_blocked_goal_centre_can_have_a_reachable_capture_region():
    obstacle=[(-2,-8),(2,-8),(2,8),(-2,8)]
    planner=VisibilityPlanner(SECTOR,[obstacle],clearance=0)
    start,goal=np.array([-25.,0]),np.array([0.,0])
    assert planner.path(start,goal) is None
    path=path_to_region(planner,start,goal)
    assert_legal(planner,path,start,goal)
    assert path[-1,0] < -2


def test_fully_blocked_region_and_disconnected_region_are_rejected():
    large=[(-10,-10),(10,-10),(10,10),(-10,10)]
    planner=VisibilityPlanner(SECTOR,[large],clearance=2)
    assert path_to_region(planner,[-25,0],[0,0]) is None
    wall=[(-2,-45),(2,-45),(2,45),(-2,45)]
    planner=VisibilityPlanner(SECTOR,[wall],clearance=2)
    assert path_to_region(planner,[-25,0],[25,0]) is None


def test_path_keeps_buffered_clearance_and_never_lengthens_point_route():
    obstacle=[(-4,-12),(4,-12),(4,12),(-4,12)]
    planner=VisibilityPlanner(SECTOR,[obstacle],clearance=2,sector_clearance=2)
    rng=np.random.default_rng(6117)
    checked=0
    for _ in range(100):
        start,goal=rng.uniform(-35,35,(2,2))
        original=planner.path(start,goal)
        if original is None:continue
        path=path_to_region(planner,start,goal)
        assert_legal(planner,path,start,goal)
        assert np.linalg.norm(np.diff(path,axis=0),axis=1).sum() <= np.linalg.norm(np.diff(original,axis=0),axis=1).sum()+1e-8
        assert LineString(path).distance(Point(0,0))>=6-1e-7
        checked+=1
    assert checked>=60


def test_endpoint_validation_can_reject_every_geometric_candidate():
    planner=VisibilityPlanner(SECTOR,[],clearance=0)
    assert path_to_region(planner,[-25,0],[25,0],endpoint_valid=lambda values:np.zeros(len(values),bool)) is None
    valid=lambda values:valid_scored_endpoints(values,[25,0],[52,4])
    path=path_to_region(planner,[-25,0],[25,0],endpoint_valid=valid)
    assert valid(path[-1:])[0]


def test_preferred_clearance_is_kept_when_lower_clearance_also_has_a_path():
    choices={c:VisibilityPlanner(SECTOR,[],clearance=c,sector_clearance=c) for c in (6,3,2,1,0)}
    selected=choose_region_route(choices,[-25,0],[25,0])
    assert selected[0].clearance==6


def test_replanning_uses_original_goal_not_previous_region_endpoint(monkeypatch):
    import bluesky as bs
    monkeypatch.setattr(bs,'sim',SimpleNamespace(simt=25.0))
    monkeypatch.setattr(bs,'traf',SimpleNamespace(lat=np.array([0.]),lon=np.array([0.]),id2idx=lambda agent:0))
    seen=[]
    planner=SimpleNamespace(visible=lambda start,targets:np.zeros(len(targets),bool),target_index=lambda *args:1)
    world=SimpleNamespace(_ensure_routes=lambda:None,_xy=lambda points:np.asarray(points),
        _routes={'A':np.array([[-25.,0],[5.,0]])},_planners={'A':planner},_route_indices={'A':1},
        _input_replan_clock={},_region_original_goals={'A':np.array([10.,0])},
        route_input_statistics={'failed_replans':0,'route_replans':0})
    world._choose_route=lambda start,goal:seen.append(goal.copy())
    GoalRegionInput._route_reference(world,'A')
    np.testing.assert_array_equal(seen[0],[10.,0])
    assert world.route_input_statistics['failed_replans']==1


@pytest.mark.parametrize('radius',[0,-1,float('nan')])
def test_invalid_region_is_rejected(radius):
    planner=VisibilityPlanner(SECTOR,[],clearance=0)
    with pytest.raises(ValueError):path_to_region(planner,[-25,0],[25,0],radius=radius)
