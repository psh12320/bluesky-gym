
"""Offline observation-scale and actor-sensitivity audit; no simulator counterfactual."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-cpa-source-v2-verify"
sys.path.insert(0,str(SOURCE))
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def main():
    from stable_baselines3 import PPO
    from atc_rl.checkpoint_identity import verified_checkpoint
    from atc_rl.cluster import verify
    verify(SOURCE)
    plan=read(WORK/"protocol.json")
    dataset=ROOT/plan["dataset"]
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==plan["dataset_sha256"]
    layout=read(ROOT/plan["layout"])
    with np.load(dataset) as archive:observations=archive["observations"].copy()
    assert observations.shape==(371,170) and np.isfinite(observations).all()
    def columns(names):
        return np.concatenate([np.arange(*layout[key]) for key in names])
    groups={
        "relative_position":columns(("x_r","y_r")),
        "relative_velocity":columns(("vx_r","vy_r")),
        "relative_track":columns(("cos_track","sin_track")),
        "traffic_distance":columns(("intruder_distance",)),
        "predictive_conflict":columns(("traffic_present","traffic_tcpa","traffic_dcpa","traffic_entry_time","traffic_predicted_conflict")),
        "navigation":columns(("cos_drift","sin_drift","waypoint_distance")),
        "obstacles":columns(("obstacle_cos_bearing","obstacle_sin_bearing","obstacle_distance","obstacle_radius")),
        "sector":columns(("sector_point_cos_bearing","sector_point_sin_bearing","sector_point_distance","inside_sector")),
        "airspeed":columns(("airspeed",)),
        "time":columns(("time_remaining",))}
    assert sorted(np.concatenate(list(groups.values())).tolist())==list(range(170))
    present=observations[:,columns(("traffic_present",))]>0
    pairs={"position":np.stack([observations[:,columns((k,))] for k in ("x_r","y_r")],axis=-1),
           "velocity":np.stack([observations[:,columns((k,))] for k in ("vx_r","vy_r")],axis=-1)}
    input_scale={}
    for key,values in pairs.items():
        norms=np.linalg.norm(values,axis=-1)[present]
        input_scale[key]={"present_slots":int(present.sum()),
            "absolute_component_p50_p90_p99":np.quantile(np.abs(values[present]),[.5,.9,.99]).tolist(),
            "vector_norm_p10_p50_p90":np.quantile(norms,[.1,.5,.9]).tolist()}
    input_scale["separation_threshold_normalized"]=9260/1000000
    results={}
    torch.set_num_threads(1)
    for name,identity in plan["models"].items():
        path=ROOT/identity["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==identity["sha256"]
        records=read(path.parent/"checkpoints.json")
        record=next(r for r in records if r["file"]==path.name)
        assert verified_checkpoint(path.parent,record)==path.resolve()
        model=PPO.load(path,device="cpu")
        assert not model.use_sde and model.observation_space["actor"].shape==(170,)
        cfg=read(path.parent/"config.json")
        x=observations.copy()
        if cfg.get("mask_conflict_features",False):x[:,groups["predictive_conflict"]]=0
        actor=torch.tensor(x,requires_grad=True)
        inputs={"actor":actor,"critic":torch.zeros((len(x),*model.observation_space["critic"].shape))}
        distribution=model.policy.get_distribution(inputs).distribution
        actions=distribution.mean
        gradients=[]
        for dimension in range(2):
            gradients.append(torch.autograd.grad(actions[:,dimension].sum(),actor,retain_graph=dimension==0)[0].detach().numpy())
        gradients=np.stack(gradients,axis=1)
        observed_std=x.std(axis=0)
        standardized=gradients*observed_std[None,None,:]
        group_values={}
        for group,index in groups.items():
            local_norm=np.sqrt(np.sum(standardized[:,:,index]**2,axis=2))
            group_values[group]={"heading_degrees_rms":float(np.sqrt(np.mean(local_norm[:,0]**2))*90),
                                "speed_normalized_rms":float(np.sqrt(np.mean(local_norm[:,1]**2))),
                                "number_of_inputs":len(index),
                                "input_std_rms":float(np.sqrt(np.mean(observed_std[index]**2)))}
        total=sum(v["heading_degrees_rms"]**2 for v in group_values.values())
        for value in group_values.values():
            value["fraction_of_squared_standardized_heading_gradient"]=value["heading_degrees_rms"]**2/total if total else None
        mean=actions.detach().numpy()
        results[name]={"model_sha256":identity["sha256"],"live_transitions":record["live_transitions"],
            "heading_mean_degrees":float(mean[:,0].mean()*90),
            "heading_std_across_reference_states_degrees":float(mean[:,0].std()*90),
            "groups":group_values}
    result={"protocol_sha256":hashlib.sha256((WORK/"protocol.json").read_bytes()).hexdigest(),
        "reference_states":len(observations),"input_scale":input_scale,"models":results,
        "performance_evidence":False,"limitations":plan["limitations"]}
    with (WORK/"results.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"input_scale":input_scale,"position_sensitivity":{k:{
        "heading_degrees_rms":v["groups"]["relative_position"]["heading_degrees_rms"],
        "fraction_squared_gradient":v["groups"]["relative_position"]["fraction_of_squared_standardized_heading_gradient"],
        "heading_variation":v["heading_std_across_reference_states_degrees"]} for k,v in results.items()}}))

if __name__=="__main__":main()
