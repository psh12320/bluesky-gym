
"""Summarize the registered actuation probe without treating it as RL performance."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json
import numpy as np

WORK=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))

def numeric(rows,key):
    return np.array([float(r[key]) for r in rows if r[key]!=""],dtype=float)

def rms(values):
    return float(np.sqrt(np.mean(values**2))) if len(values) else None

def summarize_steps(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r["episode"],r["agent"])].append(r)
    before=[];after=[]
    for records in grouped.values():
        records.sort(key=lambda r:int(r["decision"]))
        for a,b in zip(records,records[1:]):
            assert int(b["decision"])==int(a["decision"])+1
            before.append(float(a["latent_heading"]));after.append(float(b["latent_heading"]))
    latent=numeric(rows,"latent_heading")*90.
    correlation=None if np.std(before)==0 or np.std(after)==0 else float(np.corrcoef(before,after)[0,1])
    drift=numeric(rows,"drift_after_degrees")
    unfiltered=[r for r in rows if r["static_override"]=="0" and r["mapped_turn_saturated"]=="0"]
    return {
        "aircraft":len(grouped),"live_decisions":len(rows),
        "latent_heading_offset_std_degrees":float(np.std(latent)),
        "latent_heading_lag_one_correlation":correlation,
        "actual_turn_rms_degrees_per_decision":rms(numeric(rows,"actual_turn_degrees")),
        "applied_turn_command_rms_degrees":rms(numeric(rows,"applied_turn_degrees")),
        "route_drift_rms_degrees":rms(drift),
        "route_drift_absolute_p50_p90_degrees":np.quantile(np.abs(drift),[.5,.9]).tolist(),
        "mean_per_aircraft_route_drift_rms_degrees":float(np.mean([rms(numeric(v,"drift_after_degrees")) for v in grouped.values()])),
        "static_override_fraction":float(np.mean(numeric(rows,"static_override"))),
        "mapped_turn_saturation_fraction":float(np.mean(numeric(rows,"mapped_turn_saturated"))),
        "latent_heading_clipped_fraction":float(np.mean(np.abs(numeric(rows,"latent_heading_unclipped"))>1)),
        "unfiltered_unsaturated_decisions":len(unfiltered),
        "unfiltered_unsaturated_turn_rms_degrees":rms(numeric(unfiltered,"actual_turn_degrees")),
        "unfiltered_unsaturated_route_drift_rms_degrees":rms(numeric(unfiltered,"drift_after_degrees"))
    }

def main():
    complete=read(WORK/"complete.json")
    plan=read(WORK/"protocol.json")
    assert complete["zero_noise_reference_metric_matches"]==180
    expected_noise=np.random.default_rng(plan["noise_seed"]).standard_normal((2,600,10,2))
    np.testing.assert_array_equal(np.load(WORK/"standard-normal-draws.npy"),expected_noise)
    cases={}
    allrows={}
    for name,setting in plan["cases"].items():
        with (WORK/(name+"-steps.csv")).open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
        allrows[name]=rows
        cases[name]={"pooled":summarize_steps(rows),
            "worlds":{str(i):summarize_steps([r for r in rows if int(r["episode"])==i]) for i in range(2)},
            "descriptive_physical_summary":complete["cases"][name],
            "steps_csv_sha256":hashlib.sha256((WORK/(name+"-steps.csv")).read_bytes()).hexdigest()}
    lookup=lambda rows:{(r["episode"],r["decision"],r["agent"]):r for r in rows}
    iid=lookup(allrows["iid020"]);held=lookup(allrows["held020"])
    common=iid.keys()&held.keys()
    assert all(iid[key]["latent_speed"]==held[key]["latent_speed"] for key in common)
    result={"protocol_sha256":hashlib.sha256((WORK/"protocol.json").read_bytes()).hexdigest(),
            "cases":cases,"paired_speed_noise_checks":len(common),
            "zero_noise_reference_metric_matches":180,
            "limitations":plan["limitations"],
            "interpretation":"Physical-response diagnostics only. These are untrained policies, and no training or generalization benefit can be inferred."}
    with (WORK/"analysis.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({name:value["pooled"] for name,value in cases.items()}))
if __name__=="__main__":main()
