"""Reproduce exact coverage of saved teacher actions by a local maneuver vocabulary."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"runs/pretrained-ppo-source-v3-verify"))
from atc_rl.demonstrations import load_demonstrations

def main():
    results={}
    for role in ("train","validation"):
        path=ROOT/"runs/imitation-data-v1"/role
        data=load_demonstrations(path,role);a=data["arrays"]
        schema=json.loads((path/"input-schema.json").read_text())
        offsets={s["name"]:s["offset"] for s in schema}
        x=a["actor"];teacher=a["teacher_action"]
        nominal=np.clip(-np.arctan2(x[:,offsets["sin_drift"]],x[:,offsets["cos_drift"]])/(np.pi/4),-1,1).astype(np.float32)
        np.testing.assert_array_equal(nominal,a["nominal_action"][:,0])
        options=np.column_stack((nominal,np.broadcast_to(np.linspace(-1,1,19,dtype=np.float32),(len(x),19))))
        choice=np.argmin(np.abs(options-teacher[:,0,None]),axis=1)
        speeds=np.array([-1,0,1],dtype=np.float32)
        speed_choice=np.argmin(np.abs(speeds[None]-teacher[:,1,None]),axis=1)
        predicted=np.column_stack((options[np.arange(len(x)),choice],speeds[speed_choice]))
        np.testing.assert_array_equal(predicted,teacher)
        results[role]={"rows":len(x),"worlds":len(data["scenarios"]),
                       "heading_choice_histogram":np.bincount(choice,minlength=20).tolist(),
                       "speed_choice_histogram":np.bincount(speed_choice,minlength=3).tolist(),
                       "max_absolute_action_error":float(np.max(np.abs(predicted-teacher)))}
    saved=json.loads((Path(__file__).parent/"action-representation.json").read_text())["datasets"]
    for role,values in results.items():
        for key in ("rows","worlds","heading_choice_histogram","speed_choice_histogram"):
            assert values[key]==saved[role][key],key
    print(json.dumps({"exact_coverage_reproduced":True,"datasets":results}))
if __name__=="__main__":main()
