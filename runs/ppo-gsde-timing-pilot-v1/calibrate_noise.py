from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/onpolicy-gsde-source-v1-verify"
sys.path.insert(0,str(SOURCE))
def main():
    import numpy as np,torch
    from stable_baselines3 import PPO
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    sample_path=ROOT/"runs/ppo-observation-diagnostic-v1/observations.npz"
    samples=np.load(sample_path)["observations"]
    cases={"integration_seed49040":ROOT/"runs/gsde-development-v1/ppo/train/initial-model.zip",
           "registered_seed51700":WORK/"seed-51700/every1/train/initial-model.zip"}
    output={}
    for label,path in cases.items():
        initial_hash=sha(path)
        model=PPO.load(path,device="cpu")
        assert model.num_timesteps==0 and model.use_sde
        inputs={"actor":torch.as_tensor(samples),"critic":torch.zeros((len(samples),*model.observation_space["critic"].shape))}
        entries={}
        with torch.no_grad():
            for weight in (.01,.05):
                model.policy.log_std.fill_(np.log(weight))
                distribution=model.policy.get_distribution(inputs).distribution
                std=distribution.stddev.numpy()
                assert float(distribution.mean.abs().max())==0
                entries[str(weight)]={"std_mean_per_channel":std.mean(axis=0).tolist(),
                    "std_p10_p50_p90_per_channel":np.quantile(std,[.1,.5,.9],axis=0).tolist(),
                    "heading_std_median_degrees":float(np.median(std[:,0])*90)}
        assert sha(path)==initial_hash
        output[label]={"checkpoint_sha256":initial_hash,"in_memory_probe_only":True,"weights":entries}
    with (WORK/"noise-scale-review.json").open("x",encoding="utf-8") as f:
        json.dump({"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"cases":output,
            "reference_observations_sha256":sha(sample_path),"sample_count":len(samples),
            "interpretation":"Marginal action noise is input-dependent. The registered weight std .01 is unchanged; this is not a performance test.",
            "script_sha256":sha(Path(__file__))},f,indent=2)
    print(json.dumps(output))
if __name__=="__main__":main()
