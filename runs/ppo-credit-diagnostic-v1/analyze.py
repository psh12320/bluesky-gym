"""Analyze registered reward and value traces without changing a trained policy."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math
import numpy as np
WORK=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None

def advantages(rewards,values,gamma,lam):
    next_values=np.append(values[1:],0.)
    residuals=rewards+gamma*next_values-values
    result=np.zeros(len(rewards))
    carry=0.
    for t in range(len(rewards)-1,-1,-1):
        carry=residuals[t]+gamma*lam*carry
        result[t]=carry
    return result

def main():
    plan=read(WORK/"protocol.json")
    completed=read(WORK/"complete.json")
    gamma=.996508469331006
    result={"cases":{},"limitations":plan["limitations"],
            "timing_weights":{"gamma":gamma,"seconds_per_decision":5,"by_lambda":{}}}
    for lam in (.95,.99,1.):
        factor=gamma*lam
        result["timing_weights"]["by_lambda"][str(lam)]={
            "residual_e_folding_seconds":-5/math.log(factor),
            "residual_weight_after_seconds":{str(s):factor**(s/5) for s in (30,60,120,180,300)},
            "discount_weight_after_seconds":{str(s):gamma**(s/5) for s in (30,60,120,180,300)}}
    for case in plan["cases"]:
        path=WORK/(case+"-steps.csv")
        assert sha(path)==completed["cases"][case]["step_csv_sha256"]
        with path.open(newline="",encoding="utf-8") as f:raw=list(csv.DictReader(f))
        groups=defaultdict(list)
        for row in raw:groups[(int(row["episode"]),row["agent"])].append(row)
        all_values=[];all_returns=[];all_adv={str(l):[] for l in (.95,.99,1.)}
        leads=[];future_buckets=defaultdict(list);has_prediction=[];future_conflict=[]
        reward_parts={k:[] for k in ("goal","intrusion","restricted","outside","drift")}
        rows_out=[]
        for key,rows in groups.items():
            assert [int(r["decision"]) for r in rows]==list(range(len(rows)))
            assert all(int(r["terminal"])==0 for r in rows[:-1]) and int(rows[-1]["terminal"])==1
            rewards=np.array([float(r["native_reward"]) for r in rows])*.01
            values=np.array([float(r["critic_value"]) for r in rows])
            intrusion=np.array([float(r["intrusion_seconds"]) for r in rows])
            prediction=np.array([int(r["predicted_threats"])>0 for r in rows])
            returns=np.zeros(len(rows))
            carry=0.
            for t in range(len(rows)-1,-1,-1):
                carry=rewards[t]+gamma*carry
                returns[t]=carry
            adv={str(l):advantages(rewards,values,gamma,l) for l in (.95,.99,1.)}
            assert np.allclose(adv["1.0"],returns-values,rtol=1e-10,atol=1e-10)
            all_values.extend(values);all_returns.extend(returns)
            for lam in adv:all_adv[lam].extend(adv[lam])
            for part in reward_parts:reward_parts[part].extend(float(r["reward_"+part]) for r in rows)
            assert np.allclose(rewards/.01,sum(np.array([float(r["reward_"+part]) for r in rows]) for part in reward_parts))
            occupied=intrusion>0
            starts=np.flatnonzero(occupied & ~np.append(False,occupied[:-1]))
            for start in starts:
                # Look only within the 180-second predictor horizon.
                earlier=np.flatnonzero(prediction[max(0,start-36):start])+max(0,start-36)
                leads.append(float((start-earlier[0])*5) if len(earlier) else None)
            for t in range(len(rows)):
                next_intrusion=np.flatnonzero(occupied[t:])
                delay=int(next_intrusion[0])*5 if len(next_intrusion) else None
                future=delay is not None and delay<=180
                has_prediction.append(bool(prediction[t]));future_conflict.append(future)
                bucket=("no_later_intrusion" if delay is None else "current_intrusion" if delay==0
                        else "within_30s" if delay<=30 else "31_to_60s" if delay<=60
                        else "61_to_120s" if delay<=120 else "121_to_180s" if delay<=180 else "after_180s")
                future_buckets[bucket].append((returns[t],values[t],rewards[t],float(prediction[t])))
                rows_out.append({"case":case,"episode":key[0],"agent":key[1],"decision":t,
                    "critic_value":values[t],"discounted_realized_return":returns[t],"future_intrusion_delay_seconds":delay,
                    **{"advantage_lambda_"+lam:array[t] for lam,array in adv.items()}})
        with (WORK/(case+"-credit.csv")).open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
        v=np.array(all_values);ret=np.array(all_returns);pred=np.array(has_prediction);future=np.array(future_conflict)
        variance=float(np.var(ret));leads_present=[lead for lead in leads if lead is not None]
        result["cases"][case]={
            "live_decisions":len(v),"metric_matches":completed["cases"][case]["metric_matches"],
            "native_reward_components":{"total":{k:float(np.sum(a)) for k,a in reward_parts.items()},
                                       "nonzero_fraction":{k:float(np.mean(np.abs(a)>1e-12)) for k,a in reward_parts.items()}},
            "critic_on_deterministic_trajectories":{"mean_prediction":float(v.mean()),"mean_realized_return":float(ret.mean()),
                "prediction_return_correlation":correlation(v,ret),"mse":float(np.mean((v-ret)**2)),
                "realized_return_variance":variance,"explained_variance":float(1-np.var(v-ret)/variance) if variance else None},
            "offline_full_trajectory_advantages":{lam:{"rms":float(np.sqrt(np.mean(np.array(a)**2))),
                "correlation_with_realized_advantage":correlation(a,ret-v),
                "sign_agreement_with_realized_advantage":float(np.mean(np.sign(a)==np.sign(ret-v)))}
                for lam,a in all_adv.items()},
            "aircraft_intrusion_occupancy_onsets":{"count":len(leads),"predicted_earlier_within_180s":len(leads_present),
                "earliest_warning_lead_seconds_p10_p50_p90":np.quantile(leads_present,[.1,.5,.9]).tolist() if leads_present else None},
            "aircraft_level_flags_descriptive_only":{"flag_fraction":float(pred.mean()),
                "observed_intrusion_within_180s_given_flag":float(future[pred].mean()) if pred.any() else None,
                "flag_given_intrusion_within_180s":float(pred[future].mean()) if future.any() else None},
            "future_intrusion_buckets":{k:{"decisions":len(a),"mean_discounted_return":float(np.mean(a,axis=0)[0]),
                "mean_critic_value":float(np.mean(a,axis=0)[1]),"mean_immediate_scaled_reward":float(np.mean(a,axis=0)[2]),
                "predicted_flag_fraction":float(np.mean(a,axis=0)[3])} for k,a in future_buckets.items()}}
    result.update(protocol_sha256=sha(WORK/"protocol.json"),diagnostic_only=True,training_performed=False)
    with (WORK/"analysis.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
    print(json.dumps({"timing_weights":result["timing_weights"],
        "cases":{k:{"critic":v["critic_on_deterministic_trajectories"],"onsets":v["aircraft_intrusion_occupancy_onsets"]}
                 for k,v in result["cases"].items()}}))
if __name__=="__main__":main()
