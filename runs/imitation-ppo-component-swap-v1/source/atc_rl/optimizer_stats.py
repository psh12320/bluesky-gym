"""Inspect saved Adam moments; these diagnostics do not measure policy quality."""
import argparse
import json
from pathlib import Path


def statistics(path):
    import torch
    from stable_baselines3 import PPO
    from atc_rl.actor_reference import is_actor_parameter
    torch.set_num_threads(1)
    model=PPO.load(path,device='cpu')
    optimizer=model.policy.optimizer
    group_for={id(p):group for group in optimizer.param_groups for p in group['params']}
    groups={name:[] for name in ('actor_hidden','actor_head','log_std','critic')}
    attenuation={key:[] for key in groups}
    for name,parameter in model.policy.named_parameters():
        state=optimizer.state.get(parameter)
        if not state or 'exp_avg_sq' not in state:continue
        group=group_for[id(parameter)]
        step=float(state['step']);beta2=group['betas'][1]
        rms=(state['exp_avg_sq']/(1-beta2**step)).sqrt().reshape(-1).detach()
        key=('log_std' if name=='log_std' else 'actor_head' if name.startswith('action_net.')
             else 'actor_hidden' if is_actor_parameter(name) else 'critic')
        groups[key].append(rms)
        attenuation[key].append(rms/(rms+group['eps']))
    results={}
    for key,values in groups.items():
        if not values:continue
        joined=torch.cat(values);factors=torch.cat(attenuation[key])
        results[key]={'parameters':joined.numel(),'median_bias_corrected_gradient_rms':float(joined.median()),
                      'median_epsilon_attenuation_factor':float(factors.median()),
                      'fraction_attenuation_below_half':float((factors<.5).float().mean())}
    return {'checkpoint':str(Path(path).resolve()),'optimizer_steps':sorted({float(s['step']) for s in optimizer.state.values() if 'step' in s}),
        'optimizer_epsilon':sorted({g['eps'] for g in optimizer.param_groups}),
        'initial_log_standard_deviation':model.policy_kwargs.get('log_std_init'),
        'final_action_standard_deviations':model.policy.log_std.detach().exp().tolist(),
        'groups':results,'interpretation':'Saved optimizer moments are measured after clipping. Trajectories differ between trained policies; this diagnostic alone is not causal or performance evidence.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Choose a fresh output file')
    result=[statistics(path) for path in args.model]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{'checkpoint':r['checkpoint'],'actor_hidden':r['groups']['actor_hidden'],
                       'action_std':r['final_action_standard_deviations']} for r in result]))


if __name__=='__main__':main()
