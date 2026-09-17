# PPO reward-scale experiment

Status: scaling-only and combined shaping/scaling training and evaluation completed; both failed the safety advancement screen.

The raw and shaped PPO pilots provide evidence of poor learning. Saved Adam
moments from the shaped pilot have a median bias-corrected gradient RMS of
2.33e-6 in the actor hidden layers, below the optimizer epsilon of 1e-5 for 99.5%
of these parameters. The raw pilot median is 1.64e-5. These moments are measured
after global gradient clipping and do not establish causation. The measurements
are retained in runs/progress-reward-v1/optimizer-diagnostic.json.

The installed SB3 PPO implementation applies one norm bound to all policy and
value parameters. In contrast, the [MAPPO authors' implementation](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py)
clips actor and critic gradients separately and supports value normalization.
Our current MAPPO test changes critic information while retaining the existing
PPO optimizer. It does not reproduce all settings from that implementation.

The next proposed optimization ablation is a fixed learning-reward scale of 0.01,
with potential shaping disabled. It will retain the original reward ratios,
initial actor/critic, exploration distribution, seed 49900, two simulator workers,
100k live budget and other PPO settings. The evaluator will use native metrics.
This scaling changes optimization and the balance with entropy regularization;
it should not be presented as equivalent finite-sample optimization.

The wrapper changes only returned training rewards and training episode returns.
Scoring metrics, native rewards in information records, actions, observations,
terminations and inactive masks remain unchanged. Seven tests cover positive and
invalid scales, arrival boundaries, inactive slots and preservation of underlying
records. The complete native and learning returns must also be audited after any
real training run. It is not enabled in the raw PPO or MAPPO runs already running.

Use paired development worlds 20260, the initial actor and the fixed classical
benchmark. Preserve every metric and unsuccessful result. A single pilot cannot
establish generalization; retain streams 20301 and 20302 for candidate validation.


### Completed scale-only pilot

At 101,160 live transitions, scaling alone produced 8% arrivals and 4% clean
completion on 20 paired development worlds. The untrained policy had 4% arrivals.
Intrusion time increased from 100.64 to 295.43 seconds. The preset screen failed.
Native metrics, exact initial-policy identity and return accounting are preserved
under runs/ppo-reward-scale-v1. The combined potential-shaping scale 100 and
learning-reward scale 0.01 run completed 104,783 live transitions; its evaluation
is pending. Together with the completed raw and shaping-only controls, this
completes training of the four reward-transformation cells, each with one seed.
Rollout completion produces small differences in actual experience budgets.


### Completed development evaluation

Combined shaping/scaling produced 18% arrivals and 3.5% clean completion. Intrusion time was 263.31 seconds, restricted-area time 108.82 seconds and outside-sector time 1037.125 seconds. The arrival gain passed the provisional 10-point requirement, but intrusion time increased; do not advance it as a successful navigation-and-safety recipe. See runs/progress-scaled-v1/comparison-dev20 and advancement-screen.json.
