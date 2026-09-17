# Optional state-dependent exploration

Gaussian exploration remains the default. The optional gSDE mode uses Stable Baselines3's existing state-dependent noise distribution and resampling schedule; it does not repeat arbitrary actions inside the PPO collector.

Specify all three options together:

    --exploration gsde --sde-weight-std 0.05 --sde-sample-freq 12

The frequency counts five-second decisions. Twelve decisions correspond to 60 simulated seconds, and SB3 also resets noise at each rollout start. The weight standard deviation initializes the noise matrix; it is not the marginal action standard deviation. The latter depends on local actor features. Accordingly, gSDE configs record initial_action_std as null and state the matrix scale separately.

The actor's mean and noise use local features only. MAPPO retains its separate centralized critic. Neutral initialization zeros only the action mean head. Training uses the same active-aircraft buffer, original dynamics, optional recorded support settings and native scoring records. Evaluation uses the deterministic actor, without exploration noise. The gSDE mode, matrix scale and resampling frequency are included in training and evaluation metadata and checked during checkpoint audits and comparisons.

Do not interpret a Gaussian-versus-gSDE contrast as temporal correlation alone: gSDE also changes the state-dependent variance parameterization. A matched gSDE frequency-1 versus frequency-12 study, with identical initial tensors, isolates the resampling schedule more directly. Compare each trained policy against its own initial policy, then repeat across seeds before claiming improvement.

Evidence motivating the option: runs/exploration-actuation-diagnostic-v2/analysis.json contains an untrained, two-world actuation probe. Similar 18-degree heading-noise widths yielded route-reference drift RMS of 11.07 degrees with fresh five-second noise and 17.77 degrees with heading noise held for 60 seconds. Static support remained fixed, speed noise was matched, and all 180 zero-noise reference metric values reproduced the saved untrained policy. This is evidence about physical response, not an RL learning or generalization result.

References:
- Raffin, Kober and Stulp, Smooth Exploration for Robotic Reinforcement Learning: https://arxiv.org/abs/2005.05719
- SB3 PPO documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Installed SB3 2.9.0 implementation inspected for the actual rollout/noise behavior.

The existing cluster-ppo-static-v1 bundle and its registered Gaussian study remain unchanged.
