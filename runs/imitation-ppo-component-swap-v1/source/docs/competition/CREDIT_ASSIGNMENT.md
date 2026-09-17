# Credit assignment control

The training option --gae-lambda defaults to 0.95. It changes the weighting of future temporal-difference residuals used by PPO and MAPPO; it does not change gamma, the native reward, observations, actions, scoring, or the classical benchmark. Initial and final serialized models are checked against the recorded value, including their reconstructed rollout buffers.

At five seconds per decision and gamma 0.996508469331006, a residual 180 seconds later has weight 0.139 for lambda 0.95, versus 0.614 for lambda 0.99. These are GAE residual weights, not a hard planning horizon. A learned critic can carry longer-term information. Higher lambda can also raise estimator variance.

The registered diagnostic runs/ppo-credit-diagnostic-v1 replayed three preserved policies on two reused development worlds. All 540 native metric values matched their previous evaluations. The two trained critics had correlations of approximately 0.80 and 0.60 with realized discounted returns. They therefore learned useful predictions on these trajectories; this does not support a claim that the critic is generally broken. Predictions before later conflicts still differed from realized returns, and future-residual weighting is a hypothesis for an isolated training comparison.

The diagnostic uses deterministic trajectories, whereas the critic was trained for a stochastic policy. Its full-trajectory GAE calculation also differs from the actual finite training rollouts. Closer agreement with one realized return is not proof of a better policy-gradient estimator. No lambda change has yet demonstrated better training performance.

A prospective test should compare lambda 0.95 with 0.99 using matched source, initialization, Gaussian exploration, input scale, support, seed and budget. Keep this separate from the active input-scaling experiment. Evaluate each initial, intermediate and final policy and report multiple training seeds before any generalization claim.

Method: [Schulman et al., Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). The implementation uses the installed SB3 rollout-buffer recursion; no custom advantage formula replaces it.
