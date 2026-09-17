# Residual-controller research notes

Primary sources checked on 15 September 2026:

- Johannink et al., [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201). This work combines an existing controller with learned residual commands and stores residual actions in replay. It provides methodological precedent for the controller decomposition, rather than evidence about BlueSky or this competition.
- Silver et al., [Residual Policy Learning](https://arxiv.org/abs/1812.06298), sections IV-A and IV-B. The authors initialize the residual output to zero. They also describe training the critic before the actor because inaccurate initial value estimates can damage a strong starting policy. Their experiment results concern robotic manipulation, not air traffic control.

Our implementation uses SAC, bounds proposed heading adjustments to 15 degrees and
passes the combined command through joint traffic/static correction. The mapping
and planner remain outside the learned network; replay contains residual actions.
This is an application and adaptation of established residual-learning ideas, not
a claim to have invented residual RL.

Current pilots use data-collection warmup followed by ordinary SAC updates. They do
not implement critic-only burn-in. If early checkpoints lose the strong reference
behavior, a matched critic-first training ablation is a justified next experiment.
That is a hypothesis informed by the literature, not a diagnosis of a measured
failure in the current pilots. Do not alter the running experiments mid-run.
