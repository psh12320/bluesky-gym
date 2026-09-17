# Learning from command-filter corrections

This development experiment tests whether feedback about the amount of command
correction improves the learned controller. The selected deployment, ongoing full
scoring runs, simulator, planner and command filter remain fixed.

## Motivation and attribution

The current policy proposes a residual command; the filter can replace that
command before execution. Different proposals can therefore lead to the same
executed command. Penalties for action adjustments are established research,
including the SE-RL discussion in Markgraf et al., *Safe Reinforcement Learning
using Action Projection: Safeguard the Policy or the Environment?*, revision 2,
16 April 2026: https://arxiv.org/html/2509.12833v2. This is an attributed experiment
with an ATC-specific normalization, not a claim of a new general algorithm.
Our finite command search has no formal safety guarantee.

## Fixed experiment

The declaration is runs/intervention-reward-v1/protocol.json, SHA-256
89a73663fe90b4198d431678ba9342f96255b26377aa2c8d116bb75db1dd7d13.
Two arms use the same training seed 2910 and the same initial network parameters:
coefficient zero and coefficient one. Each trains fresh shared MA SAC for 25,000
counted transitions, preserving the 64x64 network, 5,000-transition warmup,
four updates per vector step, batch size 256 and 100,000-slot replay capacity.
The fixed callback has 7,996 optimizer updates; the final 8,000-update checkpoint
is retained but not selected instead. Both arms use decimal heading transport.

For each live aircraft decision, the training reward subtracts the coefficient
multiplied by:

    0.5 * ((executed_heading - nominal_heading)^2 / 90^2
           + (executed_speed - nominal_speed)^2 / 2^2)

The nominal heading is the composed route-reference plus residual command,
measured in degrees; speed is the normalized command in [-1, 1]. The cost is
bounded in [0, 1] and is zero when the filter does not intervene. Terminal
transitions receive the same treatment. This changes only returned training
rewards. The original total_reward info and all eight physical scoring metrics
remain unchanged. No extra action, route change or simulator step is introduced.

The full training description requires both experiment-configuration.json and
training/config.json. Generic resume would omit the experimental reward wrapper
and must not be used for these models. No replay continuation is planned here.

## Validation before training

Eleven focused tests passed, covering normalization, invalid commands and penalty
coefficients, reconstruction of the composed command, terminal rewards,
preservation of scoring info, zero coefficient and rejection of stale diagnostics.
The test source and XML hashes are fixed in the protocol.

Before either training arm starts, a frozen policy runs the same two development
worlds with coefficients zero and one. The required checks are identical scenario
specifications, initial metrics, decision-boundary actions, observations and
traffic states, plus all nine completed scoring records. The added training
penalty must be nonzero. This checks separation of reward feedback from simulator
behavior; it does not establish how a trained policy will behave.

Every stage uses a fresh extraction of the fixed candidate archive, verified before
and after execution. The new wrapper and runner have separate recorded hashes.
Original assigned goals and scenario specifications are checked during evaluation.

## Evaluation and advancement rule

Each trained model is evaluated without the training penalty on twenty seed-2026
MA worlds at five-second decision intervals. Comparators are the same-seed
unshaped learner, the frozen primary learner and the matching classical controller.
All metrics and failed aircraft are retained. The initial test is one training
seed on reused development worlds, not a new untouched test.

Expand to 200 development worlds only if the feedback-trained model:

- has no lower arrival rate than both the same-seed control and classical control;
- improves clean completion by at least two percentage points over both;
- does not increase mean intrusion, restricted-area or outside-sector exposure;
- increases mean flight time by no more than 5% over either comparator.

This is a fixed screening rule, not a significance test or automatic deployment
promotion. The experiment does not use seed 2027 or official seed 42 for tuning.
A failure to advance remains a reported result. Reduced filter use alone does
not count as success if physical outcomes regress.

## Execution status

The sequential runner was started on 15 September 2026 after the unit checks and
resource preflight passed. Validation is running; no training or performance
result is claimed yet. State, logs and completed stages are retained under
runs/intervention-reward-v1/execution/. Runner SHA-256:
4438a6153952acb232e9d8741937175870c266ad29e38176f83cea8025680faa.


### Priority change: deferred

The fixed-policy validation completed. The original coordinator failed before
training because of source-provenance handling; its retry and all logs remain.
When RL priorities changed, only the retry coordinator was stopped. Its already
running unshaped training child finished: 25,000 counted transitions, 13,875 live
and 11,125 padding. Queued evaluation and shaped-training stages are deferred.
No further classical/controller refinement follows from this experiment now.
