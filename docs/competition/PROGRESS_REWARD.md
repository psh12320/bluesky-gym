# PPO progress-feedback experiment

The 101,244-live-transition direct PPO pilot failed to improve navigation: arrival
fell from 4% at initialization to 1% on twenty paired development worlds, while
outside-sector time increased by 602.58 seconds per aircraft. The short pilot and
all its negative results remain fixed. A separate million-transition raw-reward
run tests the budget limitation and reproduces the earlier policy tensors exactly
at both initialization and 101,244 live transitions.

This experiment changes only the reward supplied to learning. The actor still
chooses heading and speed directly, with no route guidance or conflict filter.
It tests earlier navigation feedback before adding critic complexity.

## Registered comparison

- PPO, local critic, two workers, seed 49900 and 100,000 live transitions.
- Same 256-step rollouts, 1,024-sample batches, ten epochs and architecture.
- Shape scale 100; original reward weights and discount remain fixed.
- Evaluate initial and trained policies on twenty seed-20260 development worlds,
  with shaping disabled. Preserve all nine original metrics and scenario hashes.
- Initial advancement screen: arrival improves by at least ten percentage points
  over the matching untrained policy and outside-sector time does not increase.
  This is a development screen, not evidence of competitive performance.
- Unseen streams 20301 and 20302 remain unused.

The immutable protocol is runs/progress-reward-v1/protocol.json, SHA-256
 efb1e899f09ccd2e0195d416c5a9a8982e34b0e16c7053824995e42b73510041.

## Learning reward

Let d be distance in kilometres to the original assigned waypoint and u be the
remaining fraction of the 3,000-second task. The potential is

    Phi = -100 * min(max(d - 5, 0), 1500) / 150 * u
    shaping bonus = gamma * Phi(next) - Phi(current)

The potential is zero at both arrival and the actual task deadline. The time
factor tapers the potential before a timeout, avoiding a large final correction.
A rollout boundary is not a task terminal and retains its next potential and value
bootstrap. The same remaining-time feature is present in raw and shaped policies.

Over a completed discounted trajectory, the shaping bonuses sum to -Phi(initial).
This is an arithmetic property of the chosen transformation; approximate PPO can
still fail. The construction follows
[Grzes, AAMAS 2017](https://aamas.csc.liv.ac.uk/Proceedings/aamas2017/pdfs/p565.pdf),
with related multi-agent theory in
[Lu, Schwartz and Givigi](https://arxiv.org/abs/1401.3907).

The wrapper does not change actions, observations, original rewards stored in
scoring info, metrics, termination or scenario generation. The learner receives
the shaped reward. learning.csv records both shaped and native returns, arrival
and current optimization diagnostics; training-aircraft.csv retains native scores.
training-returns.csv records the native/shaped difference for each completed
training aircraft. Evaluation always disables shaping.

## Verification before training

Twenty-seven focused tests passed. Real paired runs then used identical actions
with scale zero and scale 100. Two goal-controller worlds produced twenty arrivals;
one circling world produced ten actual timeouts. All observations, original metrics
and terminal flags matched. The goal-controller records also matched the earlier
fixed adapter fixture. A total of 2,394 padded slots was checked, and 9,506 live
steps received nonzero shaping feedback. The largest discounted-return identity
error was 1.85e-13. These checks establish implementation behavior, not policy quality.

Validation results: runs/progress-reward-v1/validation/summary.json.
The shaped training run completed 100,217 live transitions and 2,183 padded slots,
with twenty rollouts and 1,000 optimizer steps. Both actor and critic parameters
changed and remained finite. Checkpoint hashes and all 162 completed native/learning
return records were independently verified. Outputs are under
runs/progress-reward-v1/train. Initial and trained checkpoints are now being
evaluated on twenty development worlds with shaping disabled. No performance
result is available yet. The running raw million-step PPO uses its
own extracted source; it and the original official evaluation remain unchanged.
The existing cluster-v1 archive also remains unchanged for the initial cluster setup.

## Completed development result

The twenty-world evaluations are complete. Initial and trained arrival rates were
4% and 8%; the initial evaluation CSV exactly reproduces the earlier raw-PPO
initial evaluation. Mean intrusion time rose from 100.64 to 1,338.60 seconds,
restricted-area time rose from 291.375 to 695.325 seconds, and outside-sector time
fell from 1,416.745 to 22.69 seconds. Native return deteriorated from -430.78 to
-1,614.46. These metrics use original scoring with shaping disabled.

The preset ten-percentage-point arrival improvement screen failed. This recipe
is not being extended. Paired-world confidence intervals and all metrics are in
runs/progress-reward-v1/comparison-dev20/comparison.json. This is one seed on
development scenarios, not evidence of generalization.
