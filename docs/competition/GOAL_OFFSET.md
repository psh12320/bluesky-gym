# Goal-offset PPO experiment

Status: training, three-point development evaluation and the registered screen are complete. Navigation was retained; no learned safety improvement was established.

The heading action is a learned offset in [-90, +90] degrees from the current
actor-visible goal bearing. The wrapper subtracts observed drift, adds the
offset, wraps to [-180, +180] degrees and clips the resulting turn to the existing
45-degree command bound. Speed remains a learned increment. No obstacle path or
conflict avoidance is computed by this mapping. When separately enabled, the
existing route-guidance observation supplies its reference bearing and the
existing conflict filter processes the mapped command. Neither is changed.

This action representation supplies a navigation prior. A zero actor follows
the bearing and is not a learned controller. Its initial checkpoint must be
evaluated under goal_offset semantics; direct-heading evaluations cannot be
reused merely because parameter tensors match. Evaluation and comparison tools
record and check action_reference. The fixed classical benchmark always uses
direct commands. No new source is overlaid on a running evaluation.

The first proposed pilot changes only action_reference relative to the completed
neutral-mean, 0.05 exploration PPO control: seed 49900, two worlds, 100k live
transitions, 256-step rollouts, batch 1024, ten epochs, native public_weights,
no shaping, reward scale 1, no route guidance or conflict filtering. Final versus
initial performance tests learning within this representation. Comparing with
the neutral direct-action control tests the representation change; numerical
heading noise corresponds to different physical actions and must be acknowledged.
The mapping's benefit alone cannot establish a learning contribution.

Before training, validate all four guidance/filter combinations against identical
flights made by the frozen bearing tracker. Retain native observations, rewards,
termination flags and all nine scoring metrics; compare guided/filtered output
with the existing physical adapter fixture. Unit tests cover turning direction,
wraparound, zero-offset equivalence, updated observations and invalid actions.

Use 20 development worlds from seed 20260. The provisional advancement screen
requires at least 95% arrivals, no more than a two-point loss from the untrained
reference, at least a five-point clean-completion improvement and no increase
in mean intrusion, restricted-area or outside-sector time. This is triage from
one training seed, not statistical proof. Preserve failures; retain the two
unseen streams for the selected multi-seed experiment.

Validation retained under runs/goal-offset-v1/validation: 40 paired aircraft,
888 decision pairs and 2,322 padded slots. Observations, rewards, termination flags
and all metrics were exactly identical under equivalent commands. The guided and
filtered case also matched the preserved adapter fixture.


## Completed development result

At 0 / 50,378 / 103,226 live transitions, arrival remained 100% and clean
completion was 19.5% / 20% / 18%. Intrusion time was 73.77 / 72.89 / 75.53 seconds;
restricted-area time was 127.005 / 130.665 / 131.405 seconds. Outside-sector time
remained zero. Final flight time increased from 857.2 to 890.265 seconds.
The registered learning screen failed. The navigation prior accounts for the high
arrival rate; training has not yet supplied a defensible safety gain. All three
points use the same frozen evaluator and paired development scenarios. The curve
and comparisons are retained under runs/goal-offset-v1.

The next investigation separates budget and optimization scale with matched
one-million-live-transition runs at reward scales 1 and 0.01. Their initial actor,
critic, seed, workers, exploration, action representation, support flags and
PPO settings must match. Checkpoints near 100k and one million separate scale
comparisons at a similar experience budget from learning within each run.
The native-scale run should exactly reproduce the original 103,226-step checkpoint
before extending it; verify tensor equality. These are diagnostic extensions of a
failed pilot, not promotion of a successful controller. No unseen stream is used.


## Interrupted longer-budget runs

Both million-step attempts ended without a final summary. Last durable native /
scaled checkpoints contain 400,149 / 403,577 live transitions; their last logged
counts are 443,077 / 443,993. Saved weights, optimizer counters and source archives
passed the partial-artifact audit, which does not imply completed training.
Evaluate the registered early checkpoints and both last durable endpoints on
stream 20260. Keep the missing million-step points explicit. The original runs
and stopped full MA scoring evaluation remain unchanged; no restart is implied.
