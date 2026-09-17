# RL development priorities: 16-24 September 2026

The user revised the success criterion: a working reinforcement-learning system
with a defensible learning contribution and strong performance. Classical planning
scores alone do not satisfy either the course objective or our project objective.
This plan supersedes the earlier emphasis on controller refinements and final-report
preparation. Existing evaluations finish on their original source and models.

## Fixed comparisons

- Classical benchmark: goal controller, public_route_choice_interval5, speed action
  +1, existing route guidance and joint traffic/static filtering, five-second
  decisions and decimal heading transport. Preserve its implementation in the
  decimal-heading-v1 candidate archive. No further classical refinements.
- Historical SAC: preserve the selected MA model, SHA-256
  961f525f11bec7a0f417933d209230ca99c5b4b8c00d883233a006b8ee38c4bb,
  and every existing evaluation. Its smaller training budget must remain visible.
- Shared PPO: all aircraft share a local-observation actor and local critic.
- MAPPO: the same local actor, with a critic receiving joint aircraft observations,
  alive flags and target-aircraft identity during training. A command filter is
  not a centralized critic. MAPPO is an empirical comparison, not an assumed winner.

The architectural distinction follows the centralized-training setting in
[Yu et al., 2022](https://arxiv.org/abs/2103.01955) and the
[official MAPPO implementation](https://github.com/marlbenchmark/on-policy).
The new feedforward implementation uses SB3 PPO optimization and custom actor/critic
information separation, rollout masking and BlueSky world workers. It is not a
claim to reproduce every feature or benchmark setting of the authors' repository.

## What has actually been trained

Current completed on-policy training (one development seed, 49900):

| Run | Live aircraft transitions | Optimizer updates | Development result |
| --- | ---: | ---: | --- |
| Raw shared PPO | 1,000,371 | 10,000 | 5.5% arrival at final checkpoint; full four-point curve recorded. |
| Genuine centralized-critic MAPPO | 100,744 | 1,000 | 0% arrival; raw-reward recipe not advanced. |
| PPO, progress shaping only | 100,217 | 1,000 | 8% arrival; major increase in intrusion time. |
| PPO, reward scale 0.01 only | 101,160 | 1,000 | 8% arrival; increased intrusion time. |
| PPO, shaping plus scaling | 104,783 | 1,050 | 18% arrival; intrusion time increased. Screen failed. |
| PPO, neutral initial mean and std 0.05 | 101,525 | 897 | Arrival fell from 100% to 4%; screen failed. |
| PPO, goal-offset action prior | 103,226 | 1,093 | 100% arrival retained; clean completion fell 19.5% to 18%; screen failed. |
| PPO, goal offset and reward scale 0.01 | 948,398 | 10,156 | Wall-limited partial run; clean completion 19.5% to 23.5%, safety times improved; below the registered +5-point screen. |
| MAPPO, goal offset and reward scale 0.01 | 708,127 | 7,729 | Wall-limited partial run; 21.5% clean completion, increased safety times; screen failed. |
| PPO, route guidance on / filter off | 101,144 | 1,091 | Clean completion 26% to 23.5%; shorter flights and less conflict time, but more restricted-area time; screen failed. |

The raw-PPO short run (101,244 live transitions, 1,000 updates) exactly matches
the million-step run at its corresponding checkpoint. These runs are not two
independent seeds. Earlier 640-counted-step PPO and integration tests are pipeline
checks only. Shared SAC retains historical 50k pilots, 391,520 direct-control
counted transitions and a continuation reaching 594,480 counted transitions.
Legacy counted budgets include inactive slots and are not equivalent to live
experience. Directory names never replace completed counts.

The selected MA and SA-transfer residual SAC recipes each have three training
seeds at 25,000 counted transitions. Their live experience is roughly 14k-15.5k
transitions; remaining vector slots are padding. The selected models use the
7,996-update callback. Additional SAC pilots include SA-specific training, but
selected SA weights are transferred from MA training. Full details remain in
saved configurations and training_summary.json files.

The recent SAC correction-feedback experiment is deferred. Its fixed-policy
validation passed. Its already-started unshaped training child was allowed to
finish; queued evaluation and penalized-training stages were stopped by cancelling
only that experiment's coordinator. The original attempt, retry, models and logs
remain available. This does not interrupt the existing scoring evaluations.

## Latest completed checkpoint evaluations

The goal-offset/reward-scale-0.01 runs stopped at their wall limits and saved valid
final checkpoints: PPO at 948,398 live transitions / 10,156 optimizer steps, MAPPO
at 708,127 / 7,729. Both pass the source, model, optimizer, accounting and information-
boundary audits. Neither completed the requested one-million-live-transition budget.

On twenty paired development worlds, the untrained goal-offset policy had 100%
arrival and 19.5% clean completion. Final PPO retains 100% arrival and raises clean
completion to 23.5% (change +4 percentage points; paired-world 95% interval +0.5 to
+7.5). Intrusion time decreases from 73.77 to 57.05 seconds per aircraft (change
-16.72; interval -25.02 to -8.56), and restricted-area time from 127.005 to 108.46
(change -18.545; interval -28.11 to -9.495). Flight time increases by 17.635 seconds.
This is an exploratory within-seed learning signal, not seed-robust evidence or
competitive performance. The registered +5-point clean-completion screen fails;
the fixed classical benchmark remains at 94.5% clean completion.

Final MAPPO retains 100% arrivals and reaches 21.5% clean completion. Relative to
its own initial policy, mean intrusion time increases by 10.83 seconds and
restricted-area time by 20.055 seconds; its registered screen also fails. Unequal
final budgets prohibit attributing the final-policy difference solely to algorithm.
The already-saved 700,844-step PPO and 701,204-step MAPPO checkpoints are selected
for a matched-budget comparison before their results are observed.

All nine metrics, paired-world intervals, source and checkpoint hashes, screens
and four-point curves are retained under `runs/goal-scaled-final-evaluations-v1`
and each training run's `comparison-final-dev20` and `curve-final-dev20` directories.
Bootstrap intervals quantify scenario variation only. The development search is
exploratory; three training seeds and unseen scenarios remain outstanding.

The guidance-on/filter-off PPO pilot completed 101,144 live transitions. Its exact
initial control scored 99.5% arrival, 26% clean completion, 114.07 seconds intrusion
time and 0.705 seconds restricted-area time. Final PPO scored 100% arrival and
23.5% clean completion, with 99.24 seconds intrusion time and 11.09 seconds
restricted-area time. Flight time fell by 32.38 seconds. The restricted-area-time
increase was 10.385 seconds (paired-world interval +0.105 to +27.785); this safety
trade-off fails the original screen. Both the 50k and final checkpoints, including
their negative results, remain in `runs/ppo-guidance-pilot-v1`.

Before completed learned-policy evaluations were available, two additional learner
seeds, 49920 and 49940, were committed under `runs/ppo-guidance-replication-v1`.
Together with 49900 they form a three-seed 100k pilot with disjoint simulator seed
pairs, identical training package bytes, and separate own-initial/final evaluations.
This does not replace the canonical larger-budget matrix or unseen scenarios.
The four-cell zero-experience support experiment continues sequentially to separate
route guidance and filtering before learning. High untrained navigation scores do
not count as learned gains. No classical implementation has been refined.

At the preselected approximately-700k matched checkpoints, MAPPO minus PPO was
+241.995 seconds flight time (paired-world interval +206.644 to +279.436), +24.59
seconds intrusion time (+5.73 to +45.981), and +43.345 seconds restricted-area time
(+29.950 to +57.710). Clean completion differed by +0.5 points (-3.5 to +5).
Training sources and initial actors were identical, while critic parameter counts
differed. This one-seed result supports continuing PPO development; it does not
establish a general algorithm ranking.

## Focused experiment and baseline design

Shared PPO and the first centralized-critic MAPPO pilot are trained and verified.
The goal-offset action pilot retained navigation but did not improve safety.
The matched pair of goal-offset PPO runs requested one million live transitions
at reward scales 1 and 0.01, but both processes ended without a final summary.
Their last logged experience was 443,077 / 443,993 live transitions; durable
checkpoints stop at 400,149 / 403,577. The cause is unknown. They are interrupted
partial runs, not completed million-step experiments. No exact restart is claimed.
Their initial weights match, and the native 103,226-step checkpoint exactly
reproduces the earlier pilot. The saved source, checkpoint hashes, log counters
and optimizer states passed a partial-artifact audit.

The registered saved 100k/300k points and both last durable checkpoints have now
been evaluated on the same twenty development worlds. At 403,577 live transitions,
scaled PPO retained 100% arrival and raised clean completion from 19.5% to 21.5%.
Intrusion time fell 73.77 to 63.23 seconds and restricted-area time fell 127.005 to
118.595 seconds. Only the restricted-area change excluded zero in the paired-world
95% interval; the five-point clean-completion screen still failed. Native-scale
PPO became substantially slower and accumulated more safety exposure.

The million-step points remain missing. A fresh scaled PPO attempt is running in
runs/goal-offset-scaled-million-retry-v1/train with the same initial full policy
weights and numerical settings. This repeats the interrupted seed; it is not an
independent replication or exact optimizer/simulator continuation. The new frozen
source saves completed training aircraft after every rollout. The original
100,699- and 301,272-step policies, optimizer states and numerical learning logs
have been reproduced exactly, so their existing evaluations can be reused. All
prior runs remain unchanged. Success still requires strong RL performance, multiple seeds and unseen
scenarios; no checkpoint in this comparison has passed the advancement screen.

A matched goal-offset, reward-scale-0.01 MAPPO run is also active in
runs/mappo-goal-scaled-million-v1/train. It uses the same seed, two workers,
one-million-live-transition budget and copied zero-experience PPO actor. All
32,900 actor parameters and the first 20 completed training-aircraft trajectories
match. Its joint-information critic has 423,425 parameters versus PPO's 98,305;
this is not a parameter-count-matched information-only ablation. Evaluate its
own initial checkpoint and retain all registered curve points. No MAPPO
performance advantage is assumed.

The MAPPO initial, 100,275- and 303,095-transition evaluations are complete on
20 paired development worlds. Arrival is 100% at all three points; clean completion
is 19.5%, 18.5% and 21%. At 303,095 transitions, intrusion time is 69.66 seconds
and restricted-area time is 123.14 seconds, versus 73.77 and 127.005 before
learning. Their paired changes have 95% intervals of [-11.60, 3.26] and
[-9.57, 1.69] seconds, so these learning gains remain uncertain. The clean change
is +1.5 percentage points with interval [-1, 4]; the registered advancement screen
fails. Matched PPO at 301,272 transitions has 22% clean completion, 62.41 seconds
intrusion and 116.40 seconds restricted-area exposure. No algorithm ranking across
training seeds is established. The validated comparisons and three-point MAPPO
curve are under runs/mappo-goal-scaled-million-v1/; both million-step runs continue.

A fixed SAC reference comparison is registered under runs/sac-reference-dev20-v1/.
It uses the preserved decimal-heading deployment and requires two physical checks
before its 20-world seed-20260 evaluation: zero residual must reproduce the
classical reference, and the learned actor must reproduce its saved original-harness
trajectories. Its different supports, action mapping and smaller historical budget
remain explicit; it is a system reference, not an equal-budget algorithm ranking.

The SAC reference is now complete. Both physical parity checks passed: the
zero-residual reference matched all physical metrics (reward difference at most
1.35e-9), and the learned policy exactly reproduced all nine saved harness metrics.
All twenty new scenario identities match the PPO/classical development stream.
SAC has 99.5% arrival and 95.5% clean completion versus 99.5% and 94.5% for the
classical reference. The clean difference is +1 percentage point with paired-world
95% interval [-2, 5]; flight time differs by -14 seconds with interval [-60.41,
31.33]. Intrusion time is 1.29 versus 2.51 seconds, but the difference interval
[-3.53, 0] still reaches zero. Whole-world clean completion is 70% versus 80%.
These one-seed development results do not establish a clear learning advantage.
The saved SAC model itself confirms 25,000 counted transitions and 7,996 optimizer
updates; these counted transitions include inactive slots. The reference model
and deployment remain unchanged, with no additional SAC training.

The following baseline design and canonical multi-seed comparison remain in force.

The raw baseline chooses heading and speed directly. The goal-offset variant
explicitly tracks the observed bearing plus a learned heading offset. Optional
guidance changes that bearing; its initial contribution is measured separately
from learning. The action mapping is recorded for every experiment. Every actor receives its standard observation plus remaining-time
fraction. The actual 3000-second deadline ends the task return; rollout cuts still
bootstrap. Dead aircraft contribute no policy loss, value loss, entropy loss or
advantage normalization. Report live experience separately from padded slots.

Canonical initial budget: 1,000,000 live aircraft transitions per run, ending at a
complete rollout and reporting exact overshoot. Three training seeds are 50100,
50200 and 50300. Eight simulator workers use seed offsets 0, 10, ..., 70. Each
algorithm/configuration uses the same worker count and seed assignment. A local
pilot with a different worker count is development evidence, not a canonical seed
replication. Larger budgets require measured learning curves and available time.

Use the full 2-by-2 experiment for each algorithm:

| Guidance | Joint traffic/static filter | Question |
| --- | --- | --- |
| Off | Off | What can the policy learn directly? |
| On | Off | What does guidance contribute? |
| Off | On | What does filtering contribute? |
| On | On | Does their combination help learning and final performance? |

Keep reward weights, decision interval, observation dimensions, actor architecture,
training budget and evaluation procedure fixed across this ablation. Compare each
trained actor against its own untrained checkpoint as well as the fixed classical
benchmark and historical SAC. The SAC comparison is not an equal-budget algorithm
ranking; disclose that limit.

## Evaluation and curves

Record live/count transitions, completed worlds/aircraft, optimizer steps, policy
epochs, throughput, training returns and complete aircraft metrics.
For future runs, the development trainer also records critic explained variance,
value mean-squared error and return variance over live rollout samples only, plus
the fraction of live Gaussian action samples clipped by the action box. These use
value predictions collected before optimization and are diagnostics, not evaluation
scores. The standard SB3 explained-variance field includes padded aircraft slots;
do not interpret that historical field as live-aircraft critic fit. Running frozen
experiments and their recorded statistics remain unchanged.
The diagnostic update passes all 119 RL unit tests. A matched real PPO rollout
with 2,270 live and 290 padded transitions produced identical initial/final policy
weights, optimizer states, original learning fields and completed-aircraft records
under the old and new source. Evidence is in
runs/live-diagnostics-smoke-v1/numerical-parity.json. This is a pipeline check,
not policy-quality evidence. Save initial
weights and checkpoints near 100k, 300k and 1m live transitions; actual saved counts
are authoritative. Evaluate these checkpoints to obtain performance learning curves.

Use development stream 20260 for model/configuration choices. Reserve streams
20301 and 20302 for final unseen-scenario comparisons, with 200 worlds per stream.
Do not inspect their scenarios or outcomes during development. Seeds 2027 and 42
are not tuning streams. Official seed-42 scoring follows candidate selection and
is not used to choose among models. Existing SAC outcomes on 42 remain historical.

Report all physical metrics, event counts, arrival, clean completion, all-aircraft
clean completion and failures. Pair controllers by scenario and resample entire
worlds for scenario uncertainty. Show all three training seeds and their variation;
do not count repeated worlds as independent samples or hide unsuccessful seeds.
A learned improvement must survive comparison with the matching support configuration
and must not buy low exposure by abandoning arrivals.

## Remaining sprint

| Dates | Primary work and exit evidence |
| --- | --- |
| 16 Sep | PPO correctness and real training pilot; obtain cluster allocation details and stage reproducible jobs. |
| 17-18 Sep | Three-seed PPO baselines and guidance/filter ablations; training and evaluation curves. |
| 19-20 Sep | Matched MAPPO training and critic-information checks; compare with PPO without assuming an advantage. |
| 21-22 Sep | Finish useful replications, analyze learning contribution, freeze candidates, run reserved unseen streams. |
| 23-24 Sep | Final performance/reproduction checks and an evidence-based competition/course decision. Report preparation resumes after the RL evidence exists. |

## Cluster allocation and execution

The user supplied account allusers, QoS normal, Python 3.12.3 and a writable
~/cs4246-rl directory. The gpu partition has a three-hour limit; gpu-long has a
three-day limit but its displayed resources do not include H200. No module command
is available. The prepared job uses gpu, one gpu:h200-141:1, 16 CPUs, 64 GB RAM and
eight simulator processes. It stops after a complete rollout at about 2h45m to
save its current model before the allocation ends. This is not an exact-resume
implementation; a partial budget must be reported as partial.

Each BlueSky world uses its own CPU process and runtime directory. The upstream
environment indirectly imports Torch through bluesky_gym.utils.logger, so worker
memory includes that overhead. The parent batches policy inference/optimization
on the GPU. Measure transitions per second and actual memory before increasing
workers; H200 capacity alone does not determine throughput. Training and package
setup run inside the allocation, never on the login node.

No university job has yet been submitted by this task. Passwords remain in the
user's SSH/SCP prompt. Package access from compute nodes, the NVIDIA driver,
allocation acceptance and storage quota still require the prepared setup job.
[CLUSTER_RL.md](CLUSTER_RL.md) contains exact transfer and submission commands.

## Current local execution evidence

The new PPO integration run completed 12,800 live transitions, ten rollouts and
200 optimizer steps with both actor and critic parameters changed. Its independent
two-world checkpoint reload produced no arrivals; this is execution evidence only.
The separate real adapter check completed twenty aircraft and exercised 2,394
inactive slots, with correct terminal and zero-reward padding behavior. Thirteen
focused actor/critic and masking tests passed. Worker runtime files are isolated.

The direct PPO pilot (seed 49900, two workers, ten PPO epochs, 256-step rollouts)
completed 101,244 live and 1,156 padded transitions, twenty rollouts, 1,000 optimizer
steps and sixteen complete worlds in 292.985 seconds. It uses no route guidance or
conflict filter. Its initial and trained checkpoints are being compared on the
same twenty development worlds under runs/ppo-direct-pilot-v1. It is not one of
the three canonical eight-worker cluster replications. No MAPPO training has started.


### Completed pilot evaluation and next budget check

The paired twenty-world development evaluation completed (200 aircraft per policy,
seed 20260). Direct PPO arrival was 4% at initialization and 1% after 101,244 live
transitions. Clean completion was 0% and 0.5%. The fixed supported classical system
reached 99.5% arrival and 94.5% clean completion. Initial/trained PPO had identical
support settings and scenario fingerprints; the classical system includes guidance
and filtering, so its gap is a system comparison rather than an algorithm-only one.

The pilot has not demonstrated useful learning. Mean outside-sector time increased
by 602.58 seconds, while restricted-area time fell by 150.245 seconds. Reward alone
would conceal this undesirable tradeoff. Paired-world intervals and all metrics
are preserved in runs/ppo-direct-pilot-v1/comparison-dev20/comparison.json; this is
one training seed on development worlds, not final generalization evidence.

A one-million-live-transition direct PPO budget check is now running at
runs/ppo-direct-million-v1 using an extracted, verified source snapshot, seed 49900
and two workers. It starts from the same initialization and protocol as the short
pilot; its first checkpoint can be checked against that pilot. It is a local budget
experiment, not an eight-worker canonical cluster replication. Completion and
performance are pending. MAPPO remains untrained. The immutable cluster-v1 archive
retains the status known when it was packaged; current status is recorded here.


### Controlled progress-reward pilot

Potential-based waypoint-progress feedback was added as an optional training
transformation. It passed 27 focused tests and three real paired worlds covering
20 arrivals and 10 timeouts, with unchanged scoring records and a maximum
return-identity error of 1.85e-13. The matched PPO pilot completed 100,217 live
transitions and 1,000 optimizer steps from exactly the baseline's initial policy
state. Native and shaped returns are logged separately. Its paired twenty-world
performance evaluation is running with shaping disabled; improvement is not yet
claimed. See PROGRESS_REWARD.md and runs/progress-reward-v1/ for the fixed protocol,
validation and training audit. No classical refinement or MAPPO training was added.

### Shaping outcome and centralized-critic pilot

The progress-feedback evaluation is complete: arrival 4% to 8%, intrusion time
100.64 to 1,338.60 seconds, restricted-area time 291.375 to 695.325 seconds.
The preset advancement screen failed. This shaping recipe will not be extended.

The first genuine MAPPO pilot is running under runs/mappo-direct-pilot-v1. It
uses the original reward, no guidance/filtering, and the same 49900 seed, two
workers, 100k live budget and PPO hyperparameters. Its 32,900 actor parameters
are copied exactly from the zero-experience PPO checkpoint; no learned weights
are transferred. The centralized critic has 423,425 parameters versus 98,305
for the local critic. Both joint information and increased critic capacity must
be acknowledged when interpreting this comparison. The actor remains local at
training and inference. The helper also reconstructs the initial learner RNG,
while simulator seeds remain unchanged. All 29 RL correctness checks passed
before launch. No MAPPO performance result is available yet.

An optional fixed positive reward-scale wrapper has separately passed seven
checks covering episode returns, original metrics, unchanged actions and inactive
slots. It is not enabled in the running experiments. Saved optimizer moments
suggest investigating value-target scale, but do not establish the cause of
poor learning. No reward-scale training has been launched.

### Completed MAPPO training pilot

MAPPO completed 100,744 live transitions, 1,656 padded slots, twenty rollouts,
1,000 optimizer steps and sixteen complete worlds (160 aircraft) in 454.02 seconds.
The run verified finite updates to both actor and critic and unchanged source.
Its model SHA-256 is fc6fe1eefe31ff7dbfbfe608d380a78f655b9ca386595bf64b800aab07790d5d.
Independent development evaluation is the next step; training completion is not
evidence that MAPPO improved performance.

### Completed million-transition PPO budget check

The raw PPO run completed 1,000,371 live transitions, 23,629 padded slots,
200 rollouts and 10,000 optimizer updates in 3,639.56 seconds. It completed
170 worlds and 1,701 aircraft. All twelve checkpoints, 200 learning-log rows,
archived source, serialized optimizer counters and actor/critic dependencies
passed the independent audit. Model SHA-256:
8feb479b0aa5f226e7b0429592fc4b18ec87ef61041c096b492e882debe45379.

Its registered physical learning curve uses 0, 101,244, 302,514 and 1,000,371
active transitions. The first two evaluations can be reused: complete policy
tensor identities, training settings and evaluation source match exactly.
Later evaluations remain pending. Training completion does not establish
performance.

The reward-scale pilot is now running from immutable source version 2, with
scale 0.01 and shaping, guidance and filtering disabled. Its initial actor,
critic and log-standard-deviation tensors exactly match the original PPO run.
The registered protocol is runs/ppo-reward-scale-v1/protocol.json.

### Completed MAPPO development result

On the paired twenty-world stream, the MAPPO initial policy exactly reproduced
the raw-PPO initial physical records. After 100,744 live transitions, MAPPO had
zero arrivals among 200 aircraft, versus 4% initially and 1% for raw PPO at its
100k budget. MAPPO native reward improved from -430.78 to -265.08, while every
aircraft timed out. Mean outside-sector time fell from 1,416.745 to 296.775 seconds;
intrusion time rose from 100.64 to 138.81 seconds. Improved scalar reward is
therefore not evidence of successful navigation.

All metrics and paired-world intervals are retained in
runs/mappo-direct-pilot-v1/comparison-dev20 and ppo-comparison-dev20.json.
This raw-reward MAPPO recipe is not being scaled up now. The comparison has only
one training seed; it does not show that MAPPO is generally worse than PPO.
The development controls now pass 41 focused checks.


### Development results, 16 September: budget and optimizer controls

The registered raw-PPO curve is complete: arrival rates at 0 / 101,244 /
302,514 / 1,000,371 live transitions were 4% / 1% / 4.5% / 5.5%.
The final arrival gain was 1.5 percentage points, with a paired-world 95%
interval of -3 to +6 points. Restricted-area time fell by 231.46 seconds,
but navigation remained poor. This shows some changed behavior, not a strong
competitive policy. All metrics and the unsmoothed physical learning curve
are in runs/ppo-direct-million-v1/curve-dev20 and comparison-dev20.

Scaling learning rewards by 0.01 alone completed 101,160 live transitions.
Its twenty-world arrival rate was 8%, clean completion 4%, intrusion time
295.43 seconds, restricted-area time 184.025 seconds and outside-sector time
740.13 seconds. The advancement screen failed: arrival improved only 4 points
and intrusion time increased by 194.79 seconds. No extension is justified as a
successful navigation recipe. The combined shaping/scaling cell completed
104,783 live transitions, 1,050 optimizer updates; its evaluation is pending.

Neutral mean initialization with standard deviation 0.05 completed 101,525
live transitions, 897 optimizer steps and 181 epochs (KL stopping reduced the
actual update count). The exact untrained initial policy achieved 100% arrivals
but only 19.5% clean completion. Intrusion time was 73.44 seconds and restricted
area time 126.975 seconds. These are untrained-control results. The trained
policy's evaluation is pending; learned gains must be measured against this
reference rather than the original random initial policy.

A goal-relative heading-offset representation is being validated independently
of route guidance and conflict filtering. It must have its own untrained
reference. All existing runs continue from immutable source snapshots; no
classical controller or selected SAC checkpoint was changed.


### Completed reward-combination and initialization evaluations

Combining potential shaping and reward scaling raised arrival from 4% to 18%
(+14 points, paired-world interval +7 to +21), but intrusion time rose by
162.67 seconds (interval +42.768 to +291.591). Restricted-area time fell by
182.555 seconds and outside-sector time by 379.62 seconds. The registered screen
failed because safety did not improve. All four reward-transformation cells are
now trained/evaluated; no recipe met the navigation-and-safety goal. The original
shaping-only evaluation used an earlier evaluator source revision, unlike the
other three cells. Its physical wrapper validation is retained; no exact-source
four-cell interaction estimate is claimed here.

Neutral mean and std 0.05 with direct heading increments fell from 100% to 4%
arrival and from 19.5% to 2% clean completion after 101,525 live transitions.
Intrusion time increased by 59.09 seconds, restricted-area time by 22.74 seconds
and outside-sector time by 1,937.42 seconds. This initialization alone failed.
Full comparisons and registered-screen checks are retained in each run directory.

The goal-offset pilot is a navigation-prior experiment. Its zero-experience
reference is now being evaluated separately. The frozen trainer's legacy
automatic_residual_controller field is misleading for this mode; see
runs/goal-offset-v1/action-semantics-clarification.json. The explicit action_reference
and action_mapping fields are correct, and the original configuration is preserved.


### Goal-offset pilot completed; next budget/scale experiment

Goal-offset PPO completed 103,226 live transitions, 24,774 padded slots, 1,093
optimizer steps and 243 epochs. Its three-point development curve retained
100% arrival at every checkpoint, while clean completion changed 19.5% -> 20% ->
18%. Final intrusion and restricted-area times increased by 1.76 and 4.4 seconds.
This fails the registered learning screen: the prior preserves navigation, while
RL safety gains remain unproven. Models and all negative results are preserved.

Proceed with a matched budget/scale experiment, at one million live transitions
for each of reward scales 1 and 0.01. The native-scale run must reproduce the
original pilot prefix; scale is the only recipe difference between the two new
runs. Evaluate near 100k, 300k and one million with the same initial reference
and physical evaluator. This is investigation of learning at larger budgets,
not a relaxed success criterion. Three canonical seeds and unseen scenarios
remain required before a competitive learned contribution can be claimed.


### Reproducibility and cluster matrix ready for the next stage

The native longer-budget run exactly reproduced all thirteen policy tensors at
103,226 live transitions and all twenty-five earlier learning rows, excluding
wall time. The previous development evaluation can therefore be reused under
the same frozen goal-offset evaluator. Both million-step runs remain active.

The new three-seed/four-support matrix has passed 67 total RL checks, shell syntax
validation and all 24 PPO/MAPPO CLI dry runs. No matrix training or Slurm job was
launched. MAPPO rows inherit the PPO recipe, require matching completed PPO runs
and zero-experience actor references, and run a full checkpoint audit before
training. Partial budgets and altered sources are rejected. See MATRIX_RL.md and
runs/matrix-protocol-validation-v1.json. This prepares the requested experiment
structure; it is not additional training or performance evidence.

## Three-seed guidance pilot and separate support controls completed

All three guidance-on/filter-off PPO seeds completed their registered 100k live-transition pilots: 101,144 / 101,336 / 101,894 transitions for seeds 49900 / 49920 / 49940. Each was audited and evaluated against its own initial policy on twenty paired development worlds. Clean completion was initially 26% and finished at 23.5%, 24.5% and 25%. The mean learned change is -1.667 percentage points (sample standard deviation 0.764 points). All three registered advancement screens failed. Mean intrusion time increased by 9.827 seconds; mean restricted-area time increased by 3.828 seconds. No seed is discarded. Full evidence: runs/ppo-guidance-replication-v1/three-seed-results.json.

The zero-experience support controls now cover all four guidance/filter cells. Clean completion is 19.5% with neither, 26% with guidance only, 83% with filtering only, and 97% with both. These are supporting-controller effects, not learned improvements. Their full native metrics and paired-world effects are in runs/initial-support-ablation-v1/support-effects.json.

Six newly registered filter-enabled PPO pilots use the same three seeds and 100k live budget, with guidance off/on separately. They run from the unchanged learning-source-v2 snapshot and evaluate their own initial/final policies. Their three-cell analysis also includes the completed guidance-only cohort and estimates conditional effects on learned changes. It is not the complete four-cell training matrix. Protocol: runs/ppo-filter-ablation-v1/protocol.json. The first guidance-on/filter-on seed has completed evaluation; the remaining registered runs continue.

Bootstrap recovery job 851402 passed. The selected cluster experiment is three unsupported PPO seeds at one million live transitions each, eight simulator worlds, goal-relative actions and reward scale 0.01. Its separate source bundle and launch instructions are preserved. Submission confirmation has been requested; no university training job ID has yet been supplied for this larger experiment.

An optional current-state conflict-prediction observation experiment is implemented separately; see CONFLICT_OBSERVATIONS.md. It changes the learned policy input and leaves controllers fixed. Unit and integration validation are recorded separately from performance evidence. The paired real-feature versus same-size zero-feature pilot is now running at three seeds, 100k live transitions per arm; no feature benefit has yet been established.

## Latest operational evidence

All three guidance-only pilots now have initial, approximately 50k and final physical learning curves. The combined figure and input provenance are in runs/ppo-guidance-replication-v1/three-seed-curves/. None improves clean completion at either measured training checkpoint.

The first guidance-on/filter-on PPO seed (49900) completed 102,580 live transitions and 1,140 optimizer steps. Its own initial policy achieved 99% arrivals and 97% clean completion; the trained policy achieved 98% and 94.5%. The clean-completion change is -2.5 percentage points (paired-world 95% interval -6.5 to +0.5). Flight time fell by 21.58 seconds, but mean intrusion and restricted-area times increased by 0.06 and 0.24 seconds. This single-seed result does not establish a learned advantage; the other registered seeds continue. Matching the classical mean of 94.5% clean completion does not show that learning helped, because this system was at 97% before training.

The capacity-matched observation implementation passed 133 tests plus simulator train/audit/reload and initial-behavior checks. Its first pilot arm (seed 50400, real features) completed 100,555 live transitions and 1,894 optimizer steps; evaluation is ongoing. No new university job ID has been received, and unseen evaluation streams remain unused.
