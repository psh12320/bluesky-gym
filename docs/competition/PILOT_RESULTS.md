# Local experiment results: 15 September 2026

Development comparisons use seed 2026, seeded once and then continued. Twenty-
scenario pilots and completed 200-scenario expansions are labeled separately.
Seed 2027 is now the frozen held-out stream; its outcomes are reported separately.
The selected SA model is entering the original seed-42 protocol. Earlier sections
record historical states and must not be read as current selection or run status.

## Current decision

The working system combines a learned residual controller, geometric route guidance
and a joint traffic/static command filter. It substantially improves safety over
the historical unfiltered learned baseline. Both feasibility candidates are now
frozen, using only completed seed-2026 development comparisons for selection.

The five-second MA expansion is complete on 200 scenarios / 2000 aircraft:

| Controller | Arrival | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) | Clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matching classical | 99.45% | 997.2730 | 1.767 | 0.309 | 0.003 | 95.8% |
| Frozen learned | 99.60% | 997.5775 | 2.624 | 0.391 | 0 | 94.9% |

The learner improves every objective point mean over the same learner at ten
seconds, with zero additional training. Against matching five-second classical
control, however, its arrival advantage is uncertain and it has more intrusion
and restricted-area exposure. This does not establish an overall learned advantage.
The paired MA held-out evaluation is complete on 200 seed-2027 scenarios. Both
controllers reach 99.35% of goals. Learned versus classical means are 995.1085
versus 994.0735 s flight, 2.671 versus 2.536 s intrusion, 0.900 versus 0.9275 s
restricted exposure, zero outside, and 93.9% versus 93.8% clean completion. Every
nontrivial paired interval includes zero, so an overall learned MA advantage is
not established. The learner recovers three classical misses and adds three.

The frozen arrival-reward-250 SA candidate has completed its first 200-scenario
held-out comparison. It reaches 200/200 goals versus classical 194/200: +3 percentage
points, with a paired 95% interval of +1 to +5.5 points. Mean flight is 1243.150 s
versus 1221.205 s, intrusion 17.670 s versus 18.635 s, restricted exposure 1.090 s
versus 0.925 s, outside zero versus 0.465 s, and clean completion 69% versus 68%.
The safety and flight differences remain uncertain. All six arrival recoveries,
fourteen clean recoveries and twelve clean losses are retained in the audit.
This supports a completion benefit on this sample, not universal superiority.

The SA candidate is now running the original 1000-scenario, seed-42 competition
harness unchanged. These will be local official-protocol results, not judge-verified
scores. The corrected MA deployment has started its full seed-42 run in
runs/official-decimal-v1/ma/, and corrected SA scoring is also running in
runs/official-decimal-v1/sa/. No further checkpoint selection or
controller tuning will use either held-out or official outcomes.

Source, model, configuration, complete CSV summaries and paired calculations for
the completed SA holdout are audited. All 89 regression tests passed before the
freeze; the recorder additionally reproduced all nine metrics in fifteen actual
scenario replays, including five with the frozen SA candidate. The original frozen controller files have not changed since those checks.
Historical development ablations and adverse results remain below and in the
preserved run folders; the held-out analysis is in HELDOUT_RESULTS.md.
SA seed 2902 completed its full replication with 98% arrivals, compared with
100% for the primary and 97% for classical control; clean completion is 68%,
matching classical. SA seed 2904 completed its full replication with 99%
arrivals and 68.5% clean completion. All three SA seeds are now complete. MA seed 2902 is running and MA seed 2904 has
started its declared training/evaluation sequence; see REPLICATIONS.md. A
heading-command formatting error was reproduced in one primary MA test scenario.
The opt-in correction is tested separately and is not enabled in the running
frozen comparisons; see HEADING_TRANSPORT.md.

The source-v17 archive contains the tested controller and recorder for the
SSH/Slurm cluster. Later report and held-out notes postdate that archive. No
university job has been submitted; authenticated allocation information is still
needed. All runs so far are local. The four-page report is a preserved development
working draft, described in REPORT.md. Competition readiness still needs the
full evaluations, broader replication and a final report with accurate attribution.

The 391,520-transition policy remains a fixed historical learned reference, including
its completed 200-scenario evaluation. All thirteen rows below use per-aircraft inference
on the same twenty development scenarios; times are seconds per aircraft.

| MA controller | Arrival | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100% | 838.230 | 61.450 | 112.715 | 0 | 20% |
| Public-reward SAC, 391,520 transitions | 99.5% | 1046.940 | 24.550 | 79.505 | 50.840 | 27.5% |
| Continuation, 594,480 transitions | 98% | 1083.370 | 21.360 | 63.660 | 56.960 | 30.5% |
| Same 391,520-transition weights + static filter | 97.5% | 1188.910 | 27.690 | 5.755 | 0 | 59.5% |
| Filter-trained continuation, 491,520 transitions | 97.5% | 1264.380 | 43.390 | 8.040 | 0 | 47% |
| Filter-trained final, 584,360 transitions | 98.5% | 1117.180 | 47.530 | 5.965 | 0 | 55.5% |
| Matched frozen-feature control, 491,520 transitions | 97.5% | 1155.870 | 32.760 | 5.880 | 0 | 52.5% |
| Matched predictive features, 491,520 transitions | 99% | 1110.260 | 45.530 | 9.535 | 0 | 48% |
| Original weights + joint correction v1 | 96.5% | 1227.325 | 0 | 5.035 | 0 | 89.5% |
| Original weights + route input v1 + joint correction | 97.5% | 1292.410 | 0.430 | 0.505 | 0 | 95.5% |
| Larger-network public learner, 347,540 transitions | 98% | 1148.145 | 37.470 | 58.620 | 23.590 | 28.5% |
| Route-residual SAC, 25,000 transitions | 100% | 1035.290 | 0.220 | 0 | 0 | 99% |
| Route-residual SAC, 100,000 transitions | 100% | 1104.205 | 0.850 | 0 | 0 | 98% |

Earlier direct-control, predictive-feature and static-filter experiments remain
available as ablations below. The residual candidate preserves the classical navigation reference. Its measured
efficiency gain against the zero-speed reference does not establish a gain over
the faster classical reference. It is retained separately from later runs, so
further training cannot overwrite it. The native SA 10k checkpoint loses an arrival and is not preferred to
the MA-to-SA transfer. Current candidate paths and detailed comparisons are recorded
in the latest section of this report.

Learned MA tables below use historical batched inference unless explicitly marked
per-aircraft. An original-harness integration check found that per-aircraft inference
changes the same checkpoint's trajectories and metrics. The development evaluator
now matches that call pattern; candidates must be re-evaluated before final selection.
Single-agent and model-free classical results are unaffected by this batching issue.

## Expanded reference evaluation: 200 scenarios

The original 391,520-transition checkpoint is fixed throughout this comparison.
Both controllers use development seed 2026, seeded once, with per-aircraft learned
inference. The 2000 aircraft records include 16 failed learned arrivals.

| Metric | Neutral | Retained learned policy |
| --- | ---: | ---: |
| Arrival | 100% | 99.2% |
| Flight time (s) | 851.722 | 1067.233 |
| Intrusion events | 0.864 | 0.415 |
| Intrusion time (s) | 65.183 | 23.465 |
| Restricted-area events | 0.805 | 0.639 |
| Restricted-area time (s) | 124.336 | 92.385 |
| Sector-exit events | 0 | 0.1735 |
| Outside-sector time (s) | 0 | 50.270 |
| Clean completion | 16.1% | 27.7% |
| Entirely clean scenarios | 0% | 0% |

Mean intrusion time falls 64.0%, with a paired difference of -41.718 s
[-46.832, -36.626]. Restricted-area time falls 25.7%, or -31.951 s
[-37.978, -26.041]. Clean completion rises 11.6 percentage points [9.2, 14.0].
Arrival drops 0.8 points [-1.2, -0.45], flight time increases 215.511 s
[201.700, 230.406], and outside-sector time increases 50.270 s [43.589, 57.298].
These are pointwise 95% paired scenario-bootstrap intervals; they do not include
variation across training seeds. The retained policy does not meet the internal gates.

All nine metrics for every aircraft in the first twenty learned scenarios exactly
match the earlier evaluation. The additional 180 scenarios, considered separately from that
prefix, retain the same pattern: 99.167% arrival, 23.344 s intrusion versus 65.598 s
neutral, and 93.816 s restricted-area time versus 125.627 s neutral. This is an expanded
development check, not the reserved finalist stream or official test. The learned
intrusion-time 99th percentile is 200.01 s and its maximum is 471 s; maximum outside
sector time is 1288 s. Severe outcomes remain relevant despite improved averages.

Evidence: runs/neutral-ma-200,
runs/sac-ma-public-600k-seed1400/validation-individual-200,
runs/compare-neutral-public-391k-200scenarios.json, and
runs/public-391k-200-audit/result.json. The model hash remains
b4c7a671dbd0fae8146522fa8278f19f4119c859caf2e1b07fe0bb56eae45093.

## Historical early multi-agent pilots

| Metric | Neutral | SAC: direct turns | SAC: goal-relative turns |
| --- | ---: | ---: | ---: |
| Completion | 100% | 36% | 97.5% |
| Mean flight time (s) | 838.230 | 2336.125 | 1728.755 |
| Intrusion events | 0.820 | 6.260 | 0.710 |
| Intrusion time (s) | 61.450 | 389.940 | 79.800 |
| Restricted-area events | 0.655 | 2.320 | 1.035 |
| Restricted-area time (s) | 112.715 | 404.015 | 173.345 |
| Sector-exit events | 0 | 0.010 | 0.410 |
| Outside-sector time (s) | 0 | 0.865 | 257.870 |
| Clean completion | 20% | 1.5% | 11% |
| All-aircraft clean scenarios | 0% | 0% | 0% |

Both learned policies used 50,000 SB3 transitions, training seed 0, four worlds,
64x64 SAC networks, reach reward 25, discount 0.997, and four gradient updates per
vector step after a 5,000-transition warmup. The action mapping was the intended
experimental difference. Direct turns produced 46,796 live replay transitions;
goal-relative turns produced 38,012. Equal vector budgets do not mean equal live
experience or identical runtime. Training took 324.4 and 384.2 seconds respectively;
other local workloads were running, so this is not a throughput comparison.

Goal-relative control raises completion substantially relative to the undertrained
direct-turn policy, but neither policy beats the neutral controller overall.
In particular, the goal-relative policy spends more time in restricted areas and
outside the sector, and takes much longer to arrive. It is not a submission candidate.

Paired scenario-bootstrap comparisons (10,000 resamples) give a completion difference
of +61.5 percentage points [55.5, 67.0] against direct turns. Against neutral, its
restricted-area time increases by 60.63 seconds [8.70, 120.45]. Its intrusion-time
difference against neutral is +18.35 seconds [-23.62, 62.33], so this pilot does not
establish an intrusion-time improvement. These are pointwise descriptive intervals
across just 20 scenarios, not training-seed uncertainty or evidence of rare-event safety.

## Required single-agent track

The 20-scenario neutral baseline completes all goals with mean flight time 805.3 s,
76.65 s intrusion time, 96.65 s restricted-area time, and 5% clean completion.
The completed goal-relative SAC pilot is reported below. MA results do not substitute for SA.

## Engineering checks completed

- Correct per-aircraft SAC termination and exclusion of inactive padding from replay.
- Twenty-nine automated tests for metrics, lifecycle, seeding, control mapping, route geometry, actor initialization and paired comparisons.
- Model and replay continuation verified from 400 to 600 transitions with a fresh scenario seed.
- Wall-clock stop verified: a 10,000-transition request stopped at 100 additional
  transitions and saved the model; the final uncollected step is not counted as live replay.
- Per-run source snapshots, dependency lists, configurations, and evaluation model hashes.

Raw CSVs, JSON summaries, confidence intervals, checkpoints and source snapshots are
under runs/. They are excluded from Git. Main result paths:

- runs/neutral-ma-20.csv
- runs/sac-ma-balanced-pilot-50k/validation-20.csv
- runs/sac-ma-goal-relative-pilot-50k/validation-20.csv
- runs/compare-neutral-balanced-50k.json
- runs/compare-neutral-goal-relative-50k.json
- runs/compare-balanced-goal-relative-50k.json
- runs/neutral-sa-20.csv

## Next decision

Train beyond the 50,000-transition diagnostic budget and evaluate saved checkpoints
on the same development scenarios. Keep a direct-turn comparison in the larger study;
the initial result does not prove that it cannot converge with more experience.
If longer runs still fail, isolate sector containment, speed selection, and conflict
anticipation in separate experiments. Require high completion and reduced safety
violations before selecting a finalist. University Slurm jobs still need authenticated
access and verified partition/account/resource limits.

## Failure trace: multi-agent development scenario 7

Replaying the full prefix and tracing scenario index 6 reproduced every recorded
metric exactly (including 371 s mean intrusion time). Mean sampled CAS was 154.7 m/s;
slow cruise did not explain the long flight time in this case. Seven aircraft spent
more than 60% of sampled time over 60 degrees away from their goal bearing. The
trajectories show large detours, sector excursions, and several nearby parallel paths.
This is a case-specific diagnosis, not a claim about all scenarios.

The trace is saved in runs/diagnose-goal-relative-50k-scenario7, including a route/speed
figure, 10-second trajectory samples, the scenario geometry, and exact final metrics.
A route-guided experiment now uses buffered polygon visibility paths as a navigation
reference; the learned policy can select heading offsets and speed adjustments, with
three additional route-reference observations. Zero-action route following is being
measured as a separate classical baseline. It is not itself an RL submission, and the
learned controller must demonstrate added value on aircraft separation.

## Follow-up learned and geometric results

All figures below use the same 20 development scenarios per track. Times are seconds
per aircraft. None is a final evaluation or a demonstrated competition candidate.

| MA controller | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100% | 838.230 | 61.450 | 112.715 | 0 | 20% |
| Goal-relative SAC, 114,980 transitions | 100% | 1018.965 | 128.960 | 63.300 | 1.280 | 16% |
| Geometric route reference | 99.5% | 1045.030 | 135.320 | 0.255 | 5.335 | 24% |
| Geometric reference with sector clearance | 100% | 1068.880 | 148.500 | 0 | 0 | 20.5% |

The 114,980-transition run used seed 1100 and stopped at its 900-second learning
budget, saving the model and replay. Its directory name contains the requested 250k
budget, not the actual completed count. It collected 80,972 live aircraft transitions.
Compared with neutral, restricted time improved by 49.415 s [23.370, 77.256], but
intrusion time worsened by 67.510 s [37.410, 100.011]. Different training seeds and
budgets prevent attributing differences from the earlier 50k pilot solely to training
length. The continuation on fresh seed 1200 is reported below.

The geometric reference almost eliminates restricted-area occupancy, but routes can
funnel multiple aircraft through the same corridors. It is a classical baseline,
not a learned controller. One failed aircraft caused all 51 restricted-area seconds
and most sector occupancy; a route variant now includes inward sector clearance.
The route-guided SAC candidate starts with zero mean deviations and modest exploration,
then learns heading offsets and speed commands from live simulation experience.

| SA controller | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100% | 805.300 | 76.650 | 96.650 | 0 | 5% |
| Goal-relative SAC, 50k | 100% | 1079.000 | 125.400 | 99.200 | 16.200 | 10% |
| Geometric route reference with sector clearance | 100% | 1145.550 | 111.750 | 0 | 0 | 25% |

The SA learned pilot remains inferior on key safety/efficiency measures. The geometric
reference's zero static violations in 20 scenarios is promising but does not establish
rare-event safety. Conflict avoidance remains the main unresolved performance problem.

The completed MA sector-clearance reference also has zero static/sector violations
over its 200 development flights, with all goals reached. Its higher intrusion time
(148.5 s) reinforces the need for learned traffic separation. Both route references
are classical controllers; the learned route-guided follow-up is reported below.

## Replay-budget and learned route follow-up

The following completed evaluations use the same 20 development scenarios and original
objective metrics. Route results in this table used route revision 1, before the
waypoint-progress fix. Times are per aircraft in seconds.

| MA controller | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100% | 838.230 | 61.450 | 112.715 | 0 | 20% |
| Goal-relative SAC, 189,800 total transitions | 99.5% | 1064.640 | 70.450 | 95.805 | 24.460 | 22.5% |
| Route SAC, 77,740 transitions | 97.5% | 1169.940 | 111.240 | 14.920 | 9.015 | 29% |
| Same route weights with static filter | 97.5% | 1148.225 | 109.890 | 3.335 | 0 | 33.5% |

The goal-relative continuation added 74,820 transitions over 905.2 seconds, resuming
both weights and replay from the 114,980-transition run. With two worlds, eight updates
per collection step correspond to 0.4 per counted transition. Intrusion time improved
relative to its earlier checkpoint (128.96 to 70.45 s), while restricted and sector
violations worsened. It still does not beat neutral overall. Model SHA-256:
d6057ea75fdc9e613ca782c76c96c11b4f5d813a935552c3b7a6206082854840.

The route run used seed 1300, two worlds, eight updates per collection step and the
navigation initialization. It stopped at its 900-second learning cap with 49,903 live
and 27,817 padded transitions. The static filter was applied only at evaluation, to
isolate its effect on identical learned weights. Model SHA-256:
d11d52743d0da873babeca038b88b6d869ce4596b11735fa2047fb4c26b4be7f.

Paired scenario intervals for adding the filter show restricted-time change -11.585 s
[-19.555, -4.725], outside-time change -9.015 s [-17.970, -1.970], and clean completion
+4.5 percentage points [1.5, 8.0]. Intrusion change -1.350 s [-13.760, 10.290] does not
establish a conflict reduction. Completion remains 97.5%. These are pointwise intervals
from 20 development scenarios, excluding training-seed uncertainty. Filtered and
unfiltered route policies are not submission candidates.

Result prefixes are runs/sac-ma-goal-relative-replay4-seed1200/validation-20,
runs/sac-ma-route-guided-pilot-seed1300/validation-20 and
runs/sac-ma-route-guided-pilot-seed1300/validation-filtered-20. The paired comparison is
runs/compare-route-77k-filter.json. Full CSVs retain all event counts as well as times.

## Waypoint-progress diagnosis

The first route-policy scenario was replayed with a trajectory trace. One failed
aircraft repeatedly switched target indices 4 to 5 to 4 and circled until timeout.
Route revision 2 now retains progress, and a regression test checks that evasive
deviations do not reacquire passed vertices. Observation reads do not commit progress;
action selection does, and reset clears the state. Replay generated under revision 1
is rejected for continued route training because the action mapping changed.

The same final weights with revision 2 plus static filtering still complete only 8 of
10 aircraft in that failing scenario, with 494.6 s mean intrusion time and no static
violations. A correct tracking invariant alone has not repaired this policy's behaviour.
This one-scenario screen is diagnostic, not a new full-policy comparison. Its result is
runs/sac-ma-route-guided-pilot-seed1300/screen-progress-v2-filtered-1.json.

## Longer direct-control reference

The public_weights SAC reference used seed 1400, two worlds, eight updates per
collection step, 10k warmup, a 400k replay buffer, a 600k requested transition budget
and a one-hour learning cap. It retains the original heading/speed controls. This uses
the published reward coefficients with our BlueSky collector; it is not a reproduction
of the external fork. It completed 391,520 counted transitions; the final development
result is reported below.


The matching revision-2 classical reference with static filtering has now completed
all 20 scenarios (200 aircraft): 100% completion, 1068.750 s mean flight time,
148.500 s intrusion time, zero restricted/sector events or occupancy, and 20.5% clean
completion. It has no learned policy. Compared with the earlier unfiltered revision-1
classical run, the aggregate safety metrics are unchanged and flight time differs by
0.13 s. See runs/route-guided-inset-v2-filtered-ma-20.{csv,json}. This establishes that
the revision/filter combination preserves this small navigation baseline; it does not
establish general safety or an RL improvement.

The source-transfer integrity test passes with a project-local temporary directory;
both Slurm scripts pass Bash syntax checks. A cluster source bundle and smoke job are
prepared, but authentication, CUDA execution and scheduler allocation remain unverified.
See jobs/CLUSTER.md. The local training experiments continue independently.


## Predictive observations and original-harness integration

The new public_cpa recipe appends closest-approach and separation-entry predictions
from current horizontal state, with a 180-second horizon and explicit missing-traffic
masks. Rewards and controls match public_weights. Analytical geometry tests pass.
Neutral MA evaluation over all 20 development scenarios matches the original neutral
run exactly for every aircraft on all eight objective metrics. The evidence is
runs/neutral-ma-cpa-parity.json. Reward totals differ because the reference recipe
uses different reward coefficients; reward is not an objective competition score.

Actual SAC smoke runs completed 400 transitions in SA (28.29 s) and two-worker MA
(8.11 s), saving models; MA also saved replay with 400 live and no padded transitions.
These runs establish compatibility, not policy quality or comparative throughput.
They are runs/smoke-cpa-sa-400 and runs/smoke-cpa-ma-400. Local workloads overlapped.

The original evaluator now imports atc.submission only inside its two permitted
hooks. A syntax-tree comparison proves that the remaining harness source matches
upstream. The SA learned checkpoint matches all nine recorded metrics exactly on the
first two development scenarios. The initial MA check fails against historical batched
results. Repeating the batched evaluator reproduces those old results exactly, while
the original per-aircraft calls change aggregate intrusion time by -5.6 s and
restricted time by -8.2 s across the first 20 aircraft. The discrepancy is not merely
row ordering. Evidence is in runs/harness-check-ma-goal-relative-detail.json and
runs/sac-ma-goal-relative-replay4-seed1200/validation-repeat-2.

Development evaluation and diagnostics now use per-aircraft inference by default;
--batch-inference remains available for historical reproduction. The completed per-aircraft re-evaluation and successful repeat integration check
are reported below. The original
source still specifies seed 42 and 1000 episodes; all integration checks here use
2026 in a separate development process. The official sequence remains unused.


## Verified per-aircraft development results

The original MA harness now matches all nine metrics exactly for the first two
scenarios of the corrected evaluation (20 aircraft). The SA check likewise matches
two scenarios exactly. These checks validate the integration on that prefix, not the
full official sequence. See runs/harness-check-ma-individual.json and
runs/harness-check-sa-goal-relative.json.

A direct numerical probe on the same ten initial observations found differences in
13 of 20 action components, with a maximum absolute difference of
4.76837158203125e-7. Those small differences can grow during interacting rollouts.
The probe is runs/inference-rounding-probe.json. Per-aircraft calls are now the
consistent development/final inference convention on this host. Hardware and library
versions remain part of the reproducibility record.

| MA controller, per-aircraft inference | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral (unchanged) | 100% | 838.230 | 61.450 | 112.715 | 0 | 20% |
| Goal-relative SAC, 189,800 total transitions | 99.5% | 1071.685 | 69.670 | 94.180 | 28.935 | 22.5% |
| Public-reward SAC, 200,000-transition checkpoint | 36.5% | 2457.450 | 9.690 | 140.155 | 380.470 | 11.5% |

All rows use the common 20-scenario stream (200 aircraft). The public-reward checkpoint
has 0.180 intrusion events, 1.080 restricted-area events and 1.475 sector exits per
aircraft. Its low intrusion time is accompanied by widespread failure to arrive and
large sector excursions. It is not an acceptable improvement or submission candidate.
The goal-relative controller also remains inferior to neutral on several metrics.

Result prefixes are
runs/sac-ma-goal-relative-replay4-seed1200/validation-individual-20 and
runs/sac-ma-public-600k-seed1400/validation-200k-individual-20.
The public-reward checkpoint SHA-256 is
de9faf1cd0c7568cc21a9b484b866738b6d85b41d9d570a7268c1d505087314f.
This intermediate checkpoint is superseded by the 391,520-transition result below.
The failure at 200k shows why low intrusion time alone cannot select a controller.

The matching public_cpa experiment started after the reference completed. It requests
the same seed 1400, two worlds, 600k transitions, 400k replay capacity, 10k warmup,
eight updates per collection step and one-hour cap. Its source and actual completed
count are recorded per run. Equal wall limits need not produce equal transition
counts; compare matched checkpoints before claiming an observation-only benefit.


## Stronger public-reward checkpoint

The first substantial learned improvement is the final checkpoint from
runs/sac-ma-public-600k-seed1400: 391,520 counted transitions, 310,789 live transitions
and 80,711 inactive padded transitions in 3602.62 seconds. Padded rows are excluded
from SAC replay. This is one training run, not independent-seed replication.

| MA controller, per-aircraft inference | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100% | 838.230 | 61.450 | 112.715 | 0 | 20% |
| Public-reward SAC, 391,520 transitions | 99.5% | 1046.940 | 24.550 | 79.505 | 50.840 | 27.5% |

All values are from the same 20 development scenarios (200 aircraft); times are mean
seconds per aircraft. Intrusion time is the original scorer's accumulated pairwise
occupancy, not a unique wall-clock conflict fraction. Event counts for the learned
policy are 0.390 intrusion, 0.575 restricted-area and 0.160 sector-exit events per
aircraft. No scenario has every aircraft complete without a safety violation.

Against neutral, paired scenario bootstrap intervals give intrusion-time change
-36.900 s [-51.621, -22.510] and restricted-time change -33.210 s [-49.301, -16.765].
These correspond to about 60% and 29% reductions. Flight time increases by 208.710 s
[165.723, 254.396], outside-sector time increases by 50.840 s [28.125, 76.290], and
completion changes by -0.5 percentage points [-1.5, 0]. Clean completion changes by
+7.5 percentage points [-0.5, 15.0]. These pointwise intervals exclude training-seed
uncertainty and repeated model-selection effects. This is a promising tradeoff,
not an overall safety improvement or a competition-ready result.

Evaluation: runs/sac-ma-public-600k-seed1400/validation-final-individual-20.{csv,json}.
Comparison: runs/compare-neutral-public-391k.json. Model SHA-256:
b4c7a671dbd0fae8146522fa8278f19f4119c859caf2e1b07fe0bb56eae45093.

The weights and replay are being continued with fresh scenario seed 1800, one MA
world, four updates per collection step and a further one-hour cap in
runs/sac-ma-public-to1m-seed1800. The request is 608,480 additional transitions, toward
one million total, but the cap may stop earlier. Actual counts and evaluations are
required before claiming further improvement. This is continuation, not a second
independent training seed.

## Preserving a learned policy while adding predictions

PredictionResidual retains the original flattened observation coordinates and adds
an initially zero linear projection of the 45 new conflict features. The original
actor and critic networks retain their input sizes and weights. A frozen-zero
projection supplies a matched control. Both branches reset optimizer state and use
fresh replay; old 124-dimensional replay is incompatible with the new 169-dimensional
observations. Historical transition counts are retained but are not new CPA training.

The public-reward checkpoint was extended into runs/public-cpa-initial-predictions
and runs/public-cpa-initial-control. Full evaluation of the trainable initialization
matches every aircraft record on all nine metrics across the same 20 scenarios,
including reward. Evidence: runs/public-cpa-initial-predictions/initial-parity.json.
This validates initialization preservation, not an improvement from predictions.

Analytical/model tests check initial actor and critic equality, projection gradients,
source-model isolation, save/load preservation and the frozen control. The complete
regression suite passes 31 checks. A matched, equal-budget fine-tuning experiment has
not yet been run. Earlier goal-relative initialized variants were also prepared but
are not the preferred starting point after the stronger public-reward result.


A small real-world resumed-training check completed 400 additional live transitions,
with no padding, in 13.62 seconds, using a 1000-transition smoke buffer. It saved
weights and replay in runs/smoke-public-cpa-resume-400. This checks the extension's
training path without allocating another full training buffer; it is not a quality
experiment or a controlled throughput benchmark.

## Required single-aircraft transfer screen

The 391,520-transition MA checkpoint was copied unchanged into
runs/public-sa-transfer-391k. The SA world retains ten scripted intruders and observes
the nearest nine to match the source input size. Reset assertions check the actual
intruder IDs and ten scenario routes. No single-aircraft training was performed.

| SA controller | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100% | 805.300 | 76.650 | 96.650 | 0 | 5% |
| MA-trained public-reward SAC, transferred | 100% | 1096.650 | 66.850 | 84.300 | 19.950 | 15% |

These are 20 development scenarios, one controlled aircraft each. The transferred
policy averages 1.30 intrusion events, 0.65 restricted-area events and 0.10 sector
exits. Paired intrusion-time change is -9.800 s [-56.201, 31.800] and restricted-time
change is -12.350 s [-61.351, 26.950]. Neither establishes a safety improvement.
Flight time rises by 291.350 s [115.600, 512.456], with new sector excursions.
The transfer preserves arrival on this small sample but needs SA-specific training.

Results: runs/public-sa-transfer-391k/validation-20.{csv,json}; comparison:
runs/compare-neutral-sa-transfer-391k.json. The model hash is unchanged from the MA
source. The original harness matches every metric exactly on the first two SA
scenarios, including reward: runs/harness-check-sa-transfer-391k.json. The remaining
18 scenarios were evaluated with the development evaluator. Official seed 42 and
finalist seed 2027 remain unused.


## Matched 200k conflict-feature result

| MA SAC checkpoint | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Public reward, 200,000 transitions | 36.5% | 2457.450 | 9.690 | 140.155 | 380.470 | 11.5% |
| Same reward plus CPA inputs, 200,000 transitions | 27% | 2641.240 | 48.750 | 241.970 | 203.005 | 5.5% |

These are matched 20-scenario evaluations with per-aircraft inference, the same
training seed and gradient/update budget. Adding inputs changes the input-layer
parameter count and initialization; this is not the policy-preserving warm-start
comparison. The CPA checkpoint is unsuitable. Its lower sector occupancy accompanies
worse arrival, intrusion and restricted-area means. Neither 200k policy is acceptable.
This does not prove that predictive features cannot help after longer training.

Paired scenario intervals for CPA minus the reference: intrusion time +39.060 s
[25.450, 53.050], restricted time +101.815 s [43.305, 160.366], outside time -177.465 s
[-271.963, -73.607], arrival -9.5 percentage points [-22.0, 4.0]. Evidence:
runs/compare-public-cpa-200k.json and
runs/sac-ma-cpa-600k-seed1400/validation-200k-individual-20.{csv,json}. CPA model hash:
4718826d2a7d02a1a77d83425ed8ab622fa22a219f3886cd7d16522b9186a209.

The CPA trainer finished at 274,080 counted transitions (245,389 live, 28,671 padded)
in 3604.47 seconds. Its final weights and replay are saved; the 200k evaluation is
not its final-model evaluation. Its smaller completed count than the 391,520 reference
must not be treated as an equal-budget final comparison. Local workloads overlapped.

The queued SA-specific fine-tuning has now started in
runs/sac-sa-transfer-finetune-seed2100, using the transferred 391,520-transition MA
weights, two scripted-traffic SA worlds (seeds 2100/2101), fresh replay, 10k warmup,
two updates per vector step and a further one-hour cap. The request is 300k additional
SA transitions. Historical MA counts remain in the checkpoint; inspect new_timesteps
for the amount of SA training. No post-fine-tuning performance result is available yet.

## Direct-command static projection screen

The best completed MA checkpoint was screened with an evaluation-time action filter.
This filter leaves a predicted-clear learned turn unchanged, otherwise chooses a
nearby bounded turn whose approximate path remains within a 2 km polygon/sector
margin. A 90-second constant-speed, bounded-turn prediction is used. Speed, reward,
observations, world construction and original scoring stay unchanged. The filter
has no moving-traffic predictor or formal safety guarantee.

| First five MA scenarios, 50 aircraft | Completion | Flight | Intrusion | Restricted | Outside | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Public-reward SAC, unchanged | 100% | 1040.280 | 17.080 | 97.480 | 37.140 | 24% |
| Same SAC plus projection revision 1 | 98% | 1129.240 | 27.960 | 13.240 | 0 | 62% |
| Goal tracking plus the same projection | 98% | 1199.300 | 60.160 | 13.920 | 0 | 48% |

The filter improves static occupancy and clean completion on this small screen but
loses an arrival and increases intrusion versus the unfiltered SAC. It is not a
submission candidate. The classical comparison is useful for isolating learning's
traffic-control contribution, but five scenarios cannot establish that contribution
reliably. The learned filter intervenes on 1286 of 5669 commands, with 168 cases where
no candidate satisfies the predicted margin. Those cases must remain visible.

Results: runs/sac-ma-public-600k-seed1400/screen-projected-5 and
runs/goal-projected-ma-screen-5. The unfiltered row is the first-five-scenario subset
of the existing 20-scenario result, not a separately selected scenario collection.

Projection revision 2 stops predicted paths at the first goal-circle entry because
the actual environment removes an aircraft on arrival. Revision 1 assessed travel
beyond arrival. This correctness fix leaves the five-scenario learned aggregate
metrics unchanged, so it does not explain or repair the failed arrival. The corrected
result is runs/sac-ma-public-600k-seed1400/screen-projected-v2-5. A trajectory diagnosis
of the failed scenario and a matching original-harness check are in progress.

The geometry/control tests and real SA/MA action smoke pass. The full suite passed
36 checks before the goal-entry correction; the eight relevant geometry/submission
checks pass after the correction. Source snapshots retain the exact revision used by
each result. The source-v3 cluster bundle predates the new projection implementation.


The final CPA checkpoint (274,080 transitions) completes 66% of the first five
scenarios, with 1946.54 s flight time, 51.12 s intrusion, 75.20 s restricted occupancy,
81.32 s outside-sector occupancy and 14% clean completion. It improves on its earlier
arrival failures but is still unsuitable. This is a five-scenario screen, not a full
20-scenario result or a count-matched comparison to the stronger 391,520 reference.
See runs/sac-ma-cpa-600k-seed1400/screen-final-individual-5; model SHA-256:
b444e4bd882117c0b70067d918a2cfb7645209f5e434349fb308d5843f0e1a93.

The projection revision-2 harness check now matches all nine metrics exactly on the
first two scenarios (runs/harness-check-public-projected-v2-391k.json). Both learned
and classical five-scenario revision-2 aggregate results match revision 1; every
learned aircraft metric also matches. Paired screen comparisons are saved in
runs/compare-projection-v2-screen-5.json. These small, repeatedly inspected scenarios
remain development evidence only.

The failed revision-2 aircraft KL002 in scenario index 2 takes a long boundary detour:
its goal distance grows from 124.47 km to 236.37 km, then falls to 15.96 km before the
3000 s limit. It has 116 interventions over 300 decisions, with a feasible candidate
available at every decision. It incurs no static violations in that scenario. This
is not an infeasible-geometry fallback failure. The trace and plot are under
runs/diagnose-public-projected-v2-episode2; failure_analysis.json records the summary.

Revision 3 prioritized heading toward the goal among feasible interventions. It
repairs that failed flight, but creates two failures in another scenario. Learned
five-scenario completion falls to 96%, flight time is 1150.08 s, intrusion 32.48 s,
restricted 15.36 s, outside 0 and clean completion 56%. The corresponding classical
screen also has 96% completion, 119.64 s intrusion and 24.72 s restricted occupancy.
This variant is rejected. Results are screen-projected-v3-5 under the reference run
and runs/goal-projected-v3-ma-screen-5.

Revision 4 restores nearest-command selection and retains the corrected goal-entry
handling, including strict capture for tangent/outward boundary cases. All 38
regression checks pass. The next exploratory experiment trains with this filter in
the environment rather than applying it only after learning. It is queued behind the
unfiltered MA continuation: runs/sac-ma-projected-finetune-seed2200, one world, fresh
replay, 10k warmup, four updates per collection step, 200k requested additional
transitions and a one-hour cap. Initialization is the verified 391,520 checkpoint.
This is an exploratory fine-tune, not a matched training ablation: the unfiltered
continuation reused replay and used a different scenario seed.


## Completed reference continuation at 594,480 transitions

The one-hour continuation adds 202,960 counted transitions, comprising 126,409 live
transitions and 76,541 inactive padded rows. The cumulative recorded counts are
437,198 live and 157,252 padded; elapsed learning/save time is 3605.75 seconds. This
falls short of the requested one-million total, and is not a second independent
training seed. Both the 591,520 checkpoint and final model/replay are saved.

The final 20-scenario result is 98% arrival, 1083.37 s flight, 21.36 s intrusion,
63.66 s restricted occupancy and 56.96 s outside-sector occupancy, with 30.5% clean
completion. Event means are 0.400 intrusion, 0.485 restricted-area and 0.195 sector
exits per aircraft. No scenario has all aircraft complete cleanly.

Against the 391,520 checkpoint, arrival changes by -1.5 percentage points [-4.0, 0],
intrusion time by -3.190 s [-13.970, 7.620], restricted time by -15.845 s
[-34.341, 1.966], outside time by +6.120 s [-13.105, 26.170] and flight time by
+36.430 s [-12.760, 93.971]. These pointwise development intervals do not establish
improvement over the earlier checkpoint. Its lower arrival mean misses our provisional
gate, so the earlier model remains the reference.

Against neutral, intrusion-time change is -40.090 s [-56.590, -25.480], restricted
occupancy -49.055 s [-74.681, -25.684] and clean completion +10.5 percentage points
[3.5, 17.5], but flight time, sector occupancy and arrival are worse. The complete
performance vector must remain visible. Comparison outputs are
runs/compare-public-391k-594k.json and runs/compare-neutral-public-594k.json.

Final evaluation: runs/sac-ma-public-to1m-seed1800/validation-final-individual-20.
Model SHA-256: d0019028f5f6c25b968f7604a8102a479e92249cd2d46d7cb2a6507abadf84b0.

The reference process completed normally and released the queued projected MA
fine-tune. Its actual config confirms revision 4, fresh replay, one world and seed
2200. The SA fine-tune likewise confirms seeds 2100/2101, nine observed traffic slots,
fresh replay and no static filter. Both start at the earlier 391,520 weight checkpoint
and have begun optimizer updates after warmup. Their requested counts are not results.

The verified source-v4.zip bundle contains 104 source files and the current training
implementation; it predates this latest result summary. Its SHA-256 is
79425e36f5086b8ad185e20621a82e800d50a0d78d8324bc60b900a039c1dbee.


## Public learner configuration check

The recovered E26 trial-13 architecture, learning rate and tau can now be specified
in the trainer and Slurm template. A local MA integration run with 256x256 layers
completed 400 transitions and 80 optimizer updates. Warmup (200 transitions) and
replay capacity (1000) were deliberately reduced for this check. A 200-transition
continuation performed another 40 updates and preserved the saved learner settings
exactly. Requested and actual model values are recorded separately. Local shell
rounding changed the tau argument by about 2e-18; the actual value is retained in
both configurations. Cluster argument forwarding preserves the full decimal string.

All 38 existing tests and Bash syntax checks passed. A dry argument-capture check
confirmed that 16 MA workers receive 64 updates, the 256x256 network, published
learning rate/tau, batch size 256 and warmup 50,000. No cluster job was submitted.
These are integration checks, not evidence that the larger learner scores better.
Evidence: runs/smoke-public-learner-400/learner-verification.json,
runs/smoke-public-learner-resume-200/learner-verification.json and
runs/learner-slurm-check/verified-arguments.json.


## Filter-trained checkpoint at 491,520 transitions

The checkpoint adds 100,000 counted MA transitions to the 391,520-transition source,
using seed 2200, one world, four gradient updates per world batch, fresh replay and
10,000 transitions of warmup. This compares a changed action filter plus additional
training against the retained unfiltered reference, not an isolated training ablation.
The exact revision-4 filter is used during both training and evaluation.

On the matched 20 development scenarios, the mean changes relative to the unfiltered
reference are +18.84 s intrusion [7.17, 29.96], -71.465 s restricted-area occupancy
[-93.66, -51.01], -50.84 s outside-sector occupancy [-76.29, -28.12], +217.44 s flight
[142.09, 300.39], and -2 percentage points arrival [-5, 0.5]. Clean completion increases
by 19.5 points [8, 30.5]. These are pointwise paired scenario-bootstrap intervals.
Two of twenty scenarios have every aircraft finish without an objective safety event.

There are five failed arrivals out of 200. The filter intervenes on 6258/25381 commands
(24.7%); 478 predictions have no feasible candidate under its approximate geometry.
Zero observed outside-sector time on this small sample is not a guarantee. The
increase in aircraft conflicts prevents treating the static-safety improvement as an
overall win, and 97.5% arrival fails the internal completion gate.

Model SHA-256: c5558ca89ac8102d7e88c25af4a0fbb2a707c0d41d9b134204d99b0652f91bce.
Evidence: runs/sac-ma-projected-finetune-seed2200/validation-491k-individual-20,
runs/compare-public-391k-projected-491k.json and runs/compare-neutral-projected-491k.json.

The separate reference-failure audit at runs/reference-failure-audit-20/report.json
checks distribution tails and the 196 aircraft that arrive under neutral, 391k and
594k policies. Among those common completed flights, mean intrusion is 59.27, 24.33
and 20.26 s respectively. The 391k intrusion benefit is therefore also present among
completed flights. Its overall 90th percentile is 81 s versus neutral's 195.2 s;
26 aircraft leave the sector, versus zero under neutral. Episode 14 / KL002 misses
arrival under both learned checkpoints without an intrusion, motivating a trajectory
trace rather than assuming the failure is caused by collision avoidance.


## Same-filter control: original checkpoint versus fine-tune

The original 391,520-transition checkpoint, evaluated with revision-4 projection
without additional training, reaches 97.5% arrival, 1188.91 s flight, 27.69 s intrusion,
5.755 s restricted-area occupancy and zero outside-sector time. Clean completion is
59.5%, and two of twenty scenarios are entirely clean. The filter intervenes on
5372/23873 commands (22.5%), with 344 no-feasible-candidate predictions.

Compared with this matched-filter control, the 491,520-transition fine-tune increases
intrusion time by 15.70 s [2.04, 28.47] and reduces clean completion by 12.5 percentage
points [-23, -2]. Arrival stays at 97.5% but the failed aircraft differ. The observed
static-safety gain in the earlier comparison comes from applying the filter; this
100k fine-tune has not improved it. Keep the original filtered checkpoint as a
secondary diagnostic reference, not a submission candidate or an overall winner.
Evidence: runs/sac-ma-public-600k-seed1400/validation-projected-v4-individual-20 and
runs/compare-projected-public-391k-491k.json. The original model hash is unchanged.

## Completed single-agent fine-tune

The SA fine-tune stops at 471,338 total counted transitions after adding 79,818 in
3604.8 seconds. Its replay contains 79,816 new live transitions; the final callback
stops collection before storing the last two vector slots. It uses seed 2100, two
SA worlds, fresh replay, and the transferred nine-slot observation while retaining
all ten scripted world intruders. This is a short SA-specific continuation.

| SA controller | Arrival | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral | 100% | 805.30 | 76.65 | 96.65 | 0 | 5% |
| MA-to-SA transfer, before SA training | 100% | 1096.65 | 66.85 | 84.30 | 19.95 | 15% |
| SA fine-tune, 471,338 total transitions | 95% | 1155.90 | 55.15 | 67.00 | 82.60 | 20% |

Against the transfer checkpoint, intrusion changes by -11.70 s [-54.15, 37.50],
restricted-area occupancy by -17.30 s [-36.50, -0.55], and outside-sector time by
+62.65 s [-30.80, 201.40]. One of twenty flights misses its destination. Neither an
overall gain nor a reliable intrusion improvement is established. Do not replace the
transfer reference with this fine-tune. All intervals are pointwise paired scenario
bootstrap estimates, not independent-training-seed evidence.

Model SHA-256: 7f7fbd266600dc63cb6d105d11fa4027f4550ed54a6b057db68f8ccdd08eb5af.
Evidence: runs/sac-sa-transfer-finetune-seed2100/validation-final-20,
runs/compare-sa-transfer-finetune-471k.json and runs/compare-neutral-sa-finetune-471k.json.

## Arrival-failure trace and next learner run

The retained policy's episode 14 trace exactly reproduces all nine reference metrics.
KL002 starts 160.27 km from its goal, moves as far as 206.27 km away, makes repeated
loops and is still 41.08 km away at the final 2990-second sample. It never comes within
20 km. Other aircraft have all departed by the 1970-second sample. This rules out a
near-goal capture error for this failure; sustained lack of progress consumes the
3000-second budget. The aircraft has zero separation, restricted-area and sector
violations. Evidence and plot: runs/diagnose-public-391k-episode14.

A new run, runs/sac-ma-public-learner1m-seed2500, tests the recovered public SAC learner
from scratch: 256x256 layers, learning rate 0.000442773440394527, tau
0.008242901455713948, batch 256, warmup 50k and replay capacity 1M. The exact loaded
values were checked in config.json. It uses one local MA world and four updates per
world batch (0.4 per counted aircraft transition), requests 1M transitions, and is
capped at two hours. It uses our synchronous BlueSky collector, not the public Rust
or staggered implementation. No quality result is available yet. This single run is
not a controlled architecture ablation or a reproduction of the reported E27 score.


## Expanded neutral development baseline

The 200-scenario neutral baseline completes all 2000 aircraft flights, with mean
flight time 851.722 s, intrusion events 0.864, intrusion time 65.183 s, restricted-area
events 0.805, restricted-area time 124.336 s, zero sector events and zero outside time.
Clean completion is 16.1%; no whole scenario is entirely clean. The first twenty
scenarios match every aircraft and all nine metrics of the earlier neutral evaluation
exactly. The retained learned policy's matching 200-scenario run is also complete;
its paired results appear in the expanded reference table near the beginning.
Evidence: runs/neutral-ma-200 and runs/neutral-ma-200-prefix-parity.json.


## Classical goal tracking with the same filter

On the same twenty MA scenarios, classical goal tracking with revision-4 projection
reaches 98% arrival, 1069.18 s flight, 84.26 s intrusion, 11.435 s restricted-area time
and zero outside-sector time. Clean completion is 39%, with no entirely clean world.
This is a diagnostic baseline, not an RL competition entry.

The original learned policy with the same filter reduces intrusion by 56.57 s
[-100.46, -22.48] and raises clean completion by 20.5 percentage points [7.5, 32].
It takes 119.73 s longer [5.07, 227.87]; arrival changes from 98% to 97.5%, with a
paired difference interval [-4.5, 4] percentage points. The learned component adds
traffic avoidance beyond the geometric filter. This comparison does not establish
submission readiness or final-test generalization.

Evidence: runs/goal-projected-v4-ma-20 and runs/compare-goal-projected-public-391k.json.
The same-weight filter-on/off comparison is also recorded in
runs/compare-public-391k-projection-v4-20.json: clean completion rises 32 points
[25, 39], but completion drops 2 points [-4, -0.5] and flight time rises 141.97 s
[96.55, 191.39]. Intrusion changes by +3.14 s [-3.12, 10.45].

The filter-trained continuation has now finished at 584,360 total counted transitions,
adding 192,840 in 3604.5 seconds. Its new replay contains 112,292 live transitions and
skips 80,538 inactive slots; the last ten counted slots are not stored when the deadline
callback ends collection. Its completed final twenty-scenario evaluation is reported
below; it does not replace the retained reference.


## Final filter-trained result and predictive-feature follow-up

At 584,360 total transitions, the filter-trained model reaches 98.5% arrival, 1117.18 s
flight, 47.53 s intrusion, 5.965 s restricted-area time, zero outside time and 55.5%
clean completion on the twenty development scenarios. No entire scenario is clean.
It improves arrival and flight time over its 491,520-transition checkpoint, but does
not achieve an overall improvement over the original checkpoint with the same filter.

Against that original filtered checkpoint, the final model changes arrival by +1
percentage point [-1, 3], flight time by -71.73 s [-149.50, 27.15], intrusion by
+19.84 s [-5.11, 50.80] and clean completion by -4 points [-13.5, 6]. These pointwise
intervals do not establish a population-level regression, but the observed severe
failures prevent promoting it. Its worst aircraft records 1160 intrusion seconds.
In episode 15, KL004 records one intrusion event and 1059 intrusion seconds; two
other aircraft in episode 8 each record 799 seconds. The original filtered policy's
worst observed aircraft has 213 intrusion seconds on the same scenarios.

Model SHA-256: 65600ffbb72da4193fcbd668073f6b5fa6406bf8a4dfd6492d24f355c9c1bcd4.
Evidence: runs/sac-ma-projected-finetune-seed2200/validation-final-individual-20,
runs/compare-projected-public-391k-584k.json and runs/compare-projected-491k-584k.json.

The next focused comparison retains the original 391,520-transition policy and
revision-4 filter. The two variants have identical initial policy tensors and differ
in whether the initially zero projection from CPA features may learn. Both reset
optimizers, start with fresh replay, use the same training seed and counted budget,
and retain all aircraft and the original scored world. The CPA features estimate
closest approach from current relative position and velocity. This tests whether
explicit prediction improves conflict avoidance; it does not assume the from-scratch
CPA pilot established a benefit. Replay capacity is 100k for this 100k-transition
paired pilot, avoiding unnecessary local storage and memory use.


The guarded prediction/control initializations passed an exact parity check over
2210 aircraft decisions in the first two development scenarios. All policy state
tensors match between the two initializations; actor and critic optimizer states
are empty. Both produce the original filtered policy's exact actions and all nine
reference metrics on those scenarios. The run configurations were then checked for
matching seeds, discount, architecture, learning rate, tau, gradient count, batch,
warmup, buffer, observation shape and projection revision. Only prediction-branch
trainability differs. Evidence: runs/cpa-guarded-initial-parity/result.json and
runs/cpa-guarded-ablation-seed2600/manifest.json.

The paired jobs are runs/sac-ma-cpa-guarded-predictions-seed2600 and
runs/sac-ma-cpa-guarded-control-seed2600. Each requests 100,000 additional transitions
from the same 391,520-transition initialization, using seed 2600, one MA world,
four updates per world batch, 10k warmup, batch 256 and a 100k replay buffer.
Both completed the full 100k additional counted transitions, reaching 491,520 total
and 188,600 cumulative optimizer updates. Control collected 61,096 live transitions
in 2057.1 seconds; prediction collected 60,479 in 2097.8 seconds. The remaining slots
are inactive-aircraft padding. The completed control's actor, critic and target
prediction projections remain exactly zero; the enabled variant's projections have
learned nonzero weights. Configuration, budgets and updates match. Evidence:
runs/cpa-guarded-ablation-seed2600/completion_audit.json. Both twenty-scenario
quality evaluations are complete and reported below. The comparison tests
the predictive representation and its trainable projection together; it does not
isolate their benefit from the additional trainable capacity. The features are
computed from current observed state, so no new future information is supplied.


## Prediction observation audit

A separate live-simulator check covers the first development scenario in each track,
using neutral actions and the public_cpa observations. It checks 857 MA aircraft
decisions and 78 SA decisions, including 701 absent traffic slots as MA aircraft
arrive. There are 7792 present traffic slots and 205 predicted-conflict flags across
the two tracks. MA traffic populations range from one to ten aircraft; SA retains
ten scripted intruders alongside its controlled aircraft.

After undoing normalization and body-frame rotation, positions agree with simulator
geometry to within 2.2e-10 metres and relative velocities agree with simulator ground
velocities to within 1.2e-13 metres per second. Feature values agree with a global-frame
calculation to within 7.2e-15. Both sampled scenarios have zero wind, which is checked
explicitly. This verifies the observation adapter, scale, relative-velocity sign,
rotation and presence mask in the sampled states. It does not measure forecast
accuracy under future turns, establish policy quality, or validate nonzero-wind cases.
No training feature layout or scored environment code was changed.

Evidence and reproducible check: runs/cpa-observation-state-audit/result.json and
runs/cpa-observation-state-audit/check.py.


## Reproduced prolonged-conflict geometry

The rejected 584,360-transition filtered model's development episode 15 was replayed.
All ten aircraft and all nine final metrics match the stored reference exactly.
The longest sampled pair is KL001/KL004: they remain inside 5 NM at every 10-second
sample from 490 through 1540 seconds, spanning 1050 seconds. The original one-second
scorer records 1059 intrusion seconds and one event for KL004; KL001 records 1160
seconds across three events. Both aircraft arrive.

During that interval, the pair's median heading difference is 10.32 degrees, with
68.9% of samples below 15 degrees. Median true-airspeed difference is 4.12 m/s.
Median separation is 2.669 NM, and the sampled minimum is 0.029 NM (about 54 metres).
The trace supports prolonged close flight in similar directions. Static filtering
intervenes in 53 and 66 of the 106 sampled decisions for KL001 and KL004 respectively.
Those interventions do not resolve the conflict; this replay does not isolate why
the formation developed or establish that the filter caused it.

This is a targeted failure diagnosis, not an additional aggregate policy evaluation.
The paired CPA experiment should be examined for persistent close-flight failures
as well as its mean intrusion score. Evidence: runs/diagnose-projected-584k-episode15,
including conflict_analysis.json, longest_pair.csv, longest_pair.png, trajectory.csv,
and the exact scorer metrics in summary.json.


## Matched predictive-feature result

The completed twenty-scenario comparison does not support promoting the prediction
variant. It reaches 99% arrival, 1110.26 s flight, 45.53 s intrusion, 9.535 s restricted
time, zero outside time and 48% clean completion. The matched frozen-feature control
reaches 97.5%, 1155.87 s, 32.76 s, 5.88 s, zero and 52.5% respectively. The predictive
variant has one entirely clean scenario; control has none.

Against control, prediction changes arrival by +1.5 percentage points [-0.5, 3.5],
flight time by -45.61 s [-99.19, 4.97], intrusion by +12.77 s [-6.46, 32.43], and
clean completion by -4.5 points [-18, 9.5]. Restricted-area events increase by 0.06
[0.005, 0.125]. These pointwise intervals do not establish an overall gain.

Against the original policy with the same filter, prediction raises intrusion by
17.84 s [1.38, 36.09], raises intrusion events by 0.25 [0.06, 0.45], and reduces clean
completion by 11.5 points [-21.5, -0.5]. Flight time improves by 78.65 s
[6.63, 159.05] and observed arrival rises 1.5 points, but that arrival interval
[-1.5, 4.5] includes no gain. The unfiltered reference also retains lower intrusion
and flight time on these scenarios. This pilot is rejected as a replacement.

Six predictive-policy aircraft spend more than 300 seconds in intrusion. Its worst
pair, KL001/KL002 in episode 18, each records 630 seconds in one event and then arrives.
Control's maximum is 240 seconds and the original filtered reference's maximum is
213 seconds; neither has an aircraft above 300 seconds on this twenty-scenario set.
The predictive policy therefore still exhibits prolonged conflicts. No independent
training-seed or reserved-stream generalization claim follows from this short pilot.

Model hashes:

- Control: d19609f77bcf808077de56e3b477651cfc3da45b7febf2b7f76fb549f295cc24
- Predictions: eb72857355e112fda1a5ac576766e415facb4c42155aa3fb79e3bbf3ed303f7f

Evidence: runs/cpa-guarded-ablation-seed2600/quality_summary.json,
runs/compare-cpa-guarded-control-predictions-491k.json,
runs/compare-projected-reference-cpa-predictions-491k.json,
runs/compare-projected-reference-cpa-control-491k.json, and
runs/compare-public-reference-cpa-predictions-491k.json.

The joint action-design experiment checks the learned policy's proposed actions
against traffic and static constraints together, using the retained weights and
evaluating before any additional policy training. Compare against the original filtered policy
and a classical controller using the same action correction, so the learned component's
contribution is measured. Preserve heading/speed-only control, the original world,
scoring, scenario distribution and evaluation protocol. The implementation, checks and completed pilots are reported below.


## Joint traffic/static action correction: implementation and initial checks

Revision 1 now screens heading and speed candidates against static geometry and
nearby traffic together. It uses a 90-second turn/acceleration forecast, 5-second
segments, a 2 km static margin and 1 km additional traffic margin. Closest approach
is checked within segments, and aircraft retire from forecasts after predicted goal
capture. Later aircraft in the fixed action order account for earlier selected
maneuvers. No scoring, world dynamics, scenario distribution or observation/reward
features were changed. This combined action-design experiment also adds speed search
and a new motion predictor; it does not isolate the traffic check alone.

The full suite passes 46 tests, including crossing between samples, narrow-corridor
speed recovery, existing-conflict recovery, goal retirement, parameter validation and
the original harness's two-hook restriction. Actual MA training completed 400
transitions and resumed for 200 more with replay and inherited joint-filter settings.
A two-scenario MA deployment check matches all nine original-harness metrics exactly.
The first SA smoke exposed a clock-field mismatch; the implementation now uses the
shared BlueSky clock, and the repeated SA smoke completes without that error.

With unchanged retained weights, the two MA smoke scenarios reach 100% arrival,
1168.35 s mean flight, zero intrusion, 8.8 s restricted-area time, zero outside-sector
time and 90% clean completion. The classical SA smoke reaches one of two goals,
with 13 s mean intrusion, 1 s restricted time and zero outside time. These tiny screens
verify execution and expose failures; they do not establish competitive performance.

The full twenty-scenario MA learned/classical comparisons and unchanged-weight SA
transfer are complete and reported below. Missed arrivals prevent an overall
replacement decision despite the MA intrusion reduction.
Evidence: runs/smoke-joint-projection-v1-ma-2,
runs/smoke-joint-projection-v1-sa-2-fixed,
runs/check-harness-joint-v1-ma-2.json,
runs/smoke-joint-projection-v1-train-400 and
runs/smoke-joint-projection-v1-resume-200.


## Completed joint-correction pilot

On the same twenty MA scenarios, unchanged retained weights with joint correction
reach 96.5% arrival, 1227.325 s flight, zero intrusion events/time, 5.035 s restricted
area time, zero sector exits/outside time and 89.5% clean completion. Seven of twenty
whole scenarios are entirely clean. The controller makes 7332 corrections in 24633
aircraft decisions, including 406 speed changes; it reports 429 predictions with no
jointly feasible candidate. Prediction infeasibility is distinct from the actual
one-second safety metrics. All 200 aircraft records and the two-scenario smoke prefix
were verified in runs/joint-v1-audit.json.

Against the original static-only variant, mean intrusion falls by 27.69 s
[-38.11, -18.06] and clean completion rises 30 percentage points [20.5, 40.0].
Arrival changes by -1 point [-3.5, 1.0], flight time by +38.415 s [-11.327, 94.440],
and restricted time by -0.72 s [-3.03, 1.70]. Against the unfiltered reference, arrival
falls 3 points [-5.5, -1.0] and flight time rises 180.385 s [122.325, 243.265]. All
193 aircraft arriving with the joint correction also arrive unfiltered; intrusion
time on this common-arrival subset falls from 25.161 s to zero. This diagnostic
shows the safety gain is not confined to failed arrivals. The primary estimates
remain the full scenario-level comparisons, with pointwise bootstrap intervals.
Zero observed intrusions on this development set does not prove general safety.

The classical goal controller with the same correction reaches 97.5% arrival,
1149.04 s flight, 3.47 s intrusion, 6.545 s restricted time, zero outside time and
87% clean completion. Learned-minus-classical intrusion is -3.47 s [-7.74, 0],
clean completion +2.5 points [-3.5, 8.5], and arrival -1 point [-5.0, 3.5]. This
sample does not establish an overall learned-policy advantage over that strong
classical diagnostic. The learned system is retained for further work as a safety
candidate, not selected as a submission. The pure classical controller is not an
RL competition entry.

Evidence: runs/sac-ma-public-600k-seed1400/validation-joint-v1-individual-20,
runs/goal-joint-v1-ma-20, runs/compare-projected-joint-v1-public-391k.json,
runs/compare-public-joint-v1-391k.json and runs/compare-goal-learned-joint-v1-ma.json.
The weights retain SHA-256 b4c7a671dbd0fae8146522fa8278f19f4119c859caf2e1b07fe0bb56eae45093;
runs/public-joint-v1-391k supplies the explicit deployment configuration.

### Required single-agent track

| Unchanged MA-to-SA weights | Arrival | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unfiltered | 100% | 1096.65 | 66.85 | 84.30 | 19.95 | 15% |
| Static correction v4 | 90% | 1375.75 | 61.10 | 12.40 | 0 | 25% |
| Joint correction v1 | 85% | 1432.15 | 14.25 | 12.50 | 0 | 65% |

All rows use twenty development scenarios and simulate all ten scripted intruders.
Compared with static correction, joint correction reduces intrusion by 46.85 s
[-73.85, -22.75] and increases clean completion by 40 points [20, 60], but introduces
one additional failed arrival. Two failures already occur with static correction.
The joint SA variant is not selected. The original harness exactly reproduces all
nine metrics for the first two SA and MA joint-correction scenarios. Actual training
smoke checks increase optimizer updates from 120 to 200 on resume, preserve both
filters and reject replay when the traffic filter is switched off.

Evidence: runs/public-joint-v1-sa-transfer-391k/validation-20,
runs/public-sa-transfer-391k/validation-projected-v4-20,
runs/compare-sa-static-joint-v1.json, runs/compare-sa-transfer-joint-v1.json,
runs/compare-neutral-sa-joint-v1.json, runs/check-harness-joint-v1-sa-2.json and
runs/smoke-joint-projection-v1-resume-200/verification.json.

### Navigation failure diagnosis

MA episode 3 was reproduced with exact agreement on every final metric. Failed
KL001 travels about 467 km in sampled positions, approaches no closer than 59 km
to its goal and finishes 143 km away. Failed KL0010 travels about 489 km and finishes
89 km away. During the final 1000 seconds, neither has a predicted traffic threat;
static threats trigger corrections on 90 and 70 of their 100 decisions respectively.
Their mean CAS in that interval is 147.94 and 160.75 m/s. The trace shows prolonged
static-obstacle detours far from the goals. It does not support treating these
failures as simple slow-flight or near-goal capture problems.

Evidence: runs/diagnose-joint-v1-episode3/failure_analysis.json, trajectory.csv and
trajectories.png. The route-input experiment below tests explicit static-route guidance in the
policy's navigation input while preserving the validated traffic correction.
It reuses retained weights for a bounded initial evaluation and measures the
learned component against a matching classical baseline. The route-input variant is now implemented, with six focused tests and 52 tests
passing overall. A 220-observation simulator audit covers both tracks and confirms
that every original field except the bearing is unchanged, with a 124-element
policy input and all ten actual SA intruders. Its completed quality evaluation is recorded below;
the existing route-relative controller is a different historical experiment.


## Route-bearing input v1: completed matched comparison

Both learned deployment variants use the exact original checkpoint bytes, SHA-256
b4c7a671dbd0fae8146522fa8278f19f4119c859caf2e1b07fe0bb56eae45093. No additional
training was used for this quality test. Only the goal-bearing channels change when
the final goal is blocked. Original final-goal distance and direct controls remain.
The planner re-routes from the current position when the remaining route is invisible.
Static and traffic correction retain their previous settings and the actual world,
scenarios and one-second scoring are unchanged.

| Track and controller | Arrival | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MA learned, joint correction only | 96.5% | 1227.325 | 0 | 5.035 | 0 | 89.5% |
| MA learned, route input + joint correction | 97.5% | 1292.410 | 0.430 | 0.505 | 0 | 95.5% |
| MA classical, route input + joint correction | 100% | 1125.155 | 0 | 0 | 0 | 100% |
| SA learned, joint correction only | 85% | 1432.150 | 14.250 | 12.500 | 0 | 65% |
| SA learned, route input + joint correction | 90% | 1449.750 | 16.650 | 0 | 0 | 65% |
| SA classical, route input + joint correction | 100% | 1320.100 | 8.850 | 0.100 | 0 | 75% |

Each row uses twenty development scenarios. MA has 200 aircraft records; SA has
20 controlled-aircraft records and retains all ten actual scripted intruders.
The MA learned route-input policy records 0.01 intrusion events and 0.01 restricted
entries per aircraft, with 13/20 entirely clean scenarios. Classical MA has zero
values for all six safety metrics and all 20 scenarios are entirely clean. SA learned
has 0.4 intrusion events per episode; classical SA has 0.25 and 0.05 restricted entries.

Compared with learned joint correction alone, route input increases MA clean
completion by 6 percentage points [1.5, 10.5] and lowers restricted time by 4.53 s
[-7.85, -1.63]. Arrival changes by +1 point [-1.5, 4], flight time increases by 65.09 s
[5.04, 126.70], and intrusion increases from zero to 0.43 s [0, 1.29]. It is therefore
a trade-off, not a uniform improvement. The learned policy loses 2.5 arrival points
[-5, -0.5], takes 167.25 s longer [86.18, 252.07], and loses 4.5 clean-completion
points [-7.5, -1.5] against the matching classical MA controller. The original weights
have not yet learned to use the changed navigation input effectively.

SA route input changes arrival by +5 points [-10, 20] against joint correction alone,
with unchanged 65% clean completion. The classical SA comparator reaches all goals
and has lower mean intrusion, but remains slower than the internal flight-time target.
Intervals are pointwise paired scenario bootstrap estimates; they do not cover
training-seed variation or repeated model selection on this development stream.

Route-input statistics are diagnostic observation counts, not unique decisions.
Learned MA records 258 successful and 10 failed replans; classical MA records 83
successful and no failed replans. Learned SA records 19 successful replans; classical
SA records 10, with no failed replans for either. Learned MA needs 5,067 action
corrections versus 3,153 for classical MA. These counts suggest that preserving the
classical navigation reference may be more useful than replacing its commands with
an existing policy trained on a different bearing definition.

Evidence: runs/public-route-input-v1-joint-391k/validation-20,
runs/public-route-input-v1-joint-sa-transfer-391k/validation-20,
runs/goal-route-input-v1-joint-ma-20, runs/goal-route-input-v1-joint-sa-20,
runs/compare-ma-joint-route-input-v1.json,
runs/compare-ma-classical-learned-route-input-v1.json,
runs/compare-sa-joint-route-input-v1.json and
runs/compare-sa-classical-learned-route-input-v1.json. Source/configuration provenance
is saved alongside each evaluation. The original competition harness exactly
reproduces all nine metrics for the first two route-input scenarios on both tracks:
runs/check-harness-route-input-v1-ma-2.json and
runs/check-harness-route-input-v1-sa-2.json. The 52-test suite passes. A 400-transition
training check performs 120 optimizer updates; resuming for 200 more transitions
reaches 600 cumulative transitions and 200 updates, retaining route metadata and
both filters. See runs/smoke-route-input-v1-resume-200/verification.json.

The next learning experiment should start from the classical navigation command and
learn bounded heading/speed adjustments, with the same joint correction. First verify
that zero adjustment reproduces the classical controller, then train with fresh
replay and compare against both the zero-adjustment controller and retained learned
references. Any improvement must survive independent seeds and unseen scenarios.
The classical controller alone does not satisfy the RL-based competition requirement.

## Larger-network SAC pilot: completed at 347,540 transitions

The one-world seed-2500 run reaches its two-hour limit after 347,540 counted
transitions: 254,456 live aircraft transitions, 93,074 inactive padding transitions,
and ten final counted transitions not stored because the deadline callback stopped
collection. The requested one million transitions were not completed. Elapsed time
is 7,207.98 s including final saving, with 119,012 optimizer updates. The final model
SHA-256 is c51a447ab59a50ac5d13488937825eeddc47b7b51d42889463d477243d61db81.

The learner uses a 256x256 network, learning rate 0.000442773440394527, tau
0.008242901455713948, batch 256, warmup 50,000 and replay capacity one million. The
local pilot uses four gradient steps, rather than the public experiment's 64, and
our BlueSky collector rather than the public Rust collector. This is not a full
reproduction of the public E27 result, nor an isolated architecture ablation against
the retained two-world, smaller-network reference.

Its 20-scenario result is 98% arrival, 1,148.145 s flight, 37.47 s intrusion,
58.62 s restricted and 23.59 s outside, with 28.5% clean completion. Mean event counts
are 0.58 intrusion, 0.505 restricted and 0.135 sector exits per aircraft. Against the
retained unfiltered reference, flight increases by 101.205 s [22.03, 184.79], intrusion
by 12.92 s [-2.08, 28.50], and restricted time falls by 20.885 s [-41.34, -3.56].
Arrival falls by 1.5 percentage points [-4, 0]; clean completion changes by only
+1 point [-10.5, 12]. This checkpoint is not selected as an overall replacement.
Its smaller training budget does not establish how the configuration would perform
after a full cluster run.

Evidence: runs/sac-ma-public-learner1m-seed2500/completion_audit.json,
validation-final-individual-20, runs/compare-ma-public-larger-347k.json and
runs/compare-ma-neutral-larger-347k.json. No cluster training has been submitted.

## Route-residual controller v1: verified foundation and active training

The new public_route_residual recipe preserves the classical route-following command
and learns bounded changes to it. The normalized heading action is multiplied by
15/45, added to the already clipped classical heading command, and clipped to [-1, 1].
The speed action uses the original normalized speed mapping. The same joint correction
then checks the proposed direct command. This changes an allowed action hook; the
scenario generator, flight dynamics, goal capture and original metrics are unchanged.
The residual bound limits the proposed adjustment, not any later safety correction.

A new SAC actor starts with exactly zero deterministic mean and log standard deviation
-2. Warmup uses Gaussian residuals with standard deviation 0.15. Both initial models
have zero collected transitions and zero optimizer updates. They are reference
controllers, not learned performance results. Native MA uses 124 observations;
native SA uses 131 and observes all ten actual scripted intruders.

The initial MA model exactly matches every one of the nine metrics for all 200 aircraft
records in the twenty-scenario classical reference. All 22,595 policy decisions request
zero adjustment. The native SA model exactly matches every metric for all twenty
controlled-aircraft records, with zero adjustment on all 2,647 decisions, despite
observing ten rather than the transfer reference's nine traffic slots. These results
verify numeric and simulator-level baseline preservation. They do not show an RL gain.

Initial model hashes:

- MA: 0b17e145a34d1a4a26f64115ee91a96e91b669b9a5b7781a842a559c602e2153.
- SA: 70c826f504f0e16b488dfbd46486234b1ff33c45ae592bb1013ece1fd829ea67.

Evidence: runs/route-residual-v1-zero-ma-20, runs/route-residual-v1-zero-sa-20,
runs/route-residual-v1-audit/ma-zero-parity.json, sa-zero-parity.json and
initial-models.json. The original competition evaluator exactly reproduces all
nine metrics on two-scenario checks for both native tracks:
runs/check-harness-route-residual-v1-zero-ma-2.json and
runs/check-harness-route-residual-v1-zero-sa-2.json. All 59 tests pass, including exact reference turns, residual
bounds, cached-reference dispatch, reset, metadata and zero-actor initialization.
MA smoke training performs 120 updates over 400 transitions and reaches 200 updates
at 600 transitions after resume. Native SA smoke training performs 100 updates over
200 transitions. A stale residual mapping is rejected before a resumed run directory
is created. See runs/route-residual-v1-audit/training-verification.json.

Two native-track pilots were started, each with one simulator process, a 64x64
network, learning rate 0.0003, tau 0.005, the public reward/discount recipe and a
one-hour wall-clock cap:

| Pilot | Requested transitions | Seed | Warmup | Batch | Gradient steps per collection | Replay capacity | Checkpoint interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MA residual | 100,000 | 2900 | 5,000 | 256 | 4 | 100,000 | 25,000 |
| SA residual | 40,000 | 2901 | 1,000 | 256 | 1 | 50,000 | 10,000 |

These were requested budgets; completed counts and outcomes are recorded below.
MA counts include inactive aircraft padding. Run directories are
runs/sac-ma-route-residual-v1-100k-seed2900 and
runs/sac-sa-route-residual-v1-40k-seed2901. Both use fresh replay and save initial,
intermediate and final checkpoints. The first trained checkpoint comparisons are recorded below. Final pilot outcomes
and replication remain pending. Compare the full objective vector, including flight
time and missed arrivals, before selecting a final model.

The expanded classical MA evaluation remains in progress. Its additional scenarios
already include missed arrivals and intrusions, so the perfect first-twenty result
must not be generalized to the full procedural distribution. A complete aggregate,
paired comparison with the 200-scenario retained reference and failure analysis are
still required. The final-test stream 2027 and official seed 42 remain unused.


## First trained residual checkpoints: a useful learned candidate

All rows below use twenty development scenarios seeded once with 2026. The MA policy
has 25,000 counted training transitions and 7,996 optimizer updates at the saved
checkpoint. Its learned actor mean weights have norm 0.810000 and mean bias norm
0.015495, versus exact zero for both at initialization. No SA training is performed
for its transferred version. The native SA comparison has 10,000 transitions and
8,999 optimizer updates; differing budgets, seeds and observation layouts prevent
interpreting that comparison as an isolated causal test of MA versus SA training.

| Controller | Arrival | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) | Clean completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MA classical / zero residual | 100% | 1125.155 | 0 | 0 | 0 | 100% |
| MA residual SAC, 25k | 100% | 1035.290 | 0.220 | 0 | 0 | 99% |
| SA classical / zero residual | 100% | 1320.100 | 8.850 | 0.100 | 0 | 75% |
| SA native residual SAC, 10k | 95% | 1346.500 | 14.600 | 0.150 | 0 | 70% |
| SA transferred MA residual SAC, 25k | 100% | 1135.500 | 10.100 | 0 | 0 | 80% |

The MA candidate is 89.865 s faster per aircraft than the classical controller,
with a paired 95% interval of [-123.21, -63.30], a mean reduction of about 8%.
The safety cost is one pairwise intrusion in scenario index 12: KL003 and KL005
have one 22-second intrusion each and both arrive. There are no restricted entries
or sector exits; 19/20 scenarios are entirely clean. The policy proposes nonzero
heading changes on 19,894 of 20,802 decisions and nonzero speed changes on all of
them. Mean absolute proposed heading adjustment is about 4.12 degrees, before any
joint safety correction. The learned residual is therefore active, but this comparison alone does not
establish an advantage over a stronger classical speed rule; see the later matched
speed-reference comparison.

Against the historical 391,520-transition unfiltered policy, the MA candidate lowers
intrusion by 24.33 s [-34.55, -15.02], restricted time by 79.505 s [-102.46, -57.62],
and outside time by 50.84 s [-76.29, -28.12]. Mean arrival is 100% versus 99.5% and
flight time is 11.65 s lower, although the flight difference interval crosses zero.
Clean completion increases by 71.5 percentage points [60.5, 81.5]. These development
means improve the full reported objective vector over that older learned reference.

The transferred SA policy is 184.6 s faster than its classical reference
[-340.70, -74.95], about 14%, with all goals reached. Mean intrusion changes by
+1.25 s [-4.20, 7.95]; restricted time falls from 0.1 s to zero. It has 0.25 intrusion
events per episode and 80% clean completion. It observes the nine closest aircraft
while retaining all ten actual scripted intruders, including in scoring and joint
prediction. The native SA 10k policy instead misses the first scenario and has
higher mean flight, intrusion and restricted times. It is not selected at this stage.
At that checkpoint comparison, both training pilots were continuing to their predeclared budgets.

All intervals are pointwise paired scenario bootstrap intervals. They do not cover
training-seed variation, repeated development-set selection or population guarantees
from observing no failures in twenty scenarios. The trained MA and native SA
checkpoints, and the transferred SA candidate, each exactly match all nine metrics
in two-scenario checks against the original competition evaluator.

Retained candidates:

- runs/candidate-ma-route-residual-v1-25k/model.zip and config.json.
- runs/candidate-sa-route-residual-v1-ma25k/model.zip and config.json.

Both contain identical checkpoint bytes with SHA-256
dcea8a2871e307b9b4ba030ab2d85fece64cb8a20e18517db1ad410a1854f041.
The native SA 10k checkpoint hash is
8774a64c78f1d2f2e4307064e29aeb8e91e4377d1aa4d06be907af32e0e7924c.
The retained model/configuration pairs are independent of later training saves.
They are development candidates rather than final submissions.

Evidence: runs/sac-ma-route-residual-v1-100k-seed2900/validation-25k-individual-20,
checkpoint-25k-audit.json, runs/sac-sa-route-residual-v1-40k-seed2901/validation-10k-individual-20,
checkpoint-10k-audit.json, runs/route-residual-v1-ma25k-sa-transfer/validation-20,
runs/compare-ma-zero-residual-v1-25k.json, runs/compare-ma-public-residual-v1-25k.json,
runs/compare-sa-zero-residual-v1-10k.json,
runs/compare-sa-zero-residual-v1-ma25k-transfer.json and
runs/compare-sa-native10k-ma25k-residual-v1.json. Original-evaluator checks are
runs/check-harness-route-residual-v1-ma-25k-2.json,
runs/check-harness-route-residual-v1-sa-10k-2.json and
runs/check-harness-route-residual-v1-sa-ma25k-2.json.

Expanded MA candidate evaluation is now running for 200 scenarios. The transferred
SA candidate also has a 200-scenario evaluation in progress; the same job then runs
its classical SA reference and a paired comparison. The classical MA 200-scenario
run continues and already contains failures outside the original twenty. These
expansions are development evidence; neither seed 2027 nor official seed 42 is used.
Do not promote the small-set candidate to a competition claim before reviewing the
complete expansions, independent seeds and the final official protocol.


## Verified residual conflict and stronger speed reference

A new optional diagnostic mode advances earlier scenario resets without simulating
the prefix. It is accepted only when the chosen scenario matches all nine saved
metrics for every aircraft and does not advance the scenario RNG during rollout.
For MA residual checkpoint 25k and episode index 12, all differences are exactly
zero. The reference is the original full-prefix twenty-scenario evaluation. This
validates this trace; subsequent uses must pass their own reference comparison.

KL003 and KL005 travel in opposing directions around the southern restricted area.
At t=490 s KL005 has no jointly feasible candidate under the filter's static and
traffic constraints, while the two aircraft are still 16.84 NM apart. Both have a
temporarily feasible selected forecast at t=520 s, but subsequent decisions again
lose feasibility. Both report no jointly feasible command from t=540 through the
encounter. Their sampled minimum separation is 4.654 NM at t=590 s; the original
one-second scoring records one 22-second intrusion for each. Both reach their goals,
and neither enters restricted airspace or leaves the sector.

This failure was anticipated, so merely exposing a conflict flag is unlikely to
resolve this example. Earlier route coordination or a better feasible-command search
is a hypothesis to test. The trace does not prove physical unavoidability, and a
change selected solely to fix this one scenario could regress elsewhere. No filter
or trained policy was changed for this diagnostic. Evidence and a focused plot are
under runs/diagnose-residual-v1-25k-episode12-fast: summary.json, trajectory.csv,
conflict-analysis.json, conflict-analysis.png and analyze_conflict.py.

The evaluator now accepts --goal-speed-action in [-1, 1] for classical goal control.
The default is still exactly zero. The predeclared stronger comparison uses +1,
requesting the maximum original speed increment each decision, then applying the
same route guidance, joint correction and simulator performance envelope. It is a
constant command, not a fixed target speed. MA and SA twenty-scenario comparisons
are running. They will show whether the learned candidate's improvement over the
zero-speed reference survives comparison to a simple acceleration heuristic. Pure
classical policies remain ineligible as standalone RL entries.

All 60 tests pass, including unchanged reference heading values, batch layout,
constant speed values and rejection of non-finite/out-of-range values. CLI checks
also reject incompatible controller combinations before an evaluation is created.
The diagnostic and speed reference leave original scoring and scenario generation
unchanged.

The MA training run has completed 100,000 counted transitions and 38,000 optimizer
updates, with 60,517 live transitions and 39,483 padding, in 2,382.13 s. Its final
model SHA-256 is 87aae77801f720d791ba141b0ef559095a5647181df0263b954b6c975e236ab3.
Completion evidence is runs/sac-ma-route-residual-v1-100k-seed2900/completion_audit.json.
Its final twenty-scenario evaluation and a matched 25k repeat with seed 2902 are
queued in sequence. The repeat uses the saved 25k callback checkpoint, aligning
7,996 optimizer updates with the retained first seed; the final save has four
additional updates and is kept separate.


## MA residual 100k final evaluation: retain the earlier checkpoint

The final twenty-scenario evaluation of seed 2900 reaches every goal, but it is not
an improvement over the retained 25k checkpoint. Mean flight time is 1104.205 s
versus 1035.290 s: +68.915 s, with paired 95% interval [27.168, 119.721]. Intrusion
time rises from 0.220 to 0.850 s, a +0.630 s difference with interval [-0.390, 2.280].
Intrusion events rise from 0.01 to 0.02 per aircraft. Restricted and sector metrics
remain exactly zero, while clean completion falls from 99% to 98% and entirely
clean scenarios fall from 19/20 to 18/20. The higher mean intrusion has a wide
interval; the flight-time regression is the clearer measured difference.

The proposed mean absolute heading adjustment grows from 4.12 degrees at 25k to
10.62 degrees at 100k. The later model causes 151 route replans versus 122 and 3,994
joint-filter interventions versus 3,370. These are diagnostic associations, not an
isolated causal explanation. More training under this configuration does not
monotonically improve objective performance.

The 25k candidate remains selected. An independent run now repeats that budget with
seed 2902 and otherwise matching learner, route, residual and filter settings. It
will evaluate the 25k callback checkpoint rather than the final save, preserving
the first seed's optimizer-update count. No results from that repeat are available
yet. Evidence: runs/sac-ma-route-residual-v1-100k-seed2900/validation-final-individual-20
and runs/compare-ma-residual-v1-25k-final100k.json. These remain development results
on seed 2026; no final-test scenarios were used.


## Faster classical MA reference: no demonstrated learned advantage yet

The completed twenty-scenario test changes only the classical speed command from
0 to +1; route input, joint correction, scoring, scenario seed/sequence and aircraft
dynamics are unchanged. Every aircraft arrives with zero intrusion, restricted-area
or sector metrics. Mean flight time is 1032.775 s and all 200 aircraft outcomes are
clean. There are 75 route replans, 2,912 filter interventions and 151 speed corrections.

Against this faster reference, the retained 25k learned policy is +2.515 s slower
per aircraft, with paired 95% interval [-17.535, 17.961]. Intrusion changes by
+0.220 s [0, 0.660] and clean completion by -1 percentage point [-3, 0]. Thus the
previous improvement over a zero-speed reference is insufficient evidence of a
useful learned contribution. Merely accelerating the classical follower achieves
essentially the same development flight time. This result does not establish that
the two controllers generalize equally or that the classical policy is an eligible
RL submission; broader comparisons and independent seeds are still needed.

Evidence: runs/goal-route-input-v1-fast-joint-ma-20 and
runs/compare-ma-fast-classical-residual-v1-25k-20.json. The matching SA comparison
continues in the same job. A 200-scenario expansion of this faster MA reference is
queued after the native SA final-model evaluation.

The native SA pilot stopped at its one-hour limit after 32,123 counted transitions
and 31,122 optimizer updates, not its requested 40,000 transitions. Wall time was
3,607.18 s including the stop/save boundary. Its final checkpoint SHA-256 is
d10742e80bd1e320dbb203ad684c4f13e08a1d794b3e3830b32ee9e0d182fd00.
A final twenty-scenario comparison with the transferred MA candidate is running.
See runs/sac-sa-route-residual-v1-40k-seed2901/completion_audit.json.

The seed-2902 repeat has now started. Its fully initialized config has no differing
learner, feature, filter or residual fields versus seed 2900. Only the requested
25k budget, seed/world seed and run directory differ. Its training result remains
pending.


## Faster classical SA reference: a possible but unconfirmed gain

The twenty-scenario faster classical SA reference reaches every goal, with mean
flight time 1185.700 s, 0.25 intrusion events, 10.050 s intrusion time, zero restricted
and sector metrics, and 80% clean completion. It uses the same route input and joint
correction as the transferred MA residual controller and still simulates all ten
scripted intruders. Only the classical speed command is changed to +1.

The transferred learned candidate averages 1135.500 s flight time, a reduction of
50.200 s (about 4.2%), but its paired 95% interval is [-217.300, 87.951]. The twenty
scenarios do not establish a reliable flight-time improvement. Intrusion events and
clean completion match exactly, while mean intrusion time differs by +0.050 s
[0, 0.150] because one encounter lasts one extra second. Neither policy records
restricted-area or sector violations.

The earlier approximately 14% flight-time gain against a zero-speed reference must
therefore be reported alongside this stronger comparison. No statistically clear
learned advantage over the faster reference has yet been demonstrated on either
track. The learned controller remains a working candidate and is far safer than
our historical unfiltered learned baseline; the separate question of its added
value over navigation rules remains open.

Evidence: runs/goal-route-input-v1-fast-joint-sa-20 and
runs/compare-sa-fast-classical-residual-v1-ma25k-20.json. The faster classical SA
200-scenario evaluation is now running and will compare with the expanded learned
candidate. The faster classical MA expansion remains queued after native SA final
validation. All scenario comparisons are development evidence, not official results.


## Native SA final checkpoint: not selected

The native SA residual pilot's final 32,123-step checkpoint reaches 95% of goals
on twenty scenarios, with 1397.900 s mean flight time, 0.45 intrusion events,
20.500 s intrusion time, zero restricted/sector metrics and 65% clean completion.
Against the transferred MA 25k candidate, flight time worsens by 262.400 s, with
paired 95% interval [61.546, 496.501]. Intrusion time rises by 10.400 s [-0.400, 26.000]
and clean completion falls by 15 percentage points [-30, 0]. The missed arrival is
scenario index 10. This is a candidate-selection comparison, not a matched-budget
causal comparison of native and transferred training.

The transferred MA checkpoint remains the selected SA candidate. Longer native
training did not provide a replacement. Evidence:
runs/sac-sa-route-residual-v1-40k-seed2901/validation-final-individual-20 and
runs/compare-sa-residual-v1-ma25k-native-final.json. Its completed evaluation job
has now advanced to the faster classical MA 200-scenario expansion. No model or
controller code changed as a result of this unsuccessful checkpoint comparison.


## Expanded development results and a second training seed

The original retained MA-to-SA 25k residual transfer completed 200 scenarios:
98.5% arrival, 1212.885 s flight time, 0.375 intrusion events, 18.535 s intrusion,
0.045 restricted-area events, 2.285 s restricted time, zero sector violations and
65.5% clean completion. The first twenty match all nine previously saved metrics
exactly. The additional 180 have 98.333% arrival, 1221.483 s flight time,
19.472 s intrusion, 2.539 s restricted time and 63.889% clean completion. Missed
arrivals occur at zero-based scenario indices 86, 179 and 181. This is an expanded
development sequence, not a held-out result or an official score.

The original zero-speed classical MA reference also completed 200 scenarios:
98.05% arrival, 1174.192 s flight time, 0.028 intrusion events, 1.074 s intrusion,
0.007 restricted events, 0.3035 s restricted time, 0.0005 sector exits,
0.021 s outside-sector time, 95.5% clean completion and 80.5% entirely clean worlds.
Its first twenty also match exactly. Against the historical unfiltered learned
200-scenario reference, it cuts intrusion by 22.391 s and restricted time by
92.0815 s, but loses 1.15 percentage points of arrival and adds 106.959 s flight
time. It remains a classical comparison, not an eligible standalone RL entry.

Evidence: runs/candidate-sa-route-residual-v1-ma25k/validation-200 and its
validation-200-audit.json; runs/goal-route-input-v1-joint-ma-200 and
runs/goal-route-input-v1-joint-ma-200-audit.json; and
runs/compare-ma-public-classical-route-v1-200.json. Matching faster classical
expansions and the retained learned MA expansion continue separately.

The independent MA training seed 2902 completed 25,000 counted transitions:
13,270 live and 11,730 inactive padding, in 655.43 s. Padding does not enter SAC
replay. Its matched callback checkpoint has 7,996 optimizer updates, SHA-256
70c68ac17d6884bbd02e39237673b414e6275c499db4605e7e41030c051e18bf.
The final save has 8,000 updates and is kept separate. On the same twenty scenarios,
the callback reaches all goals with 1083.655 s flight time, 0.390 s intrusion,
zero restricted/sector metrics and 99% clean completion. Compared with retained
seed 2900, flight time increases by 48.365 s [12.234, 93.206]; intrusion changes
by +0.170 s [-0.660, 1.170]. This reproduces strong small-sample navigation and
safety, but does not demonstrate an advantage over faster classical navigation.
Seed 2900 remains retained. Evidence is under
runs/sac-ma-route-residual-v1-25k-seed2902 and
runs/compare-ma-residual-v1-25k-seeds2900-2902.json.

## Verified SA failure traces and route-choice revision 1

The diagnostic now supports both tracks. The original MA scenario-12 regression
matches all nine final metrics and all 29 original trajectory columns across 940
rows exactly. SA scenarios 86, 179, 181 and 96 also match all nine full-prefix
reference metrics exactly, including total reward. Their scenario RNG states remain
unchanged during the selected rollout. SA skipped-prefix resets explicitly process
the queued route-setup commands before discarding each earlier scenario. Actual
scripted-intruder trajectories are recorded alongside the controlled aircraft.
Evidence: runs/diagnose-residual-v1-sa-episode86, episode179, episode181 and episode96
(each full directory name uses the same prefix), plus
runs/diagnose-residual-v1-ma12-sa-support-check/ma-regression-audit.json.

Scenario 179 exposes a route-selection defect. A 6 km clearance has no path;
the original first-feasible rule chooses a 508.148 km path at 3 km clearance,
although a 120.044 km path exists at the unchanged filter's 2 km margin. Scenario
181 repeatedly changes clearance and loops around a thin obstacle. Scenario 86
spends several minutes alongside scripted aircraft AC8 and later takes a long
detour; its 482 seconds of intrusion are not explained by slow flight alone.
Scenario 96 enters a narrow restricted-area gap after falling back to zero clearance.

New public_route_choice recipes compare geometric path length at clearances
6, 3, 2, 1 and 0 km. They prefer paths with at least 2 km clearance when available,
then choose the widest within 15% of the shortest preferred path. Lower-clearance
paths remain recovery references only when no preferred path exists. All learned
weights, action mappings, joint-filter parameters, scenario generation, dynamics
and original scoring remain unchanged in this ablation. The old recipe remains
available. All 66 controller tests pass.

The selected-case replay uses identical learned weights and rechecks each original
run against its saved reference before testing the new route choice:

| SA scenario index | Original outcome | Route-choice outcome | Intrusion change | Restricted time |
| --- | --- | --- | --- | --- |
| 179 | Timeout, 3000 s | Arrival, 715 s | 0 to 0 s | 0 to 0 s |
| 181 | Timeout, 3000 s | Arrival, 1792 s | 0 to 64 s | 0 to 0 s |
| 86 | Timeout, 3000 s | Arrival, 2460 s | 482 to 482 s | 0 to 0 s |
| 96 | Arrival, 1073 s | Arrival, 1073 s | 0 to 0 s | 90 to 90 s |

All sector metrics remain zero in these four cases. Scenario 179 needs no filter
interventions or replans with the new path. The other outcomes reveal a real
tradeoff: navigation improves, but conflict handling is not fixed and one case adds
an intrusion. These are selected development failures, not a population performance
estimate. Matched twenty-scenario learned/classical comparisons and original-harness
checks are required before replacing the retained controller. No new training has
been performed for this route-choice ablation.

Evidence: runs/route-choice-v1-audit/targeted-results.json and targeted.py, with
original/changed trajectories and captured source. Ablation model directories are
runs/route-choice-v1-ma25k-ma and runs/route-choice-v1-ma25k-sa; both preserve
SHA-256 dcea8a2871e307b9b4ba030ab2d85fece64cb8a20e18517db1ad410a1854f041.


## Complete 200-scenario SA comparison with both classical references

All three controllers use the original route-input revision and the same 200
seed-2026 scenarios. Every expanded first-twenty prefix reproduces its saved nine
metrics exactly. Times are seconds per aircraft.

| SA controller | Arrival | Flight | Intrusion events | Intrusion | Restricted events | Restricted | Sector exits | Outside | Clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical, zero speed command | 97.5% | 1292.855 | 0.390 | 21.190 | 0.055 | 2.300 | 0.010 | 0.665 | 66.0% |
| Classical, maximum speed increments | 99.0% | 1203.585 | 0.410 | 22.125 | 0.055 | 2.530 | 0.005 | 0.345 | 65.0% |
| Retained learned MA25k transfer | 98.5% | 1212.885 | 0.375 | 18.535 | 0.045 | 2.285 | 0 | 0 | 65.5% |

Against the zero-speed reference, learned flight time falls by 79.970 s with paired
95% interval [-129.337, -30.937]. Against the faster reference, it rises by 9.300 s
[-30.541, 49.861], while arrival falls by 0.5 percentage points [-1.5, 0]. Learned
intrusion is 3.590 s lower [-8.050, 0.535] and clean completion is 0.5 percentage
points higher [-4.5, 5.5]. These results still do not establish a clear overall
learned advantage over the stronger reference. Its two missed scenarios are 86 and
179; the learned controller additionally misses 181. The stronger reference's
single sector exit is scenario 61, with 69 s outside and an eventual arrival.

The additional 180 stronger-reference scenarios have 98.889% arrival, 1205.572 s
flight, 23.467 s intrusion, 2.811 s restricted, 0.383 s outside and 63.333% clean
completion. Separate additional-180 paired comparisons are retained in the audit
files. They are further development evidence, not a held-out test.

Evidence: runs/compare-sa-classical-residual-v1-ma25k-200.json,
runs/compare-sa-fast-classical-residual-v1-ma25k-200.json,
runs/goal-route-input-v1-joint-sa-200-audit.json and
runs/goal-route-input-v1-fast-joint-sa-200-audit.json.

## Route-choice v1: matched learned ablation and integration checks

The first twenty unchanged-weight route-choice scenarios are complete on both
tracks. Neither track has restricted-area or sector violations, and all aircraft
arrive. The routing change shortens flights but increases intrusion relative to
the same model with original routing:

| Track | Original flight | Route-choice flight | Paired change [95% interval] | Original intrusion | Route-choice intrusion | Clean completion change |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| MA | 1035.290 | 959.025 | -76.265 [-138.756, -25.814] | 0.220 | 1.240 | 99% to 97% |
| SA | 1135.500 | 984.250 | -151.250 [-353.114, -22.940] | 10.100 | 17.650 | 80% to 70% |

MA intrusion events rise from 0.01 to 0.03 per aircraft and SA events from 0.25 to
0.35. The added intrusion-time intervals are [-0.140, 2.680] s for MA and
[0, 20.300] s for SA. This is a routing efficiency gain with a safety tradeoff;
it is not evidence of improved learning. Matching faster classical route-choice
comparisons are running. The original retained candidate is preserved.

Original-harness two-scenario checks match all nine metrics exactly on both tracks.
A real training smoke performs 400 transitions and 30 optimizer updates; resuming
with compatible replay reaches 600 transitions and 50 updates. Stale route-choice
metadata is rejected before a run directory is created. These are functional
checks, not policy-quality measurements. Evidence:
runs/check-harness-route-choice-v1-ma-2.json,
runs/check-harness-route-choice-v1-sa-2.json,
runs/route-choice-v1-smoke-audit/result.json,
runs/compare-ma-route-choice-v1-20.json and
runs/compare-sa-route-choice-v1-20.json.

A single new MA run now trains on the revised routes with the same seed 2900,
25k budget, 64x64 network, learning rate, warmup and update ratio as the retained
original candidate. It uses fresh replay and evaluates the 25k callback checkpoint.
It is under runs/sac-ma-route-choice-v1-25k-seed2900. No policy-quality result from
this training run is available yet. A visual comparison of the four selected SA
failures is saved as runs/route-choice-v1-audit/targeted-paths.png and .pdf.


## Completed faster classical route-choice comparisons

Twenty matched seed-2026 scenarios are now complete for both faster classical
route-choice references. All controllers below reach every goal; all sector metrics
are zero. Times are seconds per aircraft.

| Track/controller with route choice | Flight | Intrusion events | Intrusion | Restricted events | Restricted | Clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MA classical, maximum speed increments | 953.990 | 0.020 | 1.050 | 0.005 | 0.020 | 97.5% |
| MA retained weights | 959.025 | 0.030 | 1.240 | 0 | 0 | 97.0% |
| SA classical, maximum speed increments | 1030.000 | 0.300 | 12.400 | 0 | 0 | 75.0% |
| SA retained transferred weights | 984.250 | 0.350 | 17.650 | 0 | 0 | 70.0% |

Relative to matching classical route choice, MA learned flight time changes by
+5.035 s [-17.610, 24.435], intrusion by +0.190 s [0, 0.510], restricted time by
-0.020 s [-0.060, 0], and clean completion by -0.5 percentage points [-3, 1.5].
SA learned flight time changes by -45.750 s [-215.860, 102.804], intrusion by
+5.250 s [0, 15.700], and clean completion by -5 percentage points [-15, 0].
No clear learned flight-time advantage is established on either track, and SA
adds one intrusion event in one scenario. The faster geometric routing is a useful
foundation; the learning contribution remains an open experimental question.

Evidence: runs/goal-route-choice-v1-fast-joint-ma-20,
runs/goal-route-choice-v1-fast-joint-sa-20,
runs/compare-ma-fast-classical-route-choice-v1-20.json and
runs/compare-sa-fast-classical-route-choice-v1-20.json. The new matched-budget
training run remains in progress. Neither these twenty-scenario results nor the
four selected failure replays justify an official score or a claim of a winning entry.


## Retained original MA residual: completed 200-scenario evaluation

The original 25k residual candidate reaches 98.55% arrival over 2,000 aircraft in
200 seed-2026 scenarios. It has 29 missed aircraft across 22 scenarios. Mean flight
time is 1108.5855 s, intrusion events 0.028, intrusion time 1.189 s, restricted
entries 0.009, restricted time 0.5745 s, sector exits 0.0005 and outside time
0.0165 s. Clean completion is 95.55%; 77.5% of complete scenarios have every aircraft
arrive without any safety event. The first twenty reproduce all nine saved metrics
exactly. Additional-180 arrival is 98.3889%, flight 1116.7294 s, intrusion 1.2967 s,
restricted 0.6383 s, outside 0.0183 s and clean completion 95.1667%.

Against the zero-speed classical MA reference, flight time falls by 65.6065 s
[-76.8117, -54.1950], but restricted time rises by 0.2710 s [0.0075, 0.5940]. Arrival
changes by +0.5 percentage points [-0.15, 1.2] and intrusion by +0.115 s
[-0.3921, 0.6240]. The faster classical expansion remains necessary before claiming
a learning benefit. Against the historical unfiltered learned reference, intrusion
falls by 22.276 s, restricted time by 91.8105 s and outside time by 50.2535 s,
while flight time increases by 41.3525 s and arrival falls by 0.65 percentage points.
The current system is substantially safer, but missed arrivals remain a material
weakness. Internal arrival/flight aspirations have not been met on this larger set.

Evidence: runs/candidate-ma-route-residual-v1-25k/validation-200 and
validation-200-audit.json; runs/compare-ma-classical-residual-v1-25k-200.json,
runs/compare-ma-public-residual-v1-25k-200.json and
runs/compare-ma-neutral-residual-v1-25k-200.json. No official or held-out stream
was used. Expanded evaluations of revised routing now run on both tracks.

## Training on revised routes: no clear improvement at the matched budget

The seed-2900 route-choice training run completed 25,000 counted transitions:
14,835 live, 10,165 padding, 525.51 s wall time. Its 25k callback has 7,996 optimizer
updates and SHA-256 1ad71cf4f78fdb88d6ea4bc8a2d1e7c473b54a292b400a3e65036f8b1e4310d2.
The final save has 8,000 updates and SHA-256
a424b544993f57302bb6608baade7a1cbdf89f07cf042c239e4cddf5ccf82cbe.
Configuration comparison confirms only routing, requested budget, wall cap and
output directory differ from the original seed-2900 run.

Its twenty-scenario MA callback result is 100% arrival, 956.210 s flight,
0.030 intrusion events, 1.410 s intrusion, zero restricted/sector violations,
97% clean completion and 85% entirely clean scenarios. Compared with unchanged
retained weights using revised routing, flight changes by -2.815 s
[-23.700, 24.416] and intrusion by +0.170 s [-0.510, 1.130]. Against the faster
matching classical controller, flight changes by +2.220 s [-21.145, 28.211] and
intrusion by +0.360 s [-0.120, 1.200]. No clear learning improvement is established;
the earlier retained weights remain in use for routing ablations.

Evidence: runs/sac-ma-route-choice-v1-25k-seed2900/completion_audit.json,
matched_configuration_audit.json and validation-25k-individual-20;
runs/compare-ma-route-choice-v1-trained-25k.json and
runs/compare-ma-fast-classical-route-choice-v1-trained-25k.json.

## Speed-command range: selected probes and twenty-scenario checks

The competition explicitly permits tuning d_speed. A separate experiment changes
its magnitude from 20/3 to 20 knots. The original normalized learned speed output
therefore requests a larger change, and the correction can also select larger speed
changes. This tests combined speed authority, not braking in isolation. Heading,
decision interval, learned weights, routing, filter prediction constants, simulator,
scenarios and scoring remain unchanged. Existing A320 acceleration limits still apply.

Each original selected-case replay exactly reproduces its saved route-choice result
on all nine metrics before applying the new speed range:

| SA index | Flight, original to 20 kt | Intrusion, original to 20 kt | Restricted, original to 20 kt |
| --- | --- | --- | --- |
| 86 | 2460 to 792 s | 482 to 216 s | 0 to 0 s |
| 181 | 1792 to 1871 s | 64 to 62 s | 0 to 0 s |
| 96 | 1073 to 2776 s | 0 to 0 s | 90 to 0 s |
| 179 | 715 to 705 s | 0 to 0 s | 0 to 0 s |

Every aircraft arrives and sector metrics stay zero. The improvement in selected
safety failures includes a very large delay in scenario 96. These selected cases do
not estimate population performance. Evidence and captured source are in
runs/speed-authority-probe-v1/targeted-results.json and probe.py.

The completed twenty-scenario learned tests give MA 100% arrival, 957.460 s flight,
0.030 intrusion events, 1.190 s intrusion, zero restricted/sector metrics and 97%
clean completion. Against original command range with the same route choice,
flight changes by -1.565 s [-15.170, 12.406] and intrusion by -0.050 s
[-1.380, 1.040]; neither is a clear difference. SA reaches every goal with
1024.400 s flight, 0.350 intrusion events, 18.200 s intrusion, zero restricted/sector
metrics and 70% clean completion. Its flight increases by 40.150 s
[-38.650, 151.850] and intrusion by 0.550 s [-0.100, 1.650] versus the same weights
with original command range. The first twenty therefore do not establish a general
benefit from the larger range.

The matching faster classical SA controller gives 100% arrival, 999.100 s flight,
0.300 intrusion events, 12.950 s intrusion, zero restricted/sector metrics and
75% clean completion. The learned controller adds 25.300 s flight
[-134.750, 160.803], 5.250 s intrusion [-0.400, 16.050] and loses five percentage
points of clean completion [-15, 0]. The matching classical MA run continues.

All 73 tests pass, including a narrow-corridor overtake where larger braking makes
a forecast feasible, actual command conversion in knots and metadata rejection.
Original-harness two-scenario checks match all nine metrics exactly on both tracks.
Real training/resume checks reach 400/600 transitions and 30/50 optimizer updates;
stale speed metadata is rejected before evaluation or replay-resume output is
created. The factory explicitly verifies the unchanged physical/scoring parameters.
Evidence: runs/route-choice-speed20-v1-smoke-audit/result.json,
runs/check-harness-route-choice-speed20-v1-ma-2.json,
runs/check-harness-route-choice-speed20-v1-sa-2.json and the compare files named
runs/compare-ma-route-choice-speed20-v1-20.json,
runs/compare-sa-route-choice-speed20-v1-20.json and
runs/compare-sa-fast-classical-route-choice-speed20-v1-20.json.

Both speed-range ablation directories retain checkpoint SHA-256
dcea8a2871e307b9b4ba030ab2d85fece64cb8a20e18517db1ad410a1854f041, with zero training
under the new speed mapping. This experiment is not promoted as a replacement.


## Completed classical MA speed-20 comparison

The faster classical MA route-choice controller with 20-knot increments reaches
all goals across twenty scenarios: 951.330 s flight, 0.020 intrusion events,
1.440 s intrusion, zero restricted/sector metrics, 98% clean completion and 90%
entirely clean worlds. Against it, the unchanged learned weights with the same
mapping add 6.130 s flight [-24.015, 34.751], reduce intrusion by 0.250 s
[-1.780, 0.790], and lose one percentage point of clean completion [-4, 2]. No
clear learned advantage is established. Evidence:
runs/goal-route-choice-speed20-v1-fast-joint-ma-20 and
runs/compare-ma-fast-classical-route-choice-speed20-v1-20.json.

## Reset-time events and the attainable interpretation of clean completion

The original reset calls the unchanged scoring update with zero elapsed time.
This can record an intrusion or restricted-area entry before the policy acts.
A reset-only audit used the same development sequence, checked all initial event
counts against completed evaluation records, and matched saved scenario values
exactly for SA indices 86, 96, 179 and 181 and MA index 12. Tuple/list JSON
representation was normalized for value comparison; no numeric tolerance or
scenario alteration was used. The earlier failed representation comparison was
retained separately and produced no accepted results.

For SA, four of twenty scenarios (indices 1, 4, 8 and 16) start with intrusion
entries. Thus the reset-event upper bound on clean completion is 80%; the original
transferred policy attains that value. All its conflict-bearing episodes in this
prefix already had an initial conflict, but its five total intrusion events exceed
the four initial events. It is incorrect to call all those events or their durations
unavoidable. The revised route-choice controller additionally creates conflicts in
previously clear scenarios and reaches only 70% clean completion.

Across 200 SA scenarios, 25 start with intrusion and two additional scenarios
(54 and 102) start with restricted-area entries. Initial event counts are 25
intrusions and two restricted entries, with zero initial exposure time. The
reset-only clean-completion upper bound is 86.5%; the original controller reaches
65.5%. It develops conflicts in 36 scenarios without an initial intrusion. There is
therefore still substantial room for better control. The bound does not prove an
optimal controller could attain it and is not a replacement scoring metric.

For MA, three of 2,000 aircraft start with restricted-area events: scenario 56
KL0010, scenario 77 KL008 and scenario 195 KL0010. They account for four initial
restricted entries because areas can overlap. None starts with an intrusion or
sector event; the reset-only clean-completion bound is 99.85%. None of the retained
policy's 29 missed aircraft starts with a scored violation, so initial events do
not explain its arrival deficit.

The separate planar geometry audit finds two missed MA goals inside obstacles.
Scenario 103 KL006 is about 23.595 km from the nearest free point; this suggests a
clean arrival within its 5 km capture radius is geometrically unavailable. Scenario
107 KL005 is about 4.686 km from free space, but 6.686 km from space respecting the
filter's 2 km margin. That is a candidate for goal-region planning and an explicit
clearance tradeoff. The other 27 missed goals lie in free space even at a 2 km
margin. These are geometry diagnostics, not modifications to the actual evaluator.

Evidence: runs/initial-state-audit-v1/result.json, sa-reset-metrics.csv,
ma-result.json, ma-reset-metrics.csv and the audit scripts. The 22 MA scenarios with
missed aircraft are saved under ma-failed-scenarios for reference-checked replays.
All reported evaluation scores retain initial events, exposure and failed arrivals.


## Complete 200-scenario faster classical MA comparison

The faster original-route classical reference reaches 99% arrival (20 misses),
1091.867 s flight, 0.033 intrusion events, 1.376 s intrusion, 0.008 restricted
entries, 0.3885 s restricted time, 0.0005 sector exits and 0.033 s outside.
Clean completion is 95.8%, and 82.5% of scenarios are entirely clean. Its first
twenty reproduce all nine saved metrics exactly. The additional 180 have 98.8889%
arrival, 1098.4328 s flight, 1.5289 s intrusion, 0.4317 s restricted,
0.0367 s outside and 95.3333% clean completion.

Against it, the retained learned controller loses 0.45 percentage points of arrival
[-0.90, 0], adds 16.7185 s flight [4.9027, 28.2701], changes intrusion by
-0.1870 s [-0.9030, 0.5290], restricted time by +0.1860 s [-0.0985, 0.5230],
and clean completion by -0.25 percentage points [-1.40, 0.95]. The flight-time
regression is established on this development sequence; a safety advantage is not.
The learned controller therefore has no demonstrated overall advantage over the
stronger classical reference. Evidence:
runs/compare-ma-fast-classical-residual-v1-25k-200.json and
runs/goal-route-input-v1-fast-joint-ma-200-audit.json.

## MA scenario 39: learned slowdown during a long detour

A reference-checked replay matches all nine original metrics exactly. Three of ten
aircraft miss their goals, with no intrusion, restricted-area or sector violation
anywhere in the scenario. All routes retain 6 km clearance. The failed aircraft
slow markedly while traversing a long obstacle detour:

| Aircraft | Minimum CAS (m/s) | Sampled time below 120 m/s | Proposed deceleration decisions | Executed deceleration decisions | Minimum goal distance (km) |
| --- | ---: | ---: | ---: | ---: | ---: |
| KL004 | 75.003 | 1510 s | 123 | 122 | 95.098 |
| KL007 | 67.000 | 1720 s | 117 | 117 | 84.733 |
| KL008 | 95.131 | 1780 s | 103 | 103 | 41.660 |

None of their executed decelerations occurs when the learned proposal requests
nonnegative speed change. Their no-jointly-feasible counts are only 1, 0 and 0.
These observations identify learned speed behavior as a useful target for a causal
ablation, rather than assuming the safety correction forced every slowdown. They
do not alone prove what would happen after changing those commands.

Evidence: runs/diagnose-residual-v1-ma-episode39/summary.json, trajectory.csv,
trajectories.png and speed-analysis.json. A focused next learning experiment can
initialize residual control around the faster classical speed reference and test a
critic warmup, instead of assuming more training with the current zero-speed
reference will solve the problem. No such new learning result exists yet.


## Route-choice SA expansion: completed 200-scenario audit

The unchanged retained MA checkpoint, transferred with the route-choice recipe,
completes 199/200 SA arrivals. Its first twenty scenarios reproduce all nine prior
metrics exactly, and recomputing the full CSV reproduces the reported summary.
Checkpoint SHA-256 remains dcea8a2871e307b9b4ba030ab2d85fece64cb8a20e18517db1ad410a1854f041.

| SA controller, 200 scenarios | Arrival | Flight s | Intrusion s | Restricted s | Outside s | Clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original-route learned transfer | 98.5% | 1212.885 | 18.535 | 2.285 | 0 | 65.5% |
| Route-choice, identical learned weights | 99.5% | 1111.000 | 20.960 | 2.350 | 0 | 64.0% |

The paired flight change is -101.885 s [-153.5559, -58.1599], arrival +1 pp
[-1, 3], intrusion duration +2.425 s [-0.115, 5.3251] and clean completion
-1.5 pp [-5, 2]. Intrusion events rise from 0.375 to 0.420 per aircraft,
change +0.045 [0, 0.095]. Restricted events remain 0.045; restricted duration
changes +0.065 s [-0.010, 0.195]. Intervals are pointwise scenario bootstrap,
not training-seed uncertainty. This is a routing ablation, not additional learning.

The three original timeouts (indices 86, 179, 181) now arrive. A different case,
index 109, times out with 80 s intrusion and 12 s restricted exposure. The additional
180 scenarios alone reach 99.4444% arrival, 1125.0833 s flight, 21.3278 s intrusion,
2.6111 s restricted and 63.3333% clean completion. All reported metrics retain
reset-time events. This efficiency improvement does not establish overall superiority.

Evidence: runs/route-choice-v1-ma25k-sa/validation-200.{csv,json},
validation-200-audit.json beside them, and runs/compare-sa-route-choice-v1-200.json.
The original candidate remains preserved separately.

## Fast-reference learning and actor-warmup experiment

The exact MA scenario-39 trace motivates testing learning from the faster classical
speed reference. The new recipe composes speed as clip(1 + 2*u, -1, 1), retaining
original speed increments and full braking authority. Zero learned output exactly
matches the faster reference in angle-grid tests using float32 and float64 inputs.
Positive residuals saturate at the reference; this is recorded as an explicit new
mapping, not applied silently to the older candidate.

A second arm holds actor parameters fixed for 2,000 initial critic updates.
Critics, target critics and entropy temperature continue learning. This tests an
initial value-learning period; it does not freeze all learner state. Replay-only
warmup remains 5,000 vector transitions in both arms. The residual-learning papers
and their attribution are recorded in PUBLIC_WORK.md; no novelty claim is made
for residual initialization or critic warmup.

Both arms start with exactly identical policy state, including critics, and use
seed 2900, one MA world, 64x64 layers, learning rate 0.0003, tau 0.005,
batch 256, buffer 100,000 and four gradient updates per vector step. The planned
25k callback evaluation uses 7,996 critic updates: 7,996 actor updates without
warmup, versus 5,996 with warmup. The final checkpoint has four additional updates
and is kept separate. Evaluation uses the same twenty development scenarios and
the matching route-choice classical speed-1 reference. Both runs have completed training and twenty-scenario evaluation on both tracks;
measured outcomes and the next validation decision are recorded below.

All 83 tests pass. Actual simulator training holds the actor exactly through
30 updates at 400 transitions while critics change. Resuming with replay reaches
600 transitions and 50 critic updates, correctly totaling 40 held-actor and ten
actor updates; the actor then changes. Legacy checkpoints default to no hold.
Configuration validation rejects missing or stale fast-reference metadata.

Runtime evidence: runs/fast-reference-smoke-v1/runtime-audit.json.
Training arms: runs/sac-ma-fast-reference-v1-25k-seed2900 and
runs/sac-ma-fast-reference-warmup2k-v1-25k-seed2900.
Untrained-reference rollouts are complete on both tracks. All nine metrics,
route-planner statistics, filter statistics and world-decision counts exactly
match the matching classical controller across twenty scenarios: 200 MA aircraft
and twenty SA aircraft. MA averages 953.990 s flight, 1.050 s intrusion,
0.020 s restricted and 97.5% clean completion; SA averages 1030 s flight,
12.4 s intrusion, zero restricted and 75% clean completion. Both reach every goal
on this prefix. These are untrained reference results, not learned performance.

Both tracks also reproduce all nine original-harness metrics exactly over two
development scenarios. Evidence: initial-reference-audit.json in
runs/fast-reference-smoke-v1 and runs/fast-reference-initial-sa-v1, plus
runs/check-harness-fast-reference-initial-{ma,sa}-v1-2.json. Negative warmup and
explicit resume overrides fail before creating output; see cli-validation-audit.json
in the smoke directory. Slurm syntax validation passes with the new optional setting.


Both fast-reference arms completed the full training budget. Without actor warmup:
15,214 live transitions, 9,786 padding transitions and 677.4088 s wall time. With
2,000 held-actor updates: 15,384 live, 9,616 padding and 669.2547 s. Both final models
have 8,000 critic updates; actor counts are 8,000 and 6,000. Both evaluated callback
checkpoints have 7,996 critic updates; actor counts are 7,996 and 5,996. Actor state
changes from initialization in every trained checkpoint. The completion-audit.json
in each run records the exact hashes and counts. Twenty-scenario evaluation against the matching
classical reference is now complete on both tracks. Source bundle v14 contains the tested
implementation; this later training-completion note is a documentation-only update.


## Fast-reference learning: completed twenty-scenario comparisons

All rows use the same development sequence (seed 2026 once, then unseeded resets),
original scoring and per-aircraft inference. Each MA row has 200 aircraft records;
each SA row has twenty. All rows reach every goal; no learned row enters a restricted
area or exits the sector. These are development results, not official scores.

| Track and controller | Flight s | Intrusion events | Intrusion s | Restricted s | Outside s | Clean | All-aircraft-clean worlds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MA matching fast classical | 953.990 | 0.020 | 1.050 | 0.020 | 0 | 97.5% | 85% |
| MA fast-reference SAC, no actor hold | 941.420 | 0.030 | 1.410 | 0 | 0 | 97.0% | 85% |
| MA fast-reference SAC, 2k actor hold | 972.505 | 0.020 | 1.050 | 0 | 0 | 98.0% | 90% |
| SA matching fast classical | 1030.000 | 0.300 | 12.400 | 0 | 0 | 75% | 75% |
| SA transferred SAC, no actor hold | 981.100 | 0.350 | 13.050 | 0 | 0 | 75% | 75% |
| SA transferred SAC, 2k actor hold | 1000.000 | 0.300 | 16.200 | 0 | 0 | 75% | 75% |

The no-hold MA policy changes flight time by -12.570 s [-44.1565, 14.7306]
and intrusion duration by +0.360 s [0, 1.080] against the matching classical
reference. Restricted duration decreases 0.020 s [-0.060, 0], removing one
four-second aircraft exposure. Clean completion changes -0.5 pp [-3, 1.5].
Its SA transfer changes flight time by -48.900 s [-169.450, 17.500] and intrusion
duration by +0.650 s [0, 1.950]; the same fifteen scenarios remain clean.
One SA scenario has an additional intrusion event and thirteen extra seconds of
exposure. These interval endpoints at zero do not establish strict improvement
or deterioration beyond this finite sample.

Compared directly with no hold, the 2k hold changes MA flight time by +31.085 s
[-3.7335, 68.1304], intrusion time by -0.360 s [-1.100, 0.040] and clean completion
by +1 pp [0, 3]. SA flight changes +18.900 s [-50.5512, 109.400], intrusion time
+3.150 s [-7.050, 17.800], and clean completion is unchanged on average but affects
different scenarios. It removes the conflict in SA index 5 but introduces 123 s
of exposure in index 12. No broad advantage from this warmup duration is established.

A matched comparison with the earlier freshly trained zero-speed-reference recipe
has MA flight time 956.210 -> 941.420 s, change -14.790 s [-41.9905, 11.2806],
with identical aggregate intrusion, restricted, outside and clean metrics. The
underlying scenario outcomes are not identical. This comparison changes action
composition while keeping the earlier learning budget and seed.

All intervals are pointwise 95% paired whole-scenario bootstrap with 10,000
resamples; they exclude training-seed uncertainty and selection across experiments.
The version without hold is prioritized for broader validation because it has
shorter observed flights on both tracks, full arrivals and no static exposure in
this sample. This is a screening choice, not proof that warmup cannot help.

The no-hold MA rollout changes 18,386/18,920 heading commands and 1,079 speed
commands relative to its reference; SA changes 1,879/1,971 headings and 62 speeds.
The hold variant changes 19,097/19,541 MA headings and 7,138 speeds, and 1,981/2,009
SA headings and 618 speeds. These are nominal composed commands before the joint
filter; counts show policy activity, not proof of beneficial learned contribution.
All four evaluation summaries were independently recomputed from their CSVs, and
checkpoint hashes match the training audits. SA uses unchanged MA checkpoint bytes
with zero additional training and all ten actual scripted intruders.

No-hold callback SHA-256:
961f525f11bec7a0f417933d209230ca99c5b4b8c00d883233a006b8ee38c4bb.
Hold callback SHA-256:
1b7a174d71bcd4ca6eae77b3e60730f26ae02ea17f5aa4b93fe17a88b06abe25.
Both have 25,000 counted transitions and 7,996 critic updates; actor updates differ
as specified in the training protocol.

Evidence: validation-20.{csv,json} and validation-20-audit.json in both MA training
runs and runs/fast-reference[-warmup2k]-v1-ma25k-sa. Pairwise result files are
runs/compare-{ma,sa}-fast-reference-warmup-v1-{25k,ma25k}-20.json,
runs/compare-ma-fast-reference-v1-vs-zero-reference-trained-25k-20.json, and each
arm's comparison against the matching classical reference.

Next validation: 200 scenarios for the no-hold policy on both tracks, with the
matching route-choice classical speed-1 controllers over the same sequence.
The original route-choice ablation expansion has completed on both tracks. The second development
stream and official protocol remain reserved until a finalist is selected.


## Route-choice MA expansion: completed 200-scenario audit

The earlier routing ablation, using unchanged original 25k weights, completes
1,985/2,000 arrivals over 200 worlds. Its first twenty scenarios exactly reproduce
all nine earlier metrics, and recomputing all CSV records matches the summary.
There are fifteen missed aircraft across ten scenarios, compared with 29 across
22 scenarios for the original routing. Twenty old misses are fixed and six new
misses appear, a net reduction of fourteen. All three scenario-39 slowdowns now
arrive; the routing change itself therefore addresses those selected cases.

| Original 25k learned weights, 200 MA scenarios | Arrival | Flight s | Intrusion s | Restricted s | Outside s | Clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original route input | 98.55% | 1108.5855 | 1.189 | 0.5745 | 0.0165 | 95.55% |
| Route choice | 99.25% | 1020.6840 | 2.370 | 0.6185 | 0.0305 | 94.10% |

Paired changes: arrival +0.70 pp [0.15, 1.35], flight -87.9015 s
[-111.6824, -65.9273], intrusion duration +1.181 s [0.358, 2.106], and clean
completion -1.45 pp [-2.75, -0.20]. Intrusion events increase 0.028 -> 0.050,
change +0.022 [0.007, 0.039]; restricted events increase 0.009 -> 0.016,
change +0.007 [0.001, 0.0135]. Restricted duration changes +0.044 s
[-0.227, 0.2755] and outside duration +0.014 s [-0.0495, 0.0915]. These
pointwise scenario intervals show a clear arrival/time benefit and a conflict
tradeoff. The geometric improvement is not an additional learned-policy result.

The additional 180 scenarios alone reach 99.1667% arrival, 1027.535 s flight,
2.4956 s intrusion, 0.6872 s restricted, 0.0339 s outside and 93.7778% clean
completion. Scenario 195 contains four misses and the only outside-sector record:
KL0010 has 61 s outside and 181 s restricted exposure. The complete fifteen-miss
list is saved in validation-200-audit.json; the original scenario generator and
reset-time event scoring remain unchanged.

Evidence: runs/route-choice-v1-ma25k-ma/validation-200.{csv,json}, its adjacent
validation-200-audit.json and runs/compare-ma-route-choice-v1-200.json. The model
hash remains dcea8a2871e307b9b4ba030ab2d85fece64cb8a20e18517db1ad410a1854f041.

## Exact trace of the fast-reference policy's new MA conflict

The new no-hold policy's scenario 19 is replayed exactly against all nine saved
metrics. KL004 and KL005 each record one intrusion event lasting 36 seconds;
both arrive, and all aircraft avoid restricted areas and sector exits. The
minimum separation sampled every ten seconds is 3.8980 NM at 550 s; this is
not claimed as the exact one-second minimum. Intrusion appears in samples at
540, 550 and 560 s.

Both aircraft's forecasts report a minimum separation below 5 NM by 480 s.
These minima cover all neighbors, so the earliest warning is not labeled by pair
in this trace. From 470 through 570 s, both have static-feasible commands but no
jointly feasible member of the tested finite command set. The forecast therefore
warns of traffic risk before the observed separation loss; an absent warning alone
does not explain the failure. This also does not prove physical inevitability.
Earlier coordination or greater braking authority is a focused next hypothesis,
requiring an actual counterfactual and population checks before any change is kept.

Evidence: runs/diagnose-fast-reference-v1-ma-episode19/summary.json,
trajectory.csv, trajectories.png, conflict-analysis.json and conflict-window.csv.
The diagnostic preserves scenario RNG state during the rollout. Its plot has been
visually inspected. The original-harness checks for this trained checkpoint also
match all nine metrics exactly on two scenarios for both MA and SA; see
runs/check-harness-fast-reference-v1-ma-25k-2.json and
runs/check-harness-fast-reference-v1-sa-ma25k-2.json.


## Isolated emergency-braking counterfactual: rejected

A process-local probe replays the exact fast-reference MA scenario 19, then adds
an internal speed command of -3 (a 20-knot requested decrement through the existing
20/3-knot executor) only when the original finite candidate set has no joint
solution. The policy's normal action composition, ordinary acceleration commands,
heading range, decision interval, forecast parameters, routing and original
simulator/scoring are unchanged. If the original result is jointly feasible, the
probe returns it unchanged. Production source and saved recipes are not modified.

The original arm reproduces all nine reference metrics exactly. The extension
selects stronger braking eight times, but both affected aircraft's intrusion time
increases from 36 to 41 seconds. All goals are still reached, with no restricted
or outside exposure; mean flight time changes 903.1 -> 903.5 s and clean completion
remains 80% in this one ten-aircraft world. The extra option does not solve the
selected failure and is not promoted. This rules out that simple remedy for this
encounter, not every possible braking or coordination strategy.

Evidence: runs/extended-braking-probe-v1/probe.py, extended_choose_command.py,
targeted-results.json, probe-audit.json and provenance/. The result records source,
script, model and scenario hashes, exact baseline checks and all eight emergency
actions. The audit distinguishes normalized LF source hashes from the generated
Windows file's CRLF byte hash. This is a selected-case counterfactual, not a
population estimate or an additional training result. The wider 200-scenario
comparisons for the original fast-reference policy continue unchanged.


## Fast-reference SA expansion: matched 200 scenarios

Both controllers use seed 2026 once and simulate all ten scripted intruders.
The learned MA-to-SA transfer exposes the nearest nine intruders, retains checkpoint
SHA-256 `961f525f11bec7a0f417933d209230ca99c5b4b8c00d883233a006b8ee38c4bb`,
and has zero SA training. This evaluation uses the original ten-second interval.

| Metric | Classical fast route-choice | Learned fast-reference | Learned change [95% paired interval] |
| --- | ---: | ---: | --- |
| Arrival | 100% | 99.5% | -0.5 pp [-1.5, 0] |
| Flight time | 1104.820 s | 1109.240 s | +4.420 [-27.6961, 38.8157] |
| Intrusion events | 0.415 | 0.420 | +0.005 [-0.060, 0.060] |
| Intrusion time | 21.700 s | 21.490 s | -0.210 [-3.000, 2.2701] |
| Restricted events | 0.060 | 0.060 | 0 [-0.015, 0.015] |
| Restricted time | 2.565 s | 2.430 s | -0.135 [-0.630, 0.160] |
| Outside time/events | 0 | 0 | 0 |
| Clean completion | 64% | 62% | -2 pp [-5.5, 1.5] |

Intervals use 10,000 paired whole-scenario bootstrap resamples and are pointwise;
they do not include training-seed uncertainty. The learned controller's only missed
arrival is scenario index 28, with 41 s intrusion and no static violations. The
classical controller reaches every goal. On the additional 180 scenarios, learned
flight time is 1123.4778 s versus 1113.1333 s classical, with 60.5556% versus 62.7778%
clean completion. The favorable flight-time mean on the first twenty does not
persist in the full comparison; no overall learned advantage is established.

Evidence: runs/compare-sa-fast-reference-v1-ma25k-200.json,
runs/fast-reference-v1-ma25k-sa/validation-200.{csv,json}, its -audit.json,
and runs/goal-route-choice-v1-fast-joint-sa-200.{csv,json} and -audit.json.
Both audits verify model hashes where applicable, all summary metrics and exact
first-twenty reproduction of all nine saved metrics.

## Five-second decision probe and implementation

The inspected MA scenario index 19 replays the existing ten-second reference
exactly across all nine metrics. At five seconds, both KL004 and KL005 change from
one intrusion event and 36 s exposure each to zero. All ten aircraft arrive with
zero scored safety events, and mean flight changes from 903.1 to 912.3 s. World
decisions double from 152 to 304. Checkpoint weights, per-command ranges, routing,
filter, simulator dynamics and one-second scoring remain unchanged. Decision
frequency is a permitted competition hook. This is a selected failure, not a
population estimate or proof of general safety improvement.

The experiment also permits more frequent command changes; it is not an isolated
observation-latency test. Evidence is runs/decision-interval-probe-v1/targeted-results.json.
Its script SHA-256 is `bd7b9a2ce53c72aae27a3ad887f80dad5b3a69df84f16c60780d4dcd246f497d`.
The named interval5 recipes save timing metadata and leave old recipes unchanged.
MA and SA ablations in runs/fast-reference-interval5-v1-ma25k-{ma,sa} preserve the
same checkpoint bytes and document zero additional training. Their runtime audit
checks actual five-second decisions with one-second simulation, original 5 NM
separation, 5 km capture, 3000 s limit and all ten actual SA intruders.

A real short training run completes 400 transitions and 30 updates; replay resume
continues to 600 transitions and 50 updates at five seconds. The fresh test run
passes all 89 tests. An earlier run failed during Windows dependency import with
an invalid handle; the successful retry also logged a Windows WMI diagnostic but
finished with exit code zero. No source workaround or result substitution was made.


## Fast-reference independent training seed 2902

The independent run completes 25,000 counted MA transitions: 15,426 live and 9,574
padding, with 8,000 final critic/actor updates in 848.7086 s. The evaluated callback
precedes the last four updates and therefore has 7,996 updates, matching seed 2900.
Its SHA-256 is `069d02fcce6238e68abf71fbe55508106723c944ad0d4e4647e4a4dae88aa84f`;
the final model hash is `4204b0ffccae63d892ae7de95dbc3433ff4a62822ae82943444c6ee9f3812057`.
The SA transfer preserves callback bytes and has zero SA training.

The matched-configuration audit verifies all 70 executable Python source files
between the two training snapshots and all requested learner/controller settings
except seed. Initial hidden actor weights differ, while both deterministic initial
mean outputs are zero. Evaluation began after the optional interval5 code was added;
this run retains the original recipe and ten-second default. Source snapshots are
preserved for both training and evaluation.

| Track/controller | Arrival | Flight | Intrusion events | Intrusion time | Restricted / outside | Clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MA classical | 100% | 953.990 s | 0.020 | 1.050 s | 0.020 / 0 s | 97.5% |
| MA seed 2900 | 100% | 941.420 s | 0.030 | 1.410 s | 0 / 0 s | 97% |
| MA seed 2902 | 100% | 947.620 s | 0.020 | 1.050 s | 0 / 0 s | 98% |
| SA classical | 100% | 1030.000 s | 0.300 | 12.400 s | 0 / 0 s | 75% |
| SA seed 2900 | 100% | 981.100 s | 0.350 | 13.050 s | 0 / 0 s | 75% |
| SA seed 2902 | 100% | 976.600 s | 0.350 | 14.900 s | 0 / 0 s | 70% |

All rows use the same twenty development scenarios with per-aircraft inference.
Seed 2902 versus classical changes flight by -6.370 s [-45.4514, 28.9454] in MA
and -53.400 s [-172.0500, 10.9500] in SA. MA clean completion improves by 0.5 pp
[0, 1.5], reflecting one aircraft whose restricted entry is removed; SA clean
completion declines by 5 pp [-15, 0], reflecting one additional conflict scenario.
This is a mixed result, not proof of an overall learned advantage. Two training
seeds on twenty scenarios do not establish generalization; the paired intervals
still quantify scenario uncertainty conditional on these fixed checkpoints.

Evidence: runs/sac-ma-fast-reference-v1-25k-seed2902/{matched-configuration-audit,
completion-audit,validation-20-audit}.json; runs/fast-reference-v1-seed2902-ma25k-sa/
validation-20-audit.json; runs/compare-{ma,sa}-fast-reference-v1-training-seeds-25k-20.json.
All evaluation summary metrics have been recomputed and checkpoint hashes verified.

The five-second smoke's four negative CLI checks also pass: missing and stale
interval metadata are rejected by evaluation and replay resume before any output
run is created. See runs/decision-interval-smoke-v1/completion-audit.json.
Source bundle runs/cluster/source-v15.zip contains 115 source files, 385,710 bytes,
SHA-256 `12486bd641673897dbdf6a80c88e3e955049c1c3628094319d66775b59084908`.
Its CRC, archive hashes and local hashes were verified at creation. Later result
notes do not alter that immutable source snapshot. No cluster jobs have been submitted.


## Five-second SA control: twenty matched scenarios

The unchanged learned weights reach all twenty goals with 996.600 s flight,
0.300 intrusion events, 12.550 s intrusion, zero restricted/outside exposure and
75% clean completion. The matching five-second classical route follower reaches
all goals with 992.100 s flight, 0.300 intrusion events, 12.500 s intrusion and
75% clean completion. Learned flight changes by +4.500 s [-7.850, 17.450], and
intrusion by +0.050 s [0, 0.150]. These outcomes do not show a learned advantage.

Against the same learned weights at ten seconds, five-second control changes
flight by +15.500 s [-101.3525, 162.7525] and intrusion by -0.500 s [-1.950, 0.400],
with unchanged 75% clean completion and all arrivals. The evidence is mixed and
limited to twenty scenarios. No new training is involved, and the baseline
five-second controller is a classical comparison rather than an RL entry.

The original competition harness reproduces all nine metrics exactly on the first
two scenarios. Summary, model-hash and timing audits are saved with both evaluations.
Evidence: runs/fast-reference-interval5-v1-ma25k-sa/validation-20.{csv,json},
runs/goal-route-choice-interval5-v1-fast-joint-sa-20.{csv,json},
runs/compare-sa-fast-reference-interval5-v1-ma25k-20.json,
runs/compare-sa-fast-reference-interval5-vs10-v1-ma25k-20.json, and
runs/check-harness-fast-reference-interval5-v1-sa-ma25k-2.json.


## Fast-reference SA scenario 28: arrival/conflict counterfactual

The sole missed arrival in the learned 200-scenario SA evaluation is reproduced
exactly across all nine metrics by atc.diagnose. The trace remains outside restricted
areas and the sector boundary, but makes a large loop. Its nearest sampled goal
distance is 42.6613 km at 630 s, before increasing to over 211 km and returning.
The 5 km goal-capture region is never reached by the 3000 s limit.

The fixed-scenario probe compares unchanged learned weights at ten and five seconds
with exactly zero residual output at ten seconds. Zero residual output reproduces
the classical fast route follower's saved outcome exactly across all nine metrics.

| Selected-case controller | Arrival | Flight | Intrusion events | Intrusion time | Restricted / outside |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned, ten-second decisions | No | 3000 s | 1 | 41 s | 0 / 0 s |
| Learned, five-second decisions | No | 3000 s | 1 | 41 s | 0 / 0 s |
| Zero adjustment, ten-second decisions | Yes | 888 s | 5 | 217 s | 0 / 0 s |

Faster decisions do not solve this missed arrival. Removing learned adjustment
reaches the goal but adds 176 s conflict exposure and four events. That outcome
cannot be called a safety improvement. Both learned versions retain the original
single conflict; their large subsequent detour involves traffic corrections and
later static constraints. This interaction is more informative than an aggregate
arrival statistic alone. It does not establish that the learned policy intentionally
plans the whole detour or that a safe arrival is impossible.

At 600 s in the ten-second learned trace, the nominal command predicts traffic
conflict and the finite candidate set has no jointly feasible member; the selected
turn is +45 degrees. By 640 s the attempted return toward the route also predicts
a static violation. The filter continues to reject that return while following a
large detour. These diagnostics describe the tested command set and forecast;
they are not a proof that no physically feasible trajectory exists.

The empirical next-decision heading error versus the fixed 1.5 deg/s command model
averages 0.2283 degrees in the learned ten-second trace (maximum 3.5005 degrees).
This small one-step heading diagnostic does not validate the full 90-second position
forecast or the predictions of other aircraft. No forecasting code was changed.

Evidence: runs/diagnose-fast-reference-v1-sa-episode28/summary.json and trajectory.csv;
runs/decision-interval-sa28-probe-v1/{targeted-results,probe-audit,trajectory-analysis}.json,
the three saved trajectory CSVs and comparison.png. The plot has been visually
checked. Model/script/scenario/trajectory hashes and unchanged scenario RNG are
recorded. This remains selected-case evidence, separate from population results.


## Planned fast-reference continuation to 100k

A bounded continuation is running in runs/sac-ma-fast-reference-v1-100k-continuation-seed3906.
It resumes the 25k final model and original replay, requests 75,000 additional
counted transitions and retains ten-second decisions, the same route/filter,
64x64 learner, batch 256, four updates per world batch and no actor hold.
The fresh continuation scenario seed is 3906. Its maximum training wall time is
2700 s; actual completion must be read from training_summary.json.

The predeclared comparison is the 100k callback on the same twenty development
scenarios against the original 25k callback and matching classical reference.
If the wall limit prevents that checkpoint, report the actual count before deciding
further evaluation. Intermediate 50k/75k checkpoints are saved but are not silently
substituted for the planned 100k outcome. This is a continuation, not independent
training. The starting final model has 8000 updates, whereas the saved 25k evaluation
uses the earlier callback with 7996. Source model/replay hashes and checked loaded
settings are in continuation-protocol.json. No result is claimed yet.


## Five-second MA control: twenty matched scenarios

Both controllers reach all 200 aircraft goals across twenty scenarios. The learned
controller retains exactly the 25k fast-reference bytes, with no five-second training.

| Metric | Classical five-second reference | Learned five-second ablation | Learned change [95% paired interval] |
| --- | ---: | ---: | --- |
| Arrival | 100% | 100% | 0 |
| Flight time | 943.675 s | 937.645 s | -6.030 [-24.0207, 10.8106] |
| Intrusion events | 0.030 | 0.020 | -0.010 [-0.030, 0] |
| Intrusion time | 1.360 s | 1.210 s | -0.150 [-0.450, 0] |
| Restricted events | 0.005 | 0 | -0.005 [-0.015, 0] |
| Restricted time | 0.030 s | 0 | -0.030 [-0.090, 0] |
| Outside events/time | 0 | 0 | 0 |
| Clean completion | 96.5% | 98% | +1.5 pp [0, 4] |
| All-aircraft clean scenarios | 80% | 90% | +10 pp [0, 25] |

Against the same learned weights at ten seconds, five-second control changes
flight by -3.775 s [-21.9902, 12.0256], intrusion by -0.200 s [-1.080, 0.450], and
clean completion by +1 pp [0, 3]. All arrivals and zero static violations persist.
These are favorable small-test means, with uncertainty including no improvement.
An expanded paired MA comparison is justified; they are not a final selection or
proof of competitiveness. SA results remain mixed and the selected SA missed
arrival is not fixed by faster decisions.

Learned evaluation uses 37,592 aircraft decisions and 5,789 world decisions in
693.195 s; classical evaluation uses 37,820 aircraft decisions and 5,832 world
decisions in 585.541 s. These concurrent local wall times are not controlled speed
benchmarks. Both raw summaries and model/timing metadata have been verified.
Evidence: runs/fast-reference-interval5-v1-ma25k-ma/validation-20.{csv,json},
runs/goal-route-choice-interval5-v1-fast-joint-ma-20.{csv,json}, and the paired files
runs/compare-ma-fast-reference-interval5-v1-25k-20.json and
runs/compare-ma-fast-reference-interval5-vs10-v1-25k-20.json.


## Arrival reward 250: controlled preparation

The selected SA scenario-28 replay gives undiscounted total reward -39.1679 for
the learned timeout and -167.8311 for the faster classical arrival with much more
conflict exposure. Holding those trajectories fixed, an arrival bonus of about
166.0804 would equalize their undiscounted returns. The named reach250 recipe
therefore tests a single larger arrival bonus. It changes no safety penalty,
discount, observation, route/filter or control setting. All objective metrics
remain independent of reward. This is a diagnostic motivation, not a competition
score formula or proof that the timeout is optimal under discounted SAC.

Actual replays with reward 250 preserve both trajectory CSVs byte for byte and all
eight objective metrics exactly. The missed arrival's reward is unchanged; the
classical arrival's reward increases by 212.582836 to 44.751736. Only a subsequent
training experiment can establish whether the changed incentive improves behavior.
A higher reward may also increase safety failures or change critic learning through
its value scale, so all outcomes remain part of the comparison.

The planned fresh MA run uses the original matched seed 2900, 25,000 counted
transitions, 5,000 replay-only transitions, batch 256, four gradient updates per
world batch, buffer 100,000, one CPU worker and no actor hold. It will compare the
25k callback on the same twenty development scenarios and transfer those exact
bytes to SA with zero SA training. Fresh replay is required. Training has not yet
started while the CPU is fully occupied by existing runs. Protocol, recipe diff,
return calculation and exact trajectory checks are in
runs/arrival-reward250-sa28-probe-v1/configuration-audit.json.

After adding the arrival-reward recipe and transfer mapping, the complete existing
regression suite passes: 89 tests in 57.16 s, exit code zero. The fixed-policy
reward replay also passes with exact physical metrics and trajectory bytes.


## Fast-reference MA expansion: two hundred matched scenarios

The completed ten-second comparison uses 2,000 aircraft records, seed 2026 once,
and per-aircraft learned inference. Checkpoint SHA-256 remains
`961f525f11bec7a0f417933d209230ca99c5b4b8c00d883233a006b8ee38c4bb`.

| Metric | Classical route-choice reference | Learned fast-reference | Learned change [95% paired interval] |
| --- | ---: | ---: | --- |
| Arrival | 99.5% | 99.35% | -0.15 pp [-0.60, 0.25] |
| Flight time | 1006.470 s | 1000.7025 s | -5.7675 [-18.5083, 7.0171] |
| Intrusion events | 0.0530 | 0.0550 | +0.0020 [-0.0130, 0.0170] |
| Intrusion time | 2.506 s | 2.648 s | +0.142 [-0.584, 0.906] |
| Restricted events | 0.0170 | 0.0185 | +0.0015 [-0.0030, 0.0070] |
| Restricted time | 0.6295 s | 0.641 s | +0.0115 [-0.214, 0.244] |
| Sector-exit events | 0 | 0.0005 | +0.0005 [0, 0.0015] |
| Outside time | 0 | 0.0505 s | +0.0505 [0, 0.1515] |
| Clean completion | 94.2% | 94.1% | -0.1 pp [-1.1, 0.9] |
| All-aircraft clean scenarios | 76% | 76.5% | +0.5 pp [-4, 4.5] |

The learned controller misses 13 aircraft across 11 scenarios; classical misses
10 across 7 scenarios. Five classical misses are fixed and eight new misses appear.
Eleven learned misses have no scored safety violation. The only learned sector exit
is scenario 195 KL0010: 101 s outside, 181 s restricted and a missed arrival. This
aircraft starts in a restricted area, but that does not make its subsequent sector
exit or full exposure inevitable. Scenario 196 KL0010 has 178 s intrusion and
48 s restricted exposure with a missed arrival. All other learned misses remain
free of scored safety events.

On the additional 180 scenarios, learned arrival is 99.2778%, flight 1007.2894 s,
intrusion 2.7856 s and clean completion 93.7778%. Classical values are 99.4444%,
1012.3011 s, 2.6678 s and 93.8333%. The first twenty reproduce all nine earlier
metrics exactly; both full summaries have been recomputed and the model hash checked.
No overall learned advantage is established against the matching classical reference.

Against the preserved original-route learned controller, arrival rises by 0.8 pp
[0.2, 1.45] and flights shorten by 107.883 s [-132.9375, -84.4273], while intrusion
rises by 1.459 s [0.5039, 2.5290]. Against the unchanged-weight route-choice ablation,
flights shorten by 19.9815 s [-31.3036, -7.8956], arrival changes by +0.1 pp
[-0.45, 0.65], and intrusion by +0.278 s [-0.282, 0.893]. These bundled method
comparisons do not isolate a single learning change; the classical comparison is
needed to assess the learned component.

Evidence: runs/compare-ma-fast-reference-v1-25k-200.json,
runs/compare-ma-fast-reference-v1-vs-original-residual-200.json,
runs/compare-ma-fast-reference-v1-vs-route-choice-200.json, and both completed
validation prefixes with their -audit.json files. The learned evaluation takes
5448.2009 s and classical 5186.4751 s while sharing local CPU resources; these are
not controlled inference speed benchmarks.

The reach250 training experiment has now started with the predeclared matched
configuration in runs/sac-ma-fast-reference-reach250-v1-25k-seed2900. Its results
remain pending. Five-scenario development previews for the interval5 candidate
are also being generated, with exact saved-metric checks for every rendered case.


## Completed fast-reference continuation: 100k transitions

The predeclared 100k callback completes its MA twenty-scenario comparison. No
intermediate checkpoint was substituted. Training adds 75,000 counted transitions
in 1763.185 s, bringing the total to 100,000, with 59,852 live and 40,148 inactive
padding transitions. There are 38,000 final critic/actor updates and no actor hold.
The evaluated callback has 37,996 updates and SHA-256
`dc4ad0096cc1c19cf91bada53168716c08d36ebf95014e169186add6f411c648`;
the later final model has 38,000 updates and a different recorded hash. Loaded
model counts, source model/replay and completion evidence are preserved in the
run's continuation-protocol.json and completion-audit.json.

| MA metric, twenty scenarios | Original 25k callback | Continued 100k callback | Matching classical reference |
| --- | ---: | ---: | ---: |
| Arrival | 100% | 100% | 100% |
| Flight time | 941.420 s | 971.005 s | 953.990 s |
| Intrusion events | 0.030 | 0.020 | 0.020 |
| Intrusion time | 1.410 s | 1.000 s | 1.050 s |
| Restricted events | 0 | 0.005 | 0.005 |
| Restricted time | 0 | 0.475 s | 0.020 s |
| Outside events/time | 0 | 0 | 0 |
| Clean completion | 97% | 97.5% | 97.5% |
| All-aircraft clean scenarios | 85% | 85% | 85% |

Relative to 25k, mean flight increases by 29.585 s with a paired 95% interval
[-4.4108, 71.7511], intrusion changes by -0.410 s [-1.160, 0], and restricted
exposure increases by 0.475 s [0, 1.425]. Clean completion changes by +0.5 pp
[-1.5, 3]. Scenario 10 KL004 arrives in 2546 s after 95 s in a restricted area;
this single aircraft accounts for the new restricted exposure. Every stored
summary value is recomputed from all 200 aircraft records and the evaluated model
hash matches. Evaluation takes 558.539 s while sharing the CPU with other work.

The continuation is not promoted: extra training provides no clear overall gain
and introduces a static safety failure. It remains a reproducible ablation, with
no further 200-scenario expansion or SA transfer scheduled. This decision does not
establish that every longer training run will regress.

Evidence: runs/sac-ma-fast-reference-v1-100k-continuation-seed3906/
validation-100k-20.{csv,json}, validation-100k-20-audit.json,
runs/compare-ma-fast-reference-v1-25k-vs100k-20.json, and
runs/compare-ma-fast-reference-v1-100k-vs-classical-20.json.

## Verified development previews for both tracks

The interval5 candidate now has complete animated previews of the first five
seed-2026 development scenarios on both tracks. These cases are a fixed prefix,
not selected for favorable outcomes. The MA preview shows all ten aircraft
arriving in each case, including conflicts in scenario 3 (zero-based index 2).
The SA preview also includes its conflict cases. Headers show running arrivals,
clean arrivals and the original conflict/restricted/outside time metrics.

The new atc.record command loads the same saved configuration and checkpoint as
evaluation, uses the original offscreen renderer, and verifies every aircraft ID
and all nine metrics against the reference CSV. All ten recorded scenarios match
exactly and preserve the scenario random stream. A deliberately mismatched
five-/ten-second configuration is rejected before creating an output directory.
Recording changes presentation only; it does not truncate the simulated episode.
Final headers and representative middle frames have been visually inspected.

Each track contains five individual GIFs, final PNG cards, recording.json and a
combined five-scenario-reel.gif with reel-audit.json. The MA reel has 353 frames,
44.8 s duration and SHA-256
`6b3a140a07a29c1ce46f9228a8c23df0c9eef8a9a35f32a6a566436756d7140f`.
The SA reel has 242 frames, 33.7 s duration and SHA-256
`6083e7a66d275b18127b9f886c382e398b5fe8ff7af6ae85b7ee4d35fd1dcea8`.
Both are 720x828 and replay simulated motion at approximately 200 times real time,
with a two-second final card for each case. Frame counts, duration and file hashes
are verified. They are inspectable prototype previews, not official submission
videos, a final checkpoint selection or a population performance estimate.

Evidence: runs/preview-fast-reference-interval5-v1-{ma,sa}/ and
runs/recording-configuration-negative-audit-v1.json. The complete 89-test regression
suite passed after the reward-recipe change; the later presentation module was
verified through actual one-case and five-case replays on both tracks.


## Arrival reward 250: completed matched training and twenty-scenario results

Fresh seed-2900 training completes 25,000 counted transitions in 606.994 s,
including 15,286 live and 9,714 padding transitions. All 32 initial actor, critic
and target state entries match the original seed-2900 run. Requested and actual
training settings match except the named recipe's arrival bonus, which changes
from 37.41716380974046 to 250. The evaluated callback has 7,996 updates and SHA-256
`f89ea9427a6dc2900286bbe94f39119a150ec2bb10bb589333f8c89c32349f08`.
The final model has 8,000 updates and SHA-256
`51f91759793989326ebcef44f29d134330e05ca6394ef6b43ab664e1f56e866f`.
The same callback bytes transfer to SA with zero SA training, ten actual intruders
and nine observed slots. All completed summaries, counts and model hashes are
verified in each run's audit files.

| Twenty-scenario outcome | MA original reward | MA reward 250 | SA original reward | SA reward 250 |
| --- | ---: | ---: | ---: | ---: |
| Arrival | 100% | 99% | 100% | 100% |
| Flight time | 941.420 s | 1024.675 s | 981.100 s | 1013.900 s |
| Intrusion events | 0.030 | 0.010 | 0.350 | 0.250 |
| Intrusion time | 1.410 s | 0.550 s | 13.050 s | 10.050 s |
| Restricted events/time | 0 | 0 | 0 | 0 |
| Outside events/time | 0 | 0 | 0 | 0 |
| Clean completion | 97% | 98% | 75% | 80% |
| All-aircraft clean scenarios | 85% | 90% | 75% | 80% |

MA scenario 0 KL003 and KL009 time out with no scored safety events. Two aircraft
in scenario 16 each spend 55 s in conflict. Relative to the original reward,
MA flight increases by 83.255 s [32.0345, 142.5410], arrival falls by 1 pp [-3, 0],
and intrusion changes by -0.860 s [-2.200, 0]. Higher clean completion does not hide
the missed arrivals and longer flights. The MA reward candidate is not promoted.

SA retains every arrival and has safety events only in scenarios 1, 4, 8 and 16.
Relative to the original reward, its flight changes by +32.800 s [-30.250, 97.150]
and intrusion by -3.000 s [-8.350, 0]. Against the matching classical controller,
flight changes by -16.100 s [-161.550, 91.9025], intrusion by -2.350 s [-7.050, 0],
and clean completion by +5 pp [0, 15]. These twenty-case means justify a broader
SA comparison, not a claim of superiority. Reward totals are excluded from paired
objective comparisons because the training reward definition has changed.

The predeclared SA expansion uses the unchanged callback on the first 200
seed-2026 scenarios and compares with the already completed classical and original
fast-reference results. It runs only after a two-scenario check of all nine original
harness metrics. There is no additional training or MA expansion. Full metrics,
misses, the additional 180 scenarios and prefix reproduction must be audited.
Held-out seed 2027 and official seed 42 remain unused.

Evidence: runs/sac-ma-fast-reference-reach250-v1-25k-seed2900/{matched-configuration-audit,
completion-audit,validation-20-audit}.json,
runs/fast-reference-reach250-v1-ma25k-sa/{validation-20-audit,validation-200-protocol}.json,
and runs/compare-{ma,sa}-fast-reference-reach250-v1-vs-{original,classical}-*20.json.


## Arrival-reward SA expansion and frozen held-out test

The completed 200-scenario SA development expansion reproduces every metric from
the first twenty scenarios exactly. All 200 arrivals complete; the saved model
hash remains f89ea9427a6dc2900286bbe94f39119a150ec2bb10bb589333f8c89c32349f08.
All summaries are recomputed from the CSV. The earlier original-harness check
matches all nine metrics on two scenarios, including total reward.

| Metric | Classical fast route choice | Original fast-reference learner | Arrival-reward learner |
| --- | ---: | ---: | ---: |
| Arrival | 100% | 99.5% | 100% |
| Flight time | 1104.820 s | 1109.240 s | 1150.585 s |
| Intrusion events | 0.415 | 0.420 | 0.420 |
| Intrusion time | 21.700 s | 21.490 s | 20.695 s |
| Restricted events | 0.060 | 0.060 | 0.050 |
| Restricted time | 2.565 s | 2.430 s | 2.660 s |
| Outside events/time | 0 | 0 | 0 |
| Clean completion | 64% | 62% | 64% |

Relative to classical, the revised learner increases flight by 45.765 s
[11.6494, 78.1251], changes intrusion by -1.005 s [-4.5301, 2.4150], restricted
exposure by +0.095 s [-0.775, 1.190], and clean completion by 0 pp [-4.5, 4.5].
Eleven formerly nonclean classical cases become clean and eleven formerly clean
cases develop violations. Equal aggregate clean completion does not mean identical
failure cases. Eight scenarios have restricted exposure; none exits the sector.

The prior timeout at scenario 28 now arrives in 885 s with 142 s intrusion and
three events. The original learner times out at 3000 s with 41 s intrusion and one
event; classical arrives in 888 s with 217 s intrusion and five events. The revised
learner therefore resolves that arrival with less conflict than classical, but
with more conflict than the timeout. This selected diagnostic is not a population
safety guarantee. The reward-ablation motivation used this development case, so it
must not be called an unseen recovery.

On the additional 180 scenarios, the revised learner reaches all goals with
1165.7722 s flight, 21.8778 s intrusion, 2.9556 s restricted time, zero outside and
62.2222% clean completion. The twenty-case clean rate of 80% did not persist in
the larger sample. The full run takes 2602.798 s while sharing CPU resources.

The revised learner is retained as the frozen SA feasibility candidate for its
completed arrivals, with the measured efficiency cost made explicit. Selection is
based solely on completed development results. Source, full configuration,
checkpoint hash, selection rationale and analysis commitments were saved in
runs/heldout-2027-sa-reach250-v1/protocol.json before launching learned and classical
runs on the same 200 untouched seed-2027 scenarios. The source snapshot covers
117 files. No earlier seed-2027 evaluation CSV was present at the audit. The first
held-out runs are now active, so seed 2027 must no longer be described as unused.

This is a generalization and feasibility check, not a new tuning round. Report all
outcomes, per-aircraft safety/flight tails and paired scenario uncertainty. Do not
select a new checkpoint from held-out outcomes or silently rerun a repaired policy
as if the stream were unseen. MA selection will use its 2026 comparisons only.
The original 1000-scenario seed-42 harness remains unused on both tracks.

Evidence: runs/fast-reference-reach250-v1-ma25k-sa/validation-200.{csv,json},
validation-200-audit.json, runs/compare-sa-fast-reference-reach250-v1-vs-classical-ma25k-200.json,
runs/compare-sa-fast-reference-reach250-v1-vs-original-ma25k-200.json and the frozen
held-out protocol/source snapshot. These newer result notes and COURSE_FIT.md
postdate the immutable source-v17 transfer bundle; it still contains the same
controller, trainer, evaluator and recorder code.
