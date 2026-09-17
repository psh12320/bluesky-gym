# Ten-day feasibility sprint: 15-24 September 2026

The user revised development priorities on 16 September: substantive shared PPO,
then genuine centralized-critic MAPPO, with learning-contribution ablations.
[RL_PLAN.md](RL_PLAN.md) is now the active sprint plan. Classical refinements and
final-report polishing are paused; existing evaluations and fixed benchmarks are
preserved. The earlier milestones below are historical context.

## Objective

Build a working controller and establish a substantial, reproducible improvement in
completion, separation, restricted-area avoidance, and sector containment. Decide
whether to commit to the competition and course project from the evidence on day 10.
Winning is an ambition, not a forecast. Public fork results are useful reference points;
no judge-confirmed leaderboard was found in the material reviewed.

## Schedule

| Days | Work | Evidence required |
| --- | --- | --- |
| 1-2 | Working evaluator, PPO and SAC baselines, cluster throughput | Reproducible runs and complete objective metrics |
| 3-5 | Focused improvements based on failure cases | Paired improvements over a trained baseline |
| 6-8 | Replicate best candidates and test difficult scenarios | Multiple training seeds; no completion collapse |
| 9-10 | Freeze candidate, final evaluation and decision | Full metric table, videos, reproducibility record |

Start with shared SAC and PPO. The public fork review changes the initial priority:
MAPPO is an experiment to justify, not a presumed best algorithm. Prototype geometry
and anticipatory-conflict improvements individually. Avoid broad architecture searches
until a baseline's failure modes are measured.

## Decision gates

- Day 2: both environments run; checkpoint loading and objective evaluation work;
  enough measured rollout throughput to finish planned experiments within the sprint.
- Day 5: a learned policy maintains high completion and clearly reduces at least one
  major safety failure relative to the neutral policy and the trained baseline. If
  completion remains poor, focus on reward scale, horizon, and control mapping.
- Day 8: the improvement survives independent training seeds and common held-out
  scenarios. Reject reductions in risk that come primarily from failing to arrive.
- Day 10: decide from the complete performance vector and failure videos. A useful
  aspirational MA target is >=99.5% completion, <=10 s intrusion time, <=25 s restricted
  time, <=13 s outside sector, and <=1000 s mean flight time. These are provisional
  internal targets informed by public reports, not official qualification thresholds.
  Event counts and clean-completion rates must also be examined; do not hide a safety
  regression behind a weighted sum. Tighten targets if another public result warrants it.

## Evaluation discipline

Training seeds are distinct from development seed 2026 and official seed 42. Compare
policies on the same scenario sequence. The official evaluation is 1000 scenarios,
seeded once at the start; MA yields 10000 aircraft records. Preserve original metrics
and scenario construction. Cluster uncertainty estimates by scenario, since aircraft
in the same world are correlated. Keep a second untouched development stream for the
finalist check, then run the official harness only after checkpoint selection is frozen.

The custom atc.evaluate module is a development evaluator with IDs and provenance.
Submission uses scripts/evaluate_competition.py. Its two permitted integration
functions now load our saved policy configuration; the remaining harness source is
unchanged. Integration checks on development scenarios and the later official run
are distinct requirements.

## Compute

University GPUs: H200, H100, A100, accessed through SSH and Slurm. The supplied route
is shri@sjump.comp.nus.edu.sg to shri@xlogin.comp.nus.edu.sg. Password entry remains
interactive. Partition, account, limits, Python/CUDA environment and job execution need
confirmation on the actual cluster. jobs/train_baseline.slurm is a starting template.
Do not run training on the login node. Request adequate CPUs along with a GPU because
BlueSky simulation is CPU work; one simulator per process.

## Current milestone

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

The original-format SA candidate has completed the 1000-scenario, seed-42
competition harness with 98.5% arrivals and 70.4% clean completion. All 15 misses
and the single 87-second sector exit are retained. The independent audit and
complete table are in SCORING_RESULTS.md. These are local protocol results,
not judge-verified scores. The corrected MA deployment has started its full seed-42 run in
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
arrivals and 68.5% clean completion. All three SA seeds are now complete. MA seed 2902 has completed its 200-scenario replication with 99.20% arrivals and
93.9% clean completion, without a clear advantage over the matching classical
controller. MA seed 2904 has also completed with 99.30% arrivals and 94.45% clean completion. All declared replicas are audited, and none establishes a clear overall learned MA advantage; see REPLICATIONS.md. A
heading-command formatting error was reproduced in one primary MA test scenario.
The opt-in correction is tested separately and is not enabled in the running
frozen comparisons; see HEADING_TRANSPORT.md.

The source-v17 archive retains the historical training handoff. The separate
output/candidates/decimal-heading-v1.zip archive contains the corrected deployment,
both selected models and development checks. Its 108 regression tests passed,
and fresh-extraction checks reproduce both tracks on Windows. See
jobs/EVALUATE_CANDIDATE.md for the matching Slurm evaluation job. No
university job has been submitted; authenticated allocation information is still
needed. All runs so far are local. The four-page report is a preserved development
working draft, described in REPORT.md. Competition readiness still needs the
full corrected evaluations, native Linux verification and a final report with accurate attribution.

Detailed development comparisons and hashes are in PILOT_RESULTS.md. Public source
attribution is in PUBLIC_WORK.md; cluster instructions are in jobs/CLUSTER.md.
