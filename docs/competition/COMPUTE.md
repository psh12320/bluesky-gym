# Compute measurements and scaling decisions

A diagnostic cProfile run used the fixed corrected MA deployment on the first two
seed-2026 development scenarios. Both scenarios completed, all nine metrics were
exactly equal to the archived reference, and all 145 files listed in the package
manifest remained unchanged. No model or action mapping was modified.

## What the profile measures

The run took 260.095 s wall time and recorded 246.695 s in the profiler across
50.926 million calls. It includes imports, first-time navigation-cache creation,
two scenario resets, policy decisions, environment execution and validation.
Other local jobs ran concurrently. Profiling overhead and cold startup are part
of these measurements; this is a diagnostic profile, not an isolated throughput
benchmark or a university GPU measurement.

| Region | Cumulative seconds | Calls |
| --- | ---: | ---: |
| Environment step | 134.187 | 573 |
| Policy prediction | 12.191 | 4070 |
| Joint traffic/static filter | 52.249 | 4070 |
| Trajectory forecasts | 33.708 | 8706 |
| Route/residual observations | 23.302 | 4090 |

Policy prediction and environment stepping are separate regions; policy prediction
accounts for 8.3% of their combined measured time. The filter, forecasts and
observation costs are nested within broader calls, so their cumulative times must
not be added as independent costs. A separate 35.298 s navigation-threshold load
belongs to cold initialization.

This supports prioritizing CPU simulation and controller throughput when choosing
cluster allocations. It does not measure gradient-update cost, GPU training speed,
or the effect of multiple workers. The university pilot still needs to time actual
rollouts and training updates with the chosen CPU/GPU allocation and dependency
builds. A larger GPU alone does not remove the CPU work observed in this profile.

## Initial-forecast batching experiment

Source inspection shows that the initial unchanged-motion predictions are generated
one aircraft at a time at each new decision time. The predictor already operates
on arrays for alternative commands. Batching those initial aircraft forecasts is
a concrete candidate for reducing repeated Python/NumPy overhead. An isolated
kernel experiment now verifies the array form against the scalar form for 1,050
batches / 6,150 aircraft forecasts, including zero-speed and heading-wrap edges.
Every coordinate and byte matches. Forty interleaved timing pairs at ten aircraft
give a median paired kernel speedup of 11.94 times. This measures this calculation
only, under concurrent local load; it is not an overall controller speedup.

A separate two-scenario MA controller replay exited with code 1 before producing
its result. The last saved runner state said running, but the tool confirmed exit;
runs/forecast-batch-v1/replay-ma/runner-exit-audit.json records this discrepancy.
No completed replay, measured end-to-end improvement or deployment promotion is
claimed. The 145 extracted package files remain unchanged. The experimental hook
exists only in its isolated process; scoring jobs use the fixed original release.

After resources recovered, a fresh MA retry completed both development scenarios.
All 4,070 actual aircraft forecasts across 573 decision batches, and all goal-capture
lifetimes, matched the scalar calculations byte for byte. All nine final metrics
matched the archived reference exactly. All 145 packaged files remained unchanged.
The retry reused a hashed navigation cache and took 142.324 s including mandatory
scalar verification. This is still not a throughput comparison: the diagnostic
profile had different initialization and instrumentation. The initial incomplete
attempt remains preserved separately. The SA check also completed: 2,189 aircraft
forecasts across 199 decision batches and every goal-capture lifetime matched
byte for byte, with all nine final metrics exactly equal. Its 62.728 s elapsed
time includes scalar verification and is not a speed benchmark. Evidence is in
runs/forecast-batch-v1/replay-ma-retry2/ and replay-sa-retry2/.

A separate descriptive timing experiment completed four MA development replays
in the declared order scalar, batch, batch, scalar. Each uses a fresh extraction
and the same navigation cache, and must match all nine archived metrics. The
per-forecast scalar verification is disabled only for this timing comparison;
both tracks already passed the verification above. Command-processing time and
the active replay interval are recorded separately from startup. The two paired
ratios are reported without a population confidence interval. Concurrent
local scoring load remains a limitation. Results are in timing-ma/.


Any trial must preserve per-aircraft forecast coordinates, goal-capture lifetimes,
stable command order and the selected action. Require exact numerical comparisons
against the original predictor, complete development replay checks, and a measured
profile before promoting it. Keep the currently running scoring code and selected
models fixed. The final held-out and scoring streams remain unavailable for policy
or checkpoint selection.

The raw profile, original protocol, extracted candidate, replay check and full
function timing tables are in runs/profile-candidate-v1/. Its protocol and summary
explicitly retain the timing limitations. Cluster access and initial job templates
are described in jobs/CLUSTER.md; no university job has been submitted.


All four timing replays match all nine metrics exactly. They repeat the same two
development scenarios, giving eight scenario executions rather than eight distinct
worlds. Both modes processed 4,070 actions in each replay. The active interval is
from the first command to the end of the last command; total wall time also includes
startup and validation. Scalar forecast counters are uninstrumented, not evidence
that scalar execution generated no forecasts.

| Order | Controller | Total wall (s) | Active replay (s) | Command processing (s) |
| --- | --- | ---: | ---: | ---: |
| 1 | scalar | 159.542 | 113.635 | 47.973 |
| 2 | batch | 107.670 | 73.734 | 26.634 |
| 3 | batch | 83.878 | 65.566 | 23.893 |
| 4 | scalar | 79.992 | 59.912 | 25.464 |

The scalar/batch active-time ratios are 1.541 in the first pair and 0.914 in the
reverse-order pair. Total wall-time ratios are 1.482 and 0.954, respectively. Thus
one pair favors batching and the other favors scalar execution. The descriptive
median active-time ratio of 1.227 is not a reliable overall speedup estimate.
Command-time ratios also vary substantially: 1.801 and 1.066. These measurements
support retaining the numerically equivalent batching experiment, but do not
establish a repeatable total-runtime benefit or justify a deployment promotion.
Keep both pairs and the initial failed replay in the evidence record. Further
throughput measurements belong in a stable allocation after the scoring jobs,
with the experiment and comparison order fixed before inspecting timings.

## Local resource limit

The offline WSL dependency installer was stopped after free disk space fell below
600 MB and host memory was under pressure. Its partial virtual environment and
temporary files were removed only after the installer had exited. The wheelhouse,
published hashes, Python archive, installation report and failed logs were retained.
The resource-recovery audit records 3,794,120,704 bytes free afterward. The three
Windows scoring/evaluation jobs were preserved. Native Linux simulator validation
has not run; avoid repeating the local installation alongside these jobs.
