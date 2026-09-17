# Waypoint-region routing experiment

This is an additional development experiment. The selected learned models and the
source used by the ongoing full scoring runs remain fixed. It does not use the
seed-2027 or seed-42 outcomes to choose a route, parameter or checkpoint.

## Hypothesis and implementation

The current geometric planner aims at the assigned waypoint centre. Actual arrival
uses the unchanged 5 km capture radius. The experiment searches for reachable route
endpoints within a smaller 4.75 km navigation disk, using the original waypoint as
its centre. Each endpoint must also be less than 4.95 km from that original waypoint
under the competition's own distance calculation. Neither the simulator's waypoint
nor its capture rule changes.

Endpoint candidates include the centre, radial projections from the visibility
vertices and current position, circle/obstacle-edge intersections, and the nearest
free point. A shortest-path search finds the least-cost route to this finite set.
This is an approximate geometric planner, with no claim of optimality over the
entire continuous region or feasibility under aircraft dynamics and moving traffic.

The existing clearance choices, preference for at least 2 km, 15% detour tolerance,
sector margins and joint command filter are retained. Replanning always uses the
original assigned waypoint, rather than treating a previous route endpoint as a
new goal. Observations retain the original goal distance and use its original
bearing when the waypoint centre is directly visible. Otherwise they expose the
route bearing in the existing observation layout.

## Validation and declared screen

Nine targeted geometry checks passed in 4.02 seconds. They cover reachable regions
with blocked centres, wholly blocked or disconnected regions, path clearance,
non-increasing geometric length relative to existing point routes on randomized
cases, scorer-distance validation, clearance preference, original-goal replanning
and invalid inputs. These are planner checks, not evidence of better flight scores.

The predeclared first screen compares two classical controllers on the same twenty
seed-2026 MA scenarios, ten aircraft each: the existing point-goal route and the
experimental goal-region route. Both use five-second decisions, maximum positive
speed commands, the unchanged joint filter and decimal heading serialization.
This isolates the routing change before spending more time on policy training.
No learned advantage can be inferred from this classical comparison.

The runner records and compares all scenario specifications and initial scored
metrics across both arms. It checks that assigned waypoints stay unchanged at every
step, retains all final aircraft records and computes all nine original metrics,
clean completion and paired comparisons. A fresh baseline also permits comparison
with the earlier point-goal development results. Every outcome will be retained;
this twenty-world screen alone will not justify replacing the frozen candidate.

The first attempt stopped before any scenario ran because its validation inspected
BlueSky's performance Proxy type. The retry uses BlueSky's getproxied helper to
inspect the actual implementation. The planner, parameters and declared scenario
set are unchanged. Both attempts are retained. The retry completed both arms;
the measured result is reported below.

## Evidence

The protocol, isolated source extension, tests and original failed attempt are in
runs/goal-region-v1/. Protocol SHA-256:
5e33d56dadf207105ff600d1e8283d8bf5baf8793cd84e7c4ff7534119365371.
The completed attempt is runs/goal-region-v1/retry2/, with its own execution plan,
logs, extracted baseline source and status. The complete candidate archive and
all 145 listed files are verified before and after each arm. No source file in
the selected deployment is edited by this experiment.

## Checks on the two known obstructed development goals

The saved geometries reproduce the earlier distances. In zero-based scenario 103,
KL006's waypoint is 23.5946 km from unrestricted space; neither point-goal nor
region planning finds a route at any tested clearance. In scenario 107, KL005's
waypoint is 4.6858 km from unrestricted space and 6.6858 km from space retaining
the 2 km obstacle margin. Point-goal planning has no route at any clearance.
Region planning finds a route only at zero clearance, ending 4.74976 km from the
original waypoint under the scoring distance calculation.

This does not establish a clean flight or a solution that maintains the 2 km margin.
The existing command filter still evaluates the proposed movement. These are two
selected development diagnostics, not additional population performance evidence.
The geometry records and source-scenario hashes are in
runs/goal-region-v1/known-geometry-audit.json.


The competition requires an RL-based solution. These classical arms isolate one
component; they are not a proposed standalone entry. A promising routing result
would still need integration and evaluation with a trained policy, followed by
broader evidence before replacing a selected candidate. See the
[competition rules](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/00930013219af4c17e509c3efc84a8becd0f3546/docs/competition/COMPETITION.md).


## Completed classical comparison

Both arms finished all 20 worlds / 200 aircraft. Every scenario specification and
initial metric matches across arms; assigned waypoints remain unchanged at every
step. The fresh point-goal baseline reproduces every historical aircraft record
on all nine metrics exactly. The 145 packaged files remain unchanged per arm.

| Metric | Point-goal route | Goal-region route |
| --- | ---: | ---: |
| Arrival | 100% | 100% |
| Flight time (s) | 943.675 | 943.045 |
| Intrusion events | 0.030 | 0.030 |
| Intrusion time (s) | 1.360 | 1.360 |
| Restricted events | 0.005 | 0.005 |
| Restricted time (s) | 0.030 | 0.030 |
| Sector-exit events / outside time (s) | 0 / 0 | 0 / 0 |
| Clean completion | 96.5% | 96.5% |
| All-aircraft clean completion | 80% | 80% |

The only changed physical result is zero-based scenario 8, KL006: flight time is
126 seconds shorter. Across all aircraft this is a 0.630 s reduction, with paired
95% scenario-bootstrap interval [-1.890, 0.000] s. Two further aircraft have only
small reward changes; all arrival and safety records are identical. This is not
a sizeable general improvement on the twenty-world sample. Total reward and all
changed records are preserved in the CSVs and interpretation.json.

Measured evaluation time was 496.139 s for point-goal control and 642.407 s for
region control. Concurrent local load changed, so this is not a controlled runtime
comparison. The selected deployment is not replaced on this evidence.

## Declared learned-policy comparison

learned-protocol.json fixes a subsequent two-arm, twenty-world MA comparison using
the same selected model under each routing choice. It also compares learned and
classical control under both choices. The model, routing parameters and scenario
stream stay fixed; this is transfer without further training. The runner uses
individual aircraft inference and the existing deployment loader's space checks.

The first launch was deferred before creating any run directory because fewer than
1.5 GiB of physical memory were available; learned-preflight.json preserves that
record. After the final training-seed replication finished and memory recovered,
the unchanged runner started at 16:39:42 UTC on 15 September (00:39:42 Singapore
time on 16 September). The sequential comparison is running under learned-screen/.
Its execution-plan SHA-256 is
90f149febfb20f7b3da1c051331e0fc304698698645eb608936bc77f658e1121.
No learned comparison result is available yet. It remains development evidence and
does not establish performance on an unseen scenario stream.


### Completed; further refinement paused

The already-running learned-policy screen finished. Both arms reached 100% arrival
and 98% clean completion on twenty development worlds. Mean flight time was
937.645 seconds for point-goal guidance and 937.680 for region guidance. Results
remain under learned-screen/; comparison SHA-256 is
c8d1ce355e8d3abad1d3ecd048dfc428c335be47e8e9989f0e33947fe55e3fde.
The RL priority change pauses further routing refinements and report polishing.
