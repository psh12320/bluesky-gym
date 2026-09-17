# Frozen held-out evaluation: 15 September 2026

These are local results on a previously unused scenario stream, seed 2027, seeded
once and continued for 200 scenarios per track. They are separate from the repeated
seed-2026 development comparisons. Neither controller selection nor subsequent
controller tuning uses these outcomes. The two tracks have different scenario
configurations and are analyzed separately.

## Single-aircraft track: complete

The arrival-reward-250 SA transfer was frozen before its first rollout at
2026-09-15 09:21:28 UTC. The multi-aircraft candidate was frozen at 11:27:12 UTC,
before the completed SA held-out outcomes were inspected. Selection therefore did
not use the other track's held-out results.

| Metric | Matching classical | Frozen learner | Learner minus classical, paired 95% interval |
| --- | ---: | ---: | ---: |
| Arrival | 97% | 100% | +3 pp [+1, +5.5] |
| Flight time (s) | 1221.205 | 1243.150 | +21.945 [-32.737, +75.652] |
| Intrusion events | 0.335 | 0.335 | 0 [-0.060, +0.060] |
| Intrusion time (s) | 18.635 | 17.670 | -0.965 [-5.150, +3.020] |
| Restricted-area events | 0.030 | 0.030 | 0 [-0.020, +0.020] |
| Restricted-area time (s) | 0.925 | 1.090 | +0.165 [-0.830, +1.425] |
| Sector-exit events | 0.010 | 0 | -0.010 [-0.025, 0] |
| Outside time (s) | 0.465 | 0 | -0.465 [-1.355, 0] |
| Clean completion | 68% | 69% | +1 pp [-4, +6] |

All means include every aircraft, including timeouts. Clean completion requires
arrival and zero values for all six safety-event/time measures. For SA, the
all-aircraft-clean scenario rate is identical to clean completion. Reward is in
the raw outputs but is not compared: the learned and classical recipes use
different arrival rewards.

The learner reaches all six goals missed by classical control and introduces no
new misses. The paired arrival interval excludes zero on this sample. This is a
useful learned contribution. It does not establish an overall safety advantage:
the intrusion-time and clean-completion intervals include zero, and restricted
exposure is higher in the point mean. There are fourteen newly clean scenarios
and twelve that lose clean completion, producing the net two-scenario gain.

## Arrival recoveries and tails

Scenario identifiers below are zero-based CSV identifiers. The learned controller
still has safety exposure on three of the six recovered arrivals.

| Scenario | Classical arrival / flight / intrusion / restricted / outside (s) | Learned arrival / flight / intrusion / restricted / outside (s) |
| --- | --- | --- |
| 6 | Miss / 3000 / 0 / 56 / 0 | Arrive / 841 / 14 / 0 / 0 |
| 7 | Miss / 3000 / 202 / 23 / 0 | Arrive / 1994 / 130 / 0 / 0 |
| 133 | Miss / 3000 / 0 / 0 / 0 | Arrive / 2228 / 0 / 0 / 0 |
| 141 | Miss / 3000 / 162 / 0 / 85 | Arrive / 2647 / 0 / 0 / 0 |
| 152 | Miss / 3000 / 0 / 0 / 0 | Arrive / 2730 / 0 / 0 / 0 |
| 184 | Miss / 3000 / 85 / 0 / 0 | Arrive / 874 / 65 / 0 / 0 |

The learner has intrusion exposure in 58/200 scenarios, versus classical 57/200.
Its mean exposure among affected scenarios is 60.931 s versus 65.386 s; its maximum
is 238 s versus 279 s. Restricted exposure affects six scenarios in each controller,
with a learned maximum of 97 s versus classical 56 s. Mean restricted time among
affected scenarios is 36.333 s versus 30.833 s. Neither controller avoids all risk.

Learned flight median / p95 / p99 / maximum are 1062.5 / 2382.3 / 2740.83 / 2992 s,
versus classical 1057.5 / 2417 / 3000 / 3000 s. These are descriptive aircraft-level
quantiles using linear interpolation, not independently tested tail improvements.
Flight duration does not measure excess delay over a shortest feasible route.
On the 194 scenarios where both controllers arrive, the learner takes an average
57.088 s longer. This is a descriptive common-arrival subset; the principal table
still includes all 200 outcomes, including classical timeouts.

## Reproduction and limits

Evidence is in runs/heldout-2027-sa-reach250-v1/:

- protocol.json: model/configuration/source freeze and predeclared analysis.
- learned.csv and classical.csv, with full metadata and source archives.
- comparison.json: 10,000 paired whole-scenario bootstrap resamples, seed 0.
- completion-audit.json: complete summary recomputation, independent sum checks,
  source/model/configuration verification, all failed arrivals and safety cases.
- audit_completion.py: reruns the audit without changing a controller.

Model SHA-256:
f89ea9427a6dc2900286bbe94f39119a150ec2bb10bb589333f8c89c32349f08.
The current executable sources and both evaluation source archives match all 100
executable/configuration/lock files in the frozen snapshot. Archive CRCs and every
archived file hash pass; package snapshots match. Both CSVs contain exactly the
expected 200 scenario/aircraft identities, and every reported summary and paired
comparison recomputes from the complete records. The generator draws its scenario
from the environment RNG at reset; controller rollout code does not consume that
RNG. Per-scenario geometry was not logged in these completed runs.

Pointwise intervals cover scenario-sampling variation for one selected training
seed. They do not include training-seed uncertainty, correct for multiple metric
comparisons, or demonstrate performance on all possible airspaces. Development
and held-out results must remain labeled separately.

## Multi-aircraft track and next decision

The unchanged 25k callback, deployed at five-second decisions, is frozen with
SHA-256 961f525f11bec7a0f417933d209230ca99c5b4b8c00d883233a006b8ee38c4bb.
Its learned and matching classical 200-scenario held-out evaluations are running;
no complete result is available yet. The selection protocol and source snapshot
are in runs/heldout-2027-ma-interval5-v1/.

The completed SA holdout supports running its unchanged candidate through the
original 1000-scenario, seed-42 harness. That local official-protocol evaluation
is now running in runs/official-sa-reach250-v1/. Source outside the two allowed
integration functions matches the original harness AST; seed, episode count and
scenario/scoring code remain fixed. No result has yet been produced. The MA
seed-42 evaluation remains pending. Local official-protocol results will not be
presented as judge-verified competition scores.

## Frozen-model development previews

The first five development scenarios are recorded for the frozen SA candidate at
runs/preview-frozen-sa-reach250-v1/five-scenario-reel.gif. All nine replay metrics
match the completed development evaluation exactly; model and GIF hashes are in
recording.json and reel-audit.json. The 38.4-second reel includes conflict cases,
with readable outcome labels. These five cases are a demonstration, not an
estimate of population performance or part of the unseen test.

The frozen MA model already has its matching five-second preview at
runs/preview-fast-reference-interval5-v1-ma/five-scenario-reel.gif. Its earlier
replay audit also matches every recorded metric exactly. Earlier SA five-second
previews are historical ablations and do not depict the frozen SA model.

## Runtime checks and pending command-error investigation

The exact frozen models pass runtime binding checks in
runs/frozen-runtime-compliance-v1/audit.json: their simulator step, metric update,
metric initialization and info functions are the original environment methods.
All fixed constructor and scenario-generator parameters match the base defaults.
Only two horizontal action dimensions are exposed. The supplementary
simulation-audit-canonical.json verifies the actual one-second BlueSky timestep,
10-second SA / 5-second MA decisions, compiled geography backend and built-in
conflict resolution disabled. These checks complement the completed replay
comparisons; they do not establish operational safety.

The method uses a shared neural policy and a centralized joint command filter.
The MA filter reads live traffic state and processes aircraft in a stable order;
it is not fully decentralized execution. This distinction must be retained in
the final method description.

During the primary MA held-out run, the simulator logged a heading-command parsing
error while processing scenario index 46 (displayed as scenario 47). The run
continued. The read-only diagnostic in runs/heading-parser-ma47-v1/ reproduces a rejection
of the finite heading 7.843909918392455e-05. BlueSky rejects exponent notation;
plain decimal text with the identical numeric value parses successfully. The
separate corrected replay accepts all 1,925 commands and preserves all eight
physical metrics for that diagnostic scenario. Its reward differs slightly.
See HEADING_TRANSPORT.md. The primary run still uses the original formatter and
must retain its command-error record. Full-run CSV parity with the skipped-prefix
diagnostic remains pending.
