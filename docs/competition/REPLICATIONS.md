# Training-seed replication of the frozen methods

The plan was declared at 2026-09-15 11:49:02 UTC in
runs/replication-frozen-v1/protocol.json. It quantifies training variability and
does not reopen candidate selection. Both primary models remain training seed
2900, with their previously frozen weights and deployment configurations.

## Fixed experiment

| Item | Multi-aircraft method | Single-aircraft method |
| --- | --- | --- |
| Training | Shared MA SAC, fast route-reference residual | Shared MA SAC, same architecture with arrival reward 250 |
| Budget | 25,000 counted transitions | 25,000 counted transitions |
| Checkpoint | Fixed 25k callback, 7,996 optimizer updates | Fixed 25k callback, 7,996 optimizer updates |
| Training decision interval | 10 s | 10 s |
| Deployment | MA, 5 s; no additional training | SA, 10 s; no SA training |
| Additional training seeds | 2902 and 2904 | 2902 and 2904 |
| Evaluation | 20 development scenarios, then 200 scenarios from seed 2027 | Same episode counts and stream seeds in the SA environment |

The MA seed-2902 model already exists and is reused unchanged. Three additional
training runs were declared and have completed: MA seed 2904, and arrival-reward-250
seeds 2902 and 2904.
All use one worker, 64-by-64 hidden layers, batch size 256, 5,000 warmup transitions,
four gradient updates per vector step, replay capacity 100,000, learning rate
0.0003, target update coefficient 0.005 and zero initial actor-hold updates.
The administrative wall limit is disabled so every replica can reach the same
fixed transition budget. This does not change the training algorithm.

The callback is saved before the final four updates. The final model has 8,000
updates and is retained, but it is not substituted for the predeclared callback.
Inactive MA aircraft padding is counted by the trainer; actual live and skipped
transition totals are saved for each seed. Different seeds can have different
live-experience counts despite the same counted-transition budget.

## Interpretation

Seed 2027 has already been evaluated by the primary SA controller, and the primary
MA evaluation was running when this plan was declared. Replica evaluations on that
stream are therefore explicitly described as replication on an already used
scenario stream. They will not be presented as a new untouched test. No recipe,
architecture or checkpoint selection is made from those results; seed 42 is not
used to select among replicas.

Every physical metric, event count, arrival rate and clean-completion rate will
be reported for every training seed. Per-seed paired scenario intervals compare
to the same track's classical controller. These intervals quantify scenario
variation, not training-seed uncertainty. With only three training seeds, report
individual values, their mean and range, and descriptive sample standard deviation.
Do not use three seeds to claim precise population-level training uncertainty.

Adverse results and incomplete or failed runs remain visible. The analysis does
not choose the best replica, drop a weak seed, or replace the primary submission
model. A broad replication failure would weaken the feasibility conclusion and
must be discussed in the report.

## Execution and evidence

Status update: all declared replications are complete and audited. The final
combined summary below supersedes the intermediate completion notes retained here.

Each case has its own folder under runs/replication-frozen-v1/. state.json records
stage commands and completion, with complete stage logs, original source snapshots,
model hashes, configuration matching and metric audits. The runner verifies
independent initial hidden actor weights, zero initial mean actions, the exact
checkpoint update count and unchanged primary model hashes.

Both seed-2902 replicas completed their twenty-scenario development panels. Their 200-scenario replications have now both completed. SA seed 2904 has completed its full 200-scenario replication. MA seed 2904 completed
training and its development panel; its 200-scenario evaluation started at
13:58:37 UTC. The completed small-panel results are:

| Frozen method, training seed | Arrival | Flight (s) | Intrusion (s) | Restricted / outside (s) | Clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| MA, primary 2900 | 100% | 937.645 | 1.210 | 0 / 0 | 98% |
| MA, replication 2902 | 100% | 960.060 | 1.230 | 0 / 0 | 98% |
| MA, replication 2904 | 100% | 929.885 | 1.290 | 0.065 / 0 | 97.5% |
| SA, primary 2900 | 100% | 1013.900 | 10.050 | 0 / 0 | 80% |
| SA, replication 2902 | 100% | 1062.850 | 10.000 | 0 / 0 | 80% |
| SA, replication 2904 | 100% | 939.250 | 12.250 | 0 / 0 | 80% |

All values are on the same seed-2026 twenty-scenario panel for their track. MA
means cover 200 aircraft; SA means cover 20. None of these replicas loses arrivals on this panel. Both seed-2902 replicas
have longer mean flights; SA seed 2904 has shorter flights and greater intrusion
exposure. MA seed 2904 has shorter flights, one restricted-area event totaling
13 seconds, and 97.5% clean completion. These are small development comparisons, not a completed generalization
or training-variability conclusion.

The new SA replica completed exactly 25k transitions, comprising 14,974 live and
10,026 padded transitions, in 537.149 s of local training. Its callback hash is
87e374be18cea1f85e235374c6522d489cc3b5438eb871a06eb124c1135e995e. A bookkeeping
assertion initially treated omitted and explicit-zero actor warmup as different;
actual warmup was zero in both. The audit was corrected, the failed audit record
was preserved, and evaluation reused the completed checkpoint without retraining.
The remaining configuration fields and exact update counts match the primary.

MA seed 2904 completed exactly 25,000 counted transitions (15,507 live and
9,493 padded), with 8,000 final optimizer updates, in 706.640 s. Its fixed
callback has 7,996 updates and SHA-256
b80405b8f9a1d759f600ccb5fb097a800fbf1a9fbfa8fa9ca212581942109e56.
Independent initialization and matched configuration passed the audit.

All replication is local until authenticated Slurm allocation information is
available. No university job has been submitted.

## First complete SA replication

Training seed 2902 completed all 200 seed-2027 scenarios. Its arrival rate is
98% (196/200), versus 100% for primary seed 2900 and 97% for the classical
controller. Mean flight time is 1250.800 s, intrusion time 18.350 s, restricted
time 1.495 s, outside time 0 s, and clean completion 68%. The primary values
are 1243.150 s, 17.670 s, 1.090 s, 0 s, and 69%, respectively.

Against the paired classical controller, the replica's arrival improvement is
+1 percentage point with a 95% scenario-bootstrap interval of [-1.5, +4.0].
Its clean-completion change is zero, with interval [-5.5, +5.5] percentage points.
This is weaker evidence than the primary result and does not establish a
consistent learned-policy advantage. The remaining declared seeds are retained;
no replacement of the primary model or selection of a favorable replica is made.

All nine summary metrics were recomputed from the 200 saved aircraft records.
The complete state and paired comparison are in
runs/replication-frozen-v1/sa-seed2902/. The original failed bookkeeping audit
remains preserved alongside its successful recovery.

SA seed 2904 trained for exactly 25,000 counted transitions (14,191 live and
10,809 padded), with 8,000 final optimizer updates, in 561.174 s. Its fixed
callback has 7,996 updates and SHA-256
8b99fa0662d2c7cb304afbbd99c91eab1c64863c91184f4d03c958436a5e5143.
Its independent initialization and matched configuration passed the same audit.
The 20-scenario panel has all arrivals, 939.250 s mean flight, 12.250 s intrusion,
zero restricted/outside exposure and 80% clean completion. The full replication has since completed, as reported below. These small-panel
values were not used to select a new model.

## Completed SA track: all three training seeds

Every declared SA training seed has completed the same 200 seed-2027 scenarios.
These results use the original heading formatter and are separate from corrected
full-protocol scoring. All summaries and paired comparisons have been recomputed.
The primary candidate remains seed 2900; no replica replaces it.

| Metric | Classical | Primary 2900 | Replica 2902 | Replica 2904 |
| --- | ---: | ---: | ---: | ---: |
| Arrival | 97% | 100% | 98% | 99% |
| Flight (s) | 1221.205 | 1243.150 | 1250.800 | 1198.750 |
| Intrusion events | 0.335 | 0.335 | 0.335 | 0.330 |
| Intrusion time (s) | 18.635 | 17.670 | 18.350 | 16.080 |
| Restricted events | 0.030 | 0.030 | 0.040 | 0.025 |
| Restricted time (s) | 0.925 | 1.090 | 1.495 | 1.610 |
| Sector-exit events | 0.010 | 0 | 0 | 0.005 |
| Outside time (s) | 0.465 | 0 | 0 | 0.320 |
| Clean completion | 68% | 69% | 68% | 68.5% |

Across the three training seeds, arrival has mean 99%, range 98-100%, and sample
standard deviation 1 percentage point. Clean completion has mean 68.5%, range
68-69%, and sample standard deviation 0.5 percentage points. Mean flight is
1230.900 s (sample SD 28.104), intrusion 17.367 s (SD 1.165), restricted exposure
1.398 s (SD 0.273), and outside exposure 0.107 s (SD 0.185). These are descriptive
statistics from three training runs, not precise population uncertainty bounds.

Every seed has a higher arrival point estimate than classical control, but the
paired arrival intervals are +3 pp [+1, +5.5], +1 pp [-1.5, +4], and +2 pp
[0, +4.5], respectively. Only the primary interval excludes zero. All three
clean-completion intervals include zero. The favorable arrival means do not
establish an overall safety advantage or universal arrival reliability.

Seed 2904 misses zero-based scenarios 141 and 186 and has one sector-exit event
with 64 seconds outside. Both failures and the outside exposure are retained.
Its checkpoint remains 8b99fa0662d2c7cb304afbbd99c91eab1c64863c91184f4d03c958436a5e5143.

The audited track summary is runs/replication-frozen-v1/sa-completed-track.json.
It includes every original metric, all training seeds, per-seed paired comparisons
and source/model/CSV identifiers. MA seed 2902 has since completed, as reported below. MA seed 2904 remains
incomplete, so the final combined replication table still requires that declared case.

## Completed MA replication: training seed 2902

The first additional MA seed completed all 200 seed-2027 scenarios at 14:31:28 UTC.
This is replication on the same previously evaluated stream, using the original
heading formatter. The primary model remains unchanged; the last declared MA
replica, seed 2904, is still running.

| Metric | Classical | Primary 2900 | Replica 2902 |
| --- | ---: | ---: | ---: |
| Arrival | 99.35% | 99.35% | 99.20% |
| Flight (s) | 994.0735 | 995.1085 | 999.1990 |
| Intrusion events | 0.0510 | 0.0560 | 0.0510 |
| Intrusion time (s) | 2.536 | 2.671 | 2.347 |
| Restricted events | 0.0210 | 0.0215 | 0.0180 |
| Restricted time (s) | 0.9275 | 0.9000 | 0.9435 |
| Sector-exit events | 0 | 0 | 0 |
| Outside time (s) | 0 | 0 | 0 |
| Clean completion | 93.8% | 93.9% | 93.9% |
| All-aircraft clean completion | 73.0% | 73.5% | 72.5% |

The replica misses 16 arrivals across 12 scenarios, versus 13 missed arrivals
for both the primary model and classical reference. Its arrival difference from
classical control is -0.15 percentage points, with a paired 95% scenario-bootstrap
interval of [-0.45, +0.10]. Mean flight increases by 5.1255 s [-3.9914, +14.0916],
intrusion exposure changes by -0.189 s [-0.7850, +0.3821], and restricted exposure
by +0.016 s [-0.0635, +0.1080]. Clean completion changes by +0.10 percentage
points [-0.85, +1.00]. Every nontrivial interval includes zero. The point means
therefore do not establish a consistent learned advantage over the matching
classical controller. Zero observed sector exits are not a population guarantee.

An independent audit recomputed both saved evaluation summaries and the paired
comparison, verified both source archives against their recorded hashes, checked
all 100 original frozen execution files, and read the actual checkpoint's 25,000
transitions, 7,996 updates and training seed from its saved data. Every primary
model hash is unchanged. The audit is
runs/replication-frozen-v1/ma-seed2902/completion-audit-independent.json, SHA-256
80b721da6f97bdfe1fa6115141ccb0fbe4692a47e0f075a1913401e90c2b5774.
All nine original metrics and every failed aircraft record remain available.
The complete three-seed MA distribution will be reported only after seed 2904
finishes; this partial table does not omit or replace that run.


## Completed MA track and final combined replication

The final replica, MA training seed 2904, completed all 200 scenarios / 2000
controlled aircraft. Its independent audit recomputed all nine metrics and paired
comparisons, checked both source archives, the unchanged 100 original execution
files and the exact 25k / 7996-update checkpoint. Fourteen arrivals are missed in
twelve scenarios; every record is retained. Primary candidates remain unchanged.

| Metric | Classical | Primary 2900 | Replica 2902 | Replica 2904 |
| --- | ---: | ---: | ---: | ---: |
| Arrival | 99.35% | 99.35% | 99.20% | 99.30% |
| Flight (s) | 994.0735 | 995.1085 | 999.1990 | 1001.9095 |
| Intrusion events | 0.0510 | 0.0560 | 0.0510 | 0.0490 |
| Intrusion time (s) | 2.536 | 2.671 | 2.347 | 2.527 |
| Restricted events | 0.0210 | 0.0215 | 0.0180 | 0.0195 |
| Restricted time (s) | 0.9275 | 0.9000 | 0.9435 | 0.9560 |
| Sector-exit events | 0 | 0 | 0 | 0 |
| Outside time (s) | 0 | 0 | 0 | 0 |
| Clean completion | 93.8% | 93.9% | 93.9% | 94.45% |
| All-aircraft clean completion | 73.0% | 73.5% | 72.5% | 73.5% |

Against classical control, seed 2904 changes arrival by -0.05 percentage points
with paired 95% scenario-bootstrap interval [-0.50, +0.40] points; flight by
+7.8360 s [-3.1626, +19.1424]; intrusion by -0.0090 s [-0.7060, +0.7741];
restricted exposure by +0.0285 s [-0.0895, +0.1400]; clean completion by
+0.65 points [-0.40, +1.65]; and all-aircraft clean by +0.50 points
[-4.00, +5.00]. Every nontrivial interval includes zero. As with the other MA
seeds, these results do not establish an overall learned advantage.

Across three MA training seeds, arrival has mean 99.2833%, range 99.20-99.35%
and sample standard deviation 0.0764 percentage points. Clean completion has
mean 94.0833%, range 93.90-94.45%, and sample SD 0.3175 points. Mean flight is
998.7390 s (SD 3.4238), intrusion 2.5150 s (SD 0.1623), and restricted exposure
0.9332 s (SD 0.0294). All three observed zero sector exits. These descriptive
statistics use three models on the same 200 worlds; they are not a precise
training-population interval or evidence from 600 independent scenarios.

All three declared training seeds per track are now included in
runs/replication-frozen-v1/summary.json and per-seed.csv. The combined summary
was completed at 16:39:23 UTC on 15 September 2026 (00:39:23 Singapore time on
16 September). Its SHA-256 is
0ee7b5b0fa7929a49b2701bca8923a4be7e0e8ae6bb5381567a77ccbf2f6165f.
The final replica audit is ma-seed2904/completion-audit-independent.json,
SHA-256 b748fccffae6075129b28ba0dadf4bfb2a40630d7b9fb27da508252c6f4e89c8.

SA arrival remains the more promising learned effect, with 100%, 98% and 99%
versus classical 97%; only the primary paired interval strictly excludes zero.
MA completion and safety are strong in absolute terms on this stream, but a
sizeable learned contribution over the matching classical controller remains
unproven. The full corrected seed-42 evaluations are separate and still running.
