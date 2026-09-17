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
training runs are needed: MA seed 2904, and arrival-reward-250 seeds 2902 and 2904.
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

Each case has its own folder under runs/replication-frozen-v1/. state.json records
stage commands and completion, with complete stage logs, original source snapshots,
model hashes, configuration matching and metric audits. The runner verifies
independent initial hidden actor weights, zero initial mean actions, the exact
checkpoint update count and unchanged primary model hashes.

The seed-2902 MA and SA replicas have completed their twenty-scenario development
panels and are running their 200-scenario replications. Seed-2904 cases are declared
but have not started. The completed small-panel results are:

| Frozen method, training seed | Arrival | Flight (s) | Intrusion (s) | Restricted / outside (s) | Clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| MA, primary 2900 | 100% | 937.645 | 1.210 | 0 / 0 | 98% |
| MA, replication 2902 | 100% | 960.060 | 1.230 | 0 / 0 | 98% |
| SA, primary 2900 | 100% | 1013.900 | 10.050 | 0 / 0 | 80% |
| SA, replication 2902 | 100% | 1062.850 | 10.000 | 0 / 0 | 80% |

All values are on the same seed-2026 twenty-scenario panel for their track. MA
means cover 200 aircraft; SA means cover 20. Neither replica loses arrivals on
this panel, and both have longer mean flights. These are small development
comparisons, not a completed generalization or training-variability conclusion.

The new SA replica completed exactly 25k transitions, comprising 14,974 live and
10,026 padded transitions, in 537.149 s of local training. Its callback hash is
87e374be18cea1f85e235374c6522d489cc3b5438eb871a06eb124c1135e995e. A bookkeeping
assertion initially treated omitted and explicit-zero actor warmup as different;
actual warmup was zero in both. The audit was corrected, the failed audit record
was preserved, and evaluation reused the completed checkpoint without retraining.
The remaining configuration fields and exact update counts match the primary.

All replication is local until authenticated Slurm allocation information is
available. No university job has been submitted.
