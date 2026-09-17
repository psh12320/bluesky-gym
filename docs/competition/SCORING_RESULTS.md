# Full competition-protocol scoring

The first completed full run uses the frozen single-aircraft model and the
original heading-command formatter. It finished on 15 September 2026 at
14:39:21 UTC. The corrected SA and MA deployments are still being evaluated.
Results here are local runs of the official protocol, not judge-verified scores
or a competition ranking. No checkpoint is selected or changed from these results.

## Original-format SA: all 1000 scenarios

The original harness runs 1000 scenarios, with seed 42 on the first reset and
continued RNG state thereafter. Its source matches the upstream harness outside
the two allowed integration hooks. The simulator, scenario defaults and scoring
rules are unchanged. SA metrics cover the one controlled aircraft per scenario;
all ten scripted intruders remain in the simulation.

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Arrival | 98.5% | 12.1552 percentage points |
| Flight time (s) | 1175.885 | 471.0314 |
| Intrusion events | 0.344 | 0.6274 |
| Intrusion time (s) | 17.386 | 40.0424 |
| Restricted-area events | 0.028 | 0.1767 |
| Restricted-area time (s) | 1.040 | 8.6381 |
| Sector-exit events | 0.001 | 0.0316 |
| Outside-sector time (s) | 0.087 | 2.7498 |

There are 985 arrivals and 704 clean completions (70.4%). The arrival standard
deviation in the table describes individual binary outcomes, not uncertainty in
the estimated mean. A pointwise Wilson 95% interval for arrival is 97.540-99.089%,
conditional on this trained model and the procedural sampling distribution. It
does not include training-seed variation. The original total reward is 229.2013
with standard deviation 50.0501; it is retained in the CSV but is not comparable
across teams with different rewards.

The earlier 200-scenario holdout had 200 arrivals. The larger full-protocol run
has 15 misses, so perfect arrival reliability did not generalize to this sample.
There is no matching classical 1000-scenario seed-42 run here; the controlled
learning comparisons remain the paired seed-2027 results in HELDOUT_RESULTS.md
and REPLICATIONS.md. Public fork tables also have different or incompletely
verified evaluation conditions, so they do not establish a direct ranking.

## Failures and tails

Intrusion exposure occurs in 275 scenarios, averaging 63.222 s among affected
scenarios, with a maximum of 419 s. Its overall 95th and 99th percentiles are
99.000 and 172.080 s. Restricted exposure occurs in 26 scenarios, averaging
40.000 s among affected scenarios, with a maximum of 129 s. Its 99th percentile
is 39.070 s. Mean exposure alone does not describe these longer failures.

The single sector-exit record is zero-based scenario 473: the aircraft arrives
in 2032 s, with 30 s intrusion exposure and 87 s outside the sector. Median
flight time is 1039.5 s, the 95th percentile is 2149.7 s, and the 99th percentile
and maximum are both the 3000 s horizon.

Every missed arrival is retained below. Scenario numbers are zero-based CSV
indices; the finalization slot is not an additional aircraft identity.

| Scenario | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) |
| --- | ---: | ---: | ---: | ---: |
| 196 | 3000 | 0 | 0 | 0 |
| 219 | 3000 | 37 | 0 | 0 |
| 269 | 3000 | 0 | 59 | 0 |
| 285 | 3000 | 0 | 0 | 0 |
| 321 | 3000 | 0 | 129 | 0 |
| 334 | 3000 | 0 | 0 | 0 |
| 403 | 3000 | 0 | 0 | 0 |
| 450 | 3000 | 256 | 0 | 0 |
| 451 | 3000 | 40 | 0 | 0 |
| 478 | 3000 | 0 | 0 | 0 |
| 548 | 3000 | 46 | 0 | 0 |
| 711 | 3000 | 29 | 0 | 0 |
| 798 | 3000 | 0 | 0 | 0 |
| 827 | 3000 | 21 | 0 | 0 |
| 954 | 3000 | 0 | 0 | 0 |

## Reproduction evidence and transport revision

All 1000 CSV indices are present in order. An independent audit recomputed all
nine saved summary metrics, checked the frozen model identity and source archive,
and verified that all 100 original execution files remain unchanged. The run
contains no recorded ArgumentError or SyntaxError diagnostics. The previously
reproduced MA formatting error still motivates the separate corrected deployment;
these original-format results must not be relabeled as corrected results.

- Model SHA-256: f89ea9427a6dc2900286bbe94f39119a150ec2bb10bb589333f8c89c32349f08.
- CSV SHA-256: 306f5ce2f220b6868f4c457fd701b479f81975485c401f8e7f850d2c8ca9fe50.
- Protocol SHA-256: 13bc0241adbbf279c27b93110a5672c8686f5c0c970c862cb3747359181f17d5.
- Completion-audit SHA-256: d224c81c3866fad3b5ebb11f3eb290ba5ac9256dbf4ac174af46e5ef2312db8b.
- Elapsed local scoring time: 10881.512 s, with other simultaneous local work.

The complete CSV, summary, source archive, protocol, log and failure audit are
under runs/official-sa-reach250-v1/. Corrected results will be reported separately
from runs/official-decimal-v1/{sa,ma}/. Their model weights are unchanged, and the
SA comparison will retain every row when both transport revisions have finished.
