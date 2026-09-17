# Development safety and flight-duration tails

The complete ten-second development evaluations show high mean arrival rates but
concentrated safety failures. These descriptive results use 200 seed-2026 scenarios
per track: 2,000 aircraft outcomes for MA and 200 for SA. They do not inspect the
new held-out runs and do not determine further tuning. Raw objective metrics and
all timeouts are retained.

## Frequency and severity

| Controller | Aircraft with conflict | Mean conflict among affected aircraft | 99th-percentile conflict | Maximum conflict |
| --- | ---: | ---: | ---: | ---: |
| MA classical | 89 / 2000 (4.45%) | 56.315 s | 63.000 s | 214 s |
| MA original-reward learner | 83 / 2000 (4.15%) | 63.807 s | 65.120 s | 264 s |
| SA classical | 63 / 200 (31.5%) | 68.889 s | 201.160 s | 437 s |
| SA original-reward learner | 67 / 200 (33.5%) | 64.149 s | 160.410 s | 437 s |
| SA arrival-reward learner | 66 / 200 (33.0%) | 62.712 s | 177.240 s | 437 s |

MA learning reduces the observed number of aircraft with any conflict relative to
classical control, while increasing mean exposure among affected aircraft and the
observed maximum. Its 95th-percentile conflict is zero because more than 95% of
records are conflict-free; that percentile alone would hide every conflict.
MA pair conflicts appear in both aircraft's records, as required by the original
scoring. These counts are not numbers of independent conflict events.

The frozen SA arrival-reward candidate has restricted exposure in eight cases
(4%), versus ten (5%) for classical. However, its mean restricted exposure among
affected cases is 66.5 s versus 51.3 s, and the maximum is 177 s versus 129 s.
The unchanged mean sector-exit exposure of zero is favorable, but it does not turn
these remaining obstacle and traffic failures into clean completion.

The MA learner has one sector-exit case lasting 101 s, represented by only 0.0505 s
in its global average. That case must remain visible in failure analysis. The
matching classical run has no sector exit on this development sample.

## Flight duration

| Controller | Median | 95th percentile | 99th percentile | Maximum | Missed arrivals |
| --- | ---: | ---: | ---: | ---: | ---: |
| MA classical | 910.5 s | 1756.70 s | 2597.02 s | 3000 s | 10 / 2000 |
| MA original-reward learner | 913.5 s | 1732.10 s | 2588.34 s | 3000 s | 13 / 2000 |
| SA classical | 986.0 s | 1892.05 s | 2704.18 s | 2723 s | 0 / 200 |
| SA original-reward learner | 993.0 s | 1886.75 s | 2714.07 s | 3000 s | 1 / 200 |
| SA arrival-reward learner | 1047.5 s | 1927.15 s | 2739.67 s | 2809 s | 0 / 200 |

The revised SA model removes the timeout but shifts its median and upper flight
quantiles upward. Duration is not excess delay: routes differ in distance and
obstacle geometry. No route-normalized fairness or real-world efficiency claim
follows from this table. Arrival-only flight means are saved as supplementary
analysis; the primary table includes timeouts to avoid hiding failed goals.

## Interpretation and responsible-AI limits

These are descriptive, linearly interpolated empirical quantiles; they have no
confidence intervals. The paired scenario-bootstrap tables in PILOT_RESULTS.md
remain the uncertainty analysis for mean differences. Aircraft within MA worlds
are dependent, and both tails and slot summaries use the same development worlds.

Per-slot summaries are retained for KL001 through KL0010. Learned MA clean rates
range from 92% to 95.5%, versus classical 92.5% to 96.5%. These are simulation slots,
not demographic or airline groups. The observed differences neither establish
fairness nor prove that command-processing order caused a disparity. A causal
claim would require an independently designed order/randomization experiment.
No such experiment is claimed or launched here.

The current method's finite candidate search and finite forecast are not a safety
guarantee. Report the complete means, event frequencies, affected-aircraft severity,
missed arrivals and worst cases together. Keep the remaining risk visible when
explaining the controller to a technical audience. Apply the same descriptive
analysis to frozen held-out results without selecting a new policy from it.

## Reproduction

The standalone analysis script and its input file hashes are under
runs/development-tails-v1/. Run it from the repository with:

```powershell
.venv\Scripts\python.exe runs/development-tails-v1/analyze.py
```

The outputs are tails.json, quantiles.csv and development-tails.{png,svg}. The
four-panel figure shows flight cumulative distributions and the fraction of
aircraft exceeding each conflict duration. Both the figure and its labels were
visually checked. audit.json verifies input hashes, quantile ordering, aircraft
counts, conditional means and weighted slot means. This script reproduces analysis
artifacts; it does not run new scenarios or alter any simulator score.
