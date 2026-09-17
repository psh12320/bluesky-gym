# Public competition work reviewed on 15 September 2026

The GitHub fork endpoint returned 38 public forks, including this team's fork.
All listed branch heads were inspected. Five forks had clear development on the
competition branch or a branch derived from it. A fork is not evidence of a formal
submission. No public leaderboard or judge-confirmed ranking was found in the web
search and repository material inspected. Private/email/Discord submissions were
not accessible.

## CGCooke/bluesky-gym

Reviewed competition branch commit a458870c711679e4fedb764abc1d58c8f4ffdb61,
46 commits ahead of the upstream competition branch. It includes a Rust training
simulator, PPO and SAC experiments, hyperparameter sweeps, and published results.
The author claims simulator parity; we have not independently reproduced that claim
or run their trained policy.

The earlier SAC result has committed CSVs. Their means were recomputed directly from
1000 single-agent records and 10000 multi-agent records and match RESULTS.md:

| Metric | Single-agent | Multi-agent |
| --- | ---: | ---: |
| Completion | 99.0% | 99.6% |
| Intrusion events | 1.013 | 0.2648 |
| Intrusion time (s) | 57.613 | 10.5778 |
| Restricted-area events | 0.452 | 0.4234 |
| Restricted-area time (s) | 42.396 | 40.6224 |
| Sector-exit events | 0.134 | 0.0516 |
| Outside-sector time (s) | 28.177 | 12.443 |
| Flight time (s) | 1040.753 | 983.6411 |

The later E27 experiment trains SAC with staggered decisions and deploys it with
ordinary simultaneous decisions. EXPERIMENTS.md reports official-protocol MA results:
99.93% completion, 9.4 s intrusion time, 28.5 s restricted-area time, 13.2 s outside
sector, and 942 s flight time. These later numbers are self-reported; the referenced
runs/official/e27_ma.csv was not among the committed result files inspected.

The author's scalar score is a private model-selection formula, not an official
competition score. Their reported negative experiments include some reward-shaping,
larger-network, TCPA-sorting, memory, and curriculum configurations. These findings
are specific to their architectures, budgets, and evaluation protocol.

Sources:
- https://github.com/CGCooke/bluesky-gym/blob/a458870c711679e4fedb764abc1d58c8f4ffdb61/results/official/RESULTS.md
- https://github.com/CGCooke/bluesky-gym/blob/a458870c711679e4fedb764abc1d58c8f4ffdb61/results/official/eval_ma.csv
- https://github.com/CGCooke/bluesky-gym/blob/a458870c711679e4fedb764abc1d58c8f4ffdb61/EXPERIMENTS.md

## danielkalmanson/bluesky-gym

Branch daniel/main, commit 55c636b933a8ad79fb9eb3821065daeba756e177. Experiments
include SAC/PPO, separate heading and speed policies, and polygon obstacle features.
The heading-only SAC report gives 90.2% completion, 19.94 s restricted-area time,
135.4 s intrusion time and 1309 s flight time on its reported 1000-scenario evaluation.
It explicitly identifies the trade-off against the neutral policy and reports no
multi-agent transfer result for that experiment. Metrics were read from the report;
the underlying trained model was not independently evaluated.

Source:
https://github.com/danielkalmanson/bluesky-gym/blob/55c636b933a8ad79fb9eb3821065daeba756e177/experiments/reports/sac_nav_polyobs.md

## cyf617/bluesky-gym

Branch cursor/competition-mdp-v1-4e46, commit e3875b02857194d813272465a2311e8b8d5277d8.
Adds conflict features, polygon distances, SAC training and metric-based checkpoint
selection. A committed 20-episode neutral baseline reaches 100% of goals but violates
separation and restricted areas. A subsequent commit records an initial SAC run with
0% arrival and revises its rewards. This is evidence of work in progress, not a strong
final submission. The branch name here identifies the external source exactly.

Source:
https://github.com/cyf617/bluesky-gym/tree/e3875b02857194d813272465a2311e8b8d5277d8

## Other inspected competition work

- michael2992/bluesky-gym: reward prototypes, logging and a policy checkpoint; no
  comparable final table found in the notes inspected.
- Pablomg02/bluesky-gym, competition/pablo-experiments: custom environment hooks
  and a YAML-configured PPO factory; no comparable result table in its changed files.
- SongqiyingYang and LZWuuu: competition branch heads matched upstream at inspection.

## Consequences for our sprint

1. Run SAC alongside PPO immediately; MAPPO is a challenger, not a presumed winner.
2. Maintain completion while reducing safety violations. Neutral goal reaching alone
   is a weak target because aircraft start pointed toward their destinations.
3. Prioritize a specific safety weakness once baseline failures are measured.
4. Compare complete metric vectors on matched scenarios and multiple training seeds.
5. Use the original BlueSky evaluator for final reporting and retain attribution for
   any implementation or method reused from public work.

## Training-budget follow-up

The public EXPERIMENTS.md reports that smaller turns and a frozen speed channel
looked better at 300k transitions but lost at 2M. Its E27 setup uses 16 worlds x 10
aircraft and describes 64 gradient updates per counted cycle (about 0.4 per vector
transition); our initial MA pilots used 0.1. These budgets and collectors are not
identical, and early control comparisons must not be presented as final rankings.

`atc.recipes.public_weights` records the exact reward weights and discount in the
public `explib.CHAMPION_PARAMS` as a reference for our own BlueSky runs. It does not
reproduce the author's complete training system, policy, or reported performance.
The constants retain source attribution. Source:
https://github.com/CGCooke/bluesky-gym/blob/a458870c711679e4fedb764abc1d58c8f4ffdb61/explib.py


## SAC configuration follow-up

The pinned e27_train.py confirms that E27 loads its SAC architecture, learning rate,
batch size, tau, gradient count and warmup from runs/e26_sac/trial_0013/params.json.
Those parameter-file paths returned HTTP 404, but the same pinned repository's
sweep.py identifies CONTROL_TRIAL as the E26 trial-13 recipe and provides the missing
values. The raw file was checked against the recursive Git tree's blob hash.

| SAC setting | E26 trial 13 / E27 reference | Our initial MA baseline |
| --- | ---: | ---: |
| Hidden layers | 256 x 256 | 64 x 64 |
| Learning rate | 0.000442773440394527 | 0.0003 |
| Tau | 0.008242901455713948 | 0.005 |
| Batch size | 256 | 256 |
| Warmup transitions | 50,000 | 10,000 |
| Replay capacity | 1,000,000 | 400,000 |
| Gradient updates per world batch | 64 over 16 worlds | 8 over 2 worlds |

Both MA update schedules are about 0.4 updates per counted aircraft transition.
The sweep fixes the replay capacity and train_freq=1. This resolves the earlier
uncertainty about the SAC learner settings: explib.DEFAULT_PARAMS was a PPO default,
not the source of the E27 SAC architecture. Existing 64x64 runs remain our baseline.
The trainer and cluster template now accept these learner settings explicitly, and
record the actual architecture, learning rate, tau, batch size and warmup in each run.
Source:
https://github.com/CGCooke/bluesky-gym/blob/a458870c711679e4fedb764abc1d58c8f4ffdb61/sweep.py

E27 also uses staggered decision times: individual aircraft keep their configured
control interval while their decisions are spread over the intervening simulator
ticks. Our current controller uses synchronous decisions. The public reward reference is
therefore explicitly not a reproduction of E27's full method or reported performance.
Source:
https://github.com/CGCooke/bluesky-gym/blob/a458870c711679e4fedb764abc1d58c8f4ffdb61/e27_train.py
https://github.com/CGCooke/bluesky-gym/blob/a458870c711679e4fedb764abc1d58c8f4ffdb61/rust_port/staggered.py


## Checkpoint availability in the pinned tree

The complete recursive Git tree at a458870c711679e4fedb764abc1d58c8f4ffdb61 was also
checked for model artifacts. The model archive paths found are under the inherited
scripts/common/results/models_backup directory for other environments; no competition
or E27 checkpoint was found in that tree. This is a statement about that pinned tree,
not about every release or external storage location. Consequently we have not
independently rerun E27's policy. The local evidence record is
runs/public-source-a458870c/model-availability.json. Source:
https://github.com/CGCooke/bluesky-gym/tree/a458870c711679e4fedb764abc1d58c8f4ffdb61


## Residual-policy method references

The residual controller follows established ideas in [Johannink et al.](https://arxiv.org/abs/1812.03201) and [Silver et al.](https://arxiv.org/abs/1812.06298): retain an existing controller and learn a correction. Silver et al. also motivate zero-output initialization and critic-only burn-in. Our current SAC pilots use zero mean initialization but ordinary updates after collection warmup; critic-only burn-in is a possible later ablation, not an implemented feature. These papers concern robotics and do not establish BlueSky competition performance. Our contribution must be assessed through the ATC design and measured results, rather than a claim to invent residual RL.


## Public-source refresh: 16 September 2026, Singapore time

A fresh unauthenticated GitHub API check at 16:09-16:11 UTC on 15 September
(00:09-00:11 Singapore time on 16 September) enumerated 38 public forks on a
complete single page. The repository metadata separately reported 39 forks;
the reason for that difference is not established, so it is not a claim of
39 accessible public projects. The branch lists of the five previously developed
forks were retrieved. The three previously pinned branches for CGCooke,
danielkalmanson and cyf617 still point to exactly the reviewed commits above.
This refresh does not identify a newly verified competition submission or ranking.
It is not a fresh code review of every branch across all 38 forks.

The upstream competition branch remains at
00930013219af4c17e509c3efc84a8becd0f3546. Its competition rules match the local
original after explicit UTF-8 decoding and newline normalization. An initial
comparison mistakenly used Windows cp1252 for the local Markdown; its false
mismatch and the corrected comparison are both retained. Recent public issue/PR
metadata was also saved, including the existing geography-backend issue #61.
No competition rule or scoring change was found in this bounded refresh.

Sources: [public fork listing](https://api.github.com/repos/TUDelft-CNS-ATM/bluesky-gym/forks?per_page=100&sort=newest),
[upstream competition revision](https://api.github.com/repos/TUDelft-CNS-ATM/bluesky-gym/branches/AI4REAL-NET-Competition),
and [rules at that revision](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/00930013219af4c17e509c3efc84a8becd0f3546/docs/competition/COMPETITION.md).
The raw public responses, request URLs, hashes and correction audit are in
runs/public-refresh-20260916-v1/. Previous model/result interpretations remain
conditional on their authors' reported evaluation settings.


## Training-budget calibration from the pinned public source

The cached E21 account reports a 10-million-counted-step PPO run, with its selected
checkpoint at seven million after extending a two-million-step trajectory. It
also reports large variation across PPO seeds. The E27 SAC script sets three
million counted steps with sixteen ten-aircraft worlds. These are author reports,
not independently reproduced policies or verified competition rankings. Our live
transition counts differ from their counted budgets and are not directly equivalent.

This supports treating our million-live-step runs as budget diagnostics, retaining
longer curves when useful, and reporting every replication seed. It does not justify
adopting only the best seed or assuming their settings transfer to our simulator.
The attempted live browser fetch returned cache misses. The cached experiment
Markdown exactly matched its recorded Git blob; the three cached Python files
matched after explicit CRLF-to-LF normalization. The original failed raw-byte check
and its correction are recorded in runs/public-budget-review-20260916-v1/review.json.
