# Project handoff — 17 September 2026

## Read this first

This is a working reinforcement-learning research project with trained shared-policy PPO, genuine centralized-critic MAPPO and SAC. It is **not yet a demonstrated competitive RL solution**. The strongest overall local controller is still the frozen classical benchmark. Some learned policies improve particular measures relative to their own initialization, but lose arrivals, safety or efficiency elsewhere. Do not present controller-assisted scores as evidence that learning caused those scores.

The owner's priority is a substantive RL contribution with strong performance during a short feasibility sprint. Competition submission and adoption as the CS4246 course project are conditional on progress. The six-person team is registered on Canvas. Preserve the classical benchmark and all completed/failed experiments; classical-only refinements and final-report polishing are paused.

Repository: [psh12320/bluesky-gym](https://github.com/psh12320/bluesky-gym/tree/AI4REAL-NET-Competition). Branch: `AI4REAL-NET-Competition`. Upstream base before this research commit: `00930013219af4c17e509c3efc84a8becd0f3546`. Local source: `C:\Users\Shricharan\VSCodeProjects\airtrafficcontrol\bluesky-gym`. The parent folder is not a second repository. Do not force-push. Use descriptive branch/commit names without a `codex` prefix. Retain upstream attribution and accurate contribution records.

This document supersedes the **status assessments** in older notes, including `RL_PLAN.md`, `SPRINT.md`, `REPORT.md` and `runs/rl-priority-status-20260917-v44.json`. They remain intact as history. Historical “running” and “ready” labels do not prove current job status.

## Published evidence

Source, tests, job scripts, experiment runners, protocols, CSVs, logs, summaries and comparisons are committed. Binary checkpoints, demonstrations, immutable source bundles and historical replay files are supplied through the [handoff release](https://github.com/psh12320/bluesky-gym/releases/tag/research-handoff-20260917).

- [Experiment index](docs/competition/EXPERIMENT_INDEX.md): every retained local run/record group, including smoke checks and failed attempts, with evidence entry points.
- [Artifact manifest](docs/competition/artifacts/manifest.json): file sizes, SHA-256 hashes, archive mappings and exclusions.
- [Artifact instructions](docs/competition/ARTIFACTS.md): downloading/restoring files without overwriting different evidence.
- `scripts/package_research_handoff.py`: inventory and byte-only archival; does not load models or replay objects.
- `scripts/restore_research_artifacts.py`: checksum-checked restoration into the repository.

Virtual environments, downloaded dependencies, generated simulator/worker caches, duplicate expanded source archives, the course-guideline reference, and copied third-party source mirrors are excluded. Exclusions are recorded. Original local files have not been deleted. Public-fork citations/research findings remain. Historical replay **bytes** are preserved: do not deserialize, train from, or resume those `.pkl` replay buffers.

Remote cluster checkpoints and raw CSVs have **not** been downloaded. Published cluster evidence consists of the owner's terminal transcripts, parsed receipts, local launch code and source bundles. Do not claim independent raw-data verification. Download them using existing instructions before relying on them for a report.

## Competition and measurement contract

The upstream [competition specification](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md) is authoritative. The inspected version describes procedural two-dimensional airspaces, A320s, five restricted areas, five-nautical-mile separation, a five-kilometre arrival region and a 3,000 simulated-second horizon. Native scoring samples physical outcomes each simulated second; current learned controllers act every five seconds.

Single-agent (SA): one controlled aircraft and ten scripted intruders. Multi-agent (MA): all ten aircraft controlled. A shared actor applied separately to each aircraft is a valid MA policy. MAPPO adds a centralized training critic, not privileged actor information at inference.

There is **no published single weighted competition score** in the inspected rules. Report waypoint completion, flight time, intrusion events/time, restricted-area events/time and sector-exit events/time outside, alongside novelty and documentation. Native `total_reward` is not comparable across different training rewards. “Clean completion” is our additional diagnostic: arrival without recorded safety violations. It is not an official leaderboard scalar. Our advancement screen (including a five-percentage-point clean gain and safety checks) is an internal decision rule, not organizer qualification.

Inference latency is not listed as a scoring term or stated inference budget. Flight time is simulated travel time, not execution time. Report compute/throughput separately. Strong MA performance is valuable; it does not guarantee victory.

The official harness uses 1,000 scenarios, seed 42, seeded once and continued, with only two permitted environment/policy hooks changed. Scoring, final scenario generation and final population remain fixed. Observation/reward/action design and curricula are permitted subject to those constraints. Submission includes code/model, a short report and a five-scenario video; inspected deadline: 30 November 2026. Recheck rules before submission.

## Accounting and evaluation safeguards

1. New RL budgets count **live aircraft transitions**. Inactive fixed slots are separately counted as padding and excluded from losses, advantage normalization and fresh SAC replay. Ten aircraft slots are not ten independent simulator worlds.
2. Historical SAC budgets usually count vector slots including padding. Historical “25k” is not 25,000 live samples. Keep live/count/optimizer/epoch/supervised counters separate.
3. Compare final policies to their **own initial policy** and the fixed classical benchmark. Goal-relative actions and categorical initialization already encode navigation; initial arrival performance is not learning.
4. Behavioral cloning (BC) is supervised learning. Its initial policy is already trained even when the RL counter is zero. Separate supervision from additional PPO/MAPPO updates.
5. Scenario, source, model and package identities are recorded. Reproduce old models with their frozen source/dependencies. Do not relabel current-code output as an old experiment.
6. Resample whole paired worlds for scenario uncertainty, not 200 aircraft as independent observations. Training-seed variability is separate. Twenty reused development worlds are insufficient for final generalization claims.
7. Stream **20260 is heavily reused development**. New-RL streams **20301 and 20302 remain reserved** in retained protocols. Historical 2026, 2027 and 42 have been used; 2027/42 are not globally untouched.
8. Preserve failures, interruptions, overshoot and wall-limited budgets. Do not silently select the best inspected checkpoint.
9. Linux/Windows classical results differ slightly on the twenty-world panel. Use within-platform paired controls until raw-data/source checks explain this.

## System map

| Location | Purpose |
| --- | --- |
| `atc/` | Historical SAC/controllers, route guidance, static/traffic projections, metrics and submission integration; see `atc/README.md`. |
| `atc_rl/world_worker.py`, `world_pool.py` | Separate simulator processes, ten slots per world, observations and terminal handling. |
| `atc_rl/policy.py`, `buffer.py`, `train.py` | Shared local actor; local PPO or centralized MAPPO critic; live-sample accounting. |
| `atc_rl/train_sac.py`, `sac_support.py` | Fresh SAC comparison and active replay. |
| `atc_rl/evaluate.py`, `competition.py`, `deployment.py`, `audit.py`, `compare.py`, `curves.py` | Evaluation, deployment, provenance, comparisons and curves. |
| `atc_rl/demonstrations.py`, `imitation.py`, `pretrained.py` | Teacher data, supervised actors and verified BC-to-RL initialization. |
| `atc_rl/maneuvers.py`, `policy_components.py` | Categorical actions; diagnostic swaps between fixed policies. |
| `jobs/`, `tests/rl/` | Slurm cohorts and tests of accounting, information boundaries, reload and transport. |
| `runs/`, `docs/competition/`, `output/` | Immutable evidence, historical notes, candidate/media/report artifacts. |

MAPPO's critic receives joint observations, alive flags and target identity. Structural/gradient probes verify that the actor does not use joint context and the centralized critic does. This is implemented MAPPO, not ten-aircraft PPO renamed. Critic/actor capacities differ; comparisons do not isolate all architectural variables.

Route guidance, static filtering and aircraft-conflict filtering are separate support factors. Earlier `filter` experiments used combined static/traffic projection. New `static_filter` leaves aircraft conflicts to the policy. CPA predictive inputs are observations, not a conflict controller.

## Fixed classical benchmark

Public route choice every five seconds, speed +1, route guidance and combined static/traffic filtering, decimal heading transport; frozen with classical-only refinement paused.

Local twenty-world panel, `runs/ppo-direct-pilot-v1/eval-classical-dev20`:

| Arrival | Clean | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 99.5% | 94.5% | 1082.130 | 2.510 | 0.270 | 0 |

CSV SHA-256: `202955dcbe08427b0b537f51f4dcd59593b4a91e99cd6589c68c9d8105e61641`.

The paired cluster baseline reports 95% clean, 1081.735 s flight, 2.510 s intrusion and 0.040 s restricted. This discrepancy needs raw record/source/dependency investigation. Preserve `runs/sac-comparison-development-v1/protected-source.json` and frozen bundles.

## Experiment history

Values below are per-aircraft means; percentages are arrivals/clean completion and times are simulated seconds. New pilots normally use twenty development worlds (200 aircraft). Exact recipes, curves and technical attempts are indexed in `EXPERIMENT_INDEX.md`.

### Historical SAC and controller work

This predates the explicit RL-first change. It established infrastructure and strong supported controllers, but does not establish a strong learning contribution.

| Experiment | Counted-slot budget | Main result |
| --- | ---: | --- |
| Public-reward SAC | 391,520 | 99.5% arrival, 27.5% clean; 24.55 s intrusion, 79.505 s restricted, 50.84 s outside. |
| Continuation | 594,480 total | 98% arrival, 30.5% clean; poor safety. |
| Same weights plus static projection | No extra training | 97.5% arrival, 59.5% clean. |
| Training with projection | 491,520 / 584,360 | 47% / 55.5% clean. |
| CPA versus matched guarded control | About 491,520 | 48% versus 52.5% clean. |
| Original weights plus joint projection | No extra training | 96.5% arrival, 89.5% clean; zero intrusion but restricted exposure. |
| Route-input plus joint support | No extra training | 97.5% arrival, 95.5% clean; mostly support contribution. |
| Larger SAC learner | 347,540 | 98% arrival, 28.5% clean. |
| Route-residual SAC | 25k / 100k | 99% / 98% clean on a small early panel; classical comparison initially had the wrong decision interval. |

Correcting the interval: 200 development worlds gave classical 95.8% clean versus learned 94.9%. On 200 seed-2027 MA worlds, both had 99.35% arrival; classical 93.8% clean versus learned 93.9%, with no established paired improvement. Three residual seeds (2900/2902/2904) gave 93.9%, 93.9%, 94.45% clean against 93.8% classical. Nominal 25k budgets contain roughly 14–15.5k live samples; selected checkpoints had 7,996 optimizer callbacks.

SA transfer on the historical 200-world panel gave arrivals 100%, 98%, 99% versus classical 97%; clean 69%, 68%, 68.5% versus 68%. The primary +3-point arrival contrast had a paired interval about +1 to +5.5 points, not universal across seeds. These are MA-trained policies transferred to SA, not three native SA training successes. Arrival reward rescaling and controller timing are separate factors.

Evidence: `PILOT_RESULTS.md`, `HELDOUT_RESULTS.md`, `REPLICATIONS.md`, `HEADING_TRANSPORT.md`, `NATIVE_POLICY.md` under `docs/competition`, and corresponding SAC/heldout run folders.

### Direct PPO, MAPPO and basic changes

| Recipe | Live transitions | Optimizer steps | Outcome |
| --- | ---: | ---: | --- |
| Direct PPO | 1,000,371 | 10,000 | 5.5% arrivals. |
| Direct centralized MAPPO | 100,744 | 1,000 | 0% arrivals. |
| PPO potential progress shaping | 100,217 | 1,000 | 8% arrivals; worse conflicts. |
| Reward scale 0.01 | 101,160 | 1,000 | 8% arrivals. |
| Shaping plus scale | 104,783 | 1,050 | 18% arrivals. |
| Neutral mean / Gaussian std 0.05 | 101,525 | 897 | Initial 100% arrival prior deteriorated to 4%. |
| Goal-relative action | 103,226 | 1,093 | 100% arrivals retained; clean 19.5% to 18%. |
| Goal-relative scaled PPO | 948,398 | 10,156 | Wall-limited below 1M; 100% arrivals, 23.5% clean; intrusion 57.05 s, restricted 108.46 s. |
| Goal-relative scaled MAPPO | 708,127 | 7,729 | Wall-limited; 100% arrivals, 21.5% clean; exposure worse than initial. |
| Route-guided PPO, conflict filter off | 101,144 | 1,091 | Clean 26% to 23.5%; restricted 0.705 to 11.09 s. |

Matched approximately 700k checkpoints (PPO 700,844; MAPPO 701,204): MAPPO flight +241.995 s, intrusion +24.59 s, restricted +43.345 s; clean only +0.5 points with paired interval [-3.5, +5]. Initial actors matched, central critic verified. This does not prove MAPPO generally inferior; it rejects assuming a central critic fixes this recipe.

Evidence: `RL_PLAN.md`, `PROGRESS_REWARD.md`, `REWARD_SCALE.md`, `INITIALIZATION.md`, `GOAL_OFFSET.md`; `runs/goal-scaled-final-evaluations-v1` and individual training folders.

### Support, input and exploration ablations

- **Guidance × combined filter factorial:** seeds 49900/49920/49940, about 100k live per cell. Initial to final mean clean: neither 19.5% to 18.667%; guidance 26% to 24.333%; combined filter 83% to 84.167%; both 97% to 94%. All failed the registered screen. Most high initial performance is support. `runs/ppo-support-factorial-v1/four-cell-results.json`.
- **Real versus zeroed CPA inputs:** identical-size 170-dimensional inputs and matched initial weights; seeds 50400/50500/50600, six runs totalling 604,447 live. Guidance on, both filters off. Real-feature clean change -0.833 points versus -1.5 masked; feature-arm intrusion +5.54 s, restricted +11.648 s. Neither advances. Does not test this contrast under static support. `runs/ppo-cpa-matched-v1/paired-three-seed-results.json`.
- **Traffic position scale 1 versus 20:** one matched seed with static support, 101,437/100,623 live; 2,019/1,967 updates. Final clean 23%/27.5%; flight 1149.170/1221.330 s; intrusion 128.010/136.660 s. Both fail. `runs/ppo-position-scale-pilot-v1/paired-results.json`.
- **Gaussian std 0.05 versus 0.20:** seed 51200, 101,257/100,557 live; 1,999/2,250 updates. Clean 22.5%/30.5%, arrival 95.5%/93.5%, flight 1236.885/1794.805 s, intrusion 121.92/152.99 s. Clean gain masks other degradation. `runs/ppo-exploration-static-pilot-v1/paired-results.json`.
- **Input sensitivity:** on 371 saved observations, relative positions accounted for about 0.1–0.36% of squared standardized heading gradients. Descriptive, not causal importance or proof attention will work. `runs/traffic-input-sensitivity-v1/results.json`.
- **Actuation persistence:** an untrained two-world probe found greater realized reference drift with sixty-second correlated noise than fresh five-second noise at similar width. It motivated gSDE; it is not learned performance. `runs/exploration-actuation-diagnostic-v2/analysis.json`.

### Completed cluster PPO baseline

Jobs 851486–851489 report completed seeds 50100/50200/50300: **3,024,460 live transitions and 30,430 optimizer steps total**, roughly one million live each. Final mean: 100% arrivals, 19.667% clean, 869.870 s flight, 71.463 s intrusion, 124.250 s restricted. Initial arrival was already 100%, clean 19.5%. Per-seed clean gains: +0.5, -0.5, +0.5 points. Training works; competitive learning is not established.

Evidence: `runs/cluster-receipt-851486-851489/receipt.json`, based on user-returned output, with raw files pending. Reported training took about 317 seconds per seed. This establishes useful cluster throughput, not unlimited scalability across recipes or evaluation workloads.

### gSDE: selected pilot, fresh replication and cluster scaling

Frequency 1 redraws noise each five-second decision; frequency 12 every sixty seconds, also resetting at rollout boundaries. Matrix noise std is not marginal action std. Guidance/static support/CPA are on; traffic-conflict filtering is off.

Selected pilot seed 51700: persistent noise yielded a sixteen-point clean gain to 41.5%, with 100% arrivals, but restricted exposure increased to 1.285 s. Failed full screen; excluded from fresh aggregate.

All fresh local seeds 52200/52300/52400 are complete: **604,230 live samples and 8,372 optimizer steps across six runs**. An interrupted 52300 frequent-noise attempt is retained separately; recovery changed the wall-time allowance. `runs/ppo-gsde-recovery-v1/three-seed-results.json` includes the six completed cells and incomplete attempt.

| Local 100k recipe | Arrival | Clean | Flight (s) | Intrusion (s) | Restricted (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common initial | 99.5% | 25.5% | 1090.180 | 113.340 | 0.380 |
| Frequency 1 final | 95.833% | 31.167% | 1250.395 | 128.557 | 0.490 |
| Frequency 12 final | 98.5% | 27.5% | 1143.132 | 125.920 | 1.045 |

Frequency-1 clean gains: +16, +2, -1 points; frequency-12: +8.5, -1.5, -1. Neither mean nor any individual fresh run passes all checks. The selected persistent-noise signal did not replicate consistently.

The cluster study completed too: jobs 852634/852635 training, 852636/852637 evaluation, 852638 summary. Seeds 52600/52700/52800 × both schedules: **6,041,129 live samples, 39,792 optimizer steps**, roughly one million live per run.

| Cluster 1M recipe | Arrival | Clean | Flight (s) | Intrusion (s) | Restricted (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common initial | 99.5% | 25.5% | 1090.180 | 113.340 | 0.380 |
| Frequency 1 final | 98.5% | 25.833% | 1089.832 | 119.533 | 0.305 |
| Frequency 12 final | 99.0% | 27.0% | 1048.092 | 117.043 | 1.658 |

Outside time is zero in these aggregates. Mean clean gains only +0.333/+1.5 points; persistent noise worsens restricted exposure in every seed. Both fail. **Do not scale unchanged gSDE further.** Local and cluster studies change workers/budgets/seeds, so are not a pure continuation learning curve.

Receipt: `runs/cluster-receipt-852634-852638/receipt.json`. Six train/eval completion markers and 143-file verification reported; raw artifacts pending. Bundle SHA-256: `8a9a4496f94323cf896ca0b51c6d27fd1cb4a738b4ed484005ac2f20121818bd`. `GSDE_CLUSTER.md` contains download commands. Do not resubmit completed jobs.

### Fresh SAC and credit assignment

Fresh SAC versus PPO used the same support/observation recipe around 100k live (seed 51800), without historical replay. Lambda 0.95/0.99 is a within-PPO contrast; SAC also differs in architecture/optimization.

| Recipe | Arrival | Clean | Flight (s) | Intrusion (s) | Restricted (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAC | 89% | 41.5% | 1775.380 | 95.650 | 3.930 |
| PPO lambda 0.95 | 92.5% | 23% | 1463.730 | 191.600 | 0.080 |
| PPO lambda 0.99 | 98% | 26% | 1182.395 | 128.440 | 0.830 |

All fail. SAC's clean rate does not automatically offset missed arrivals and delay. Evidence: `runs/sac-ppo-matched-pilot-v1`, `sac-ppo-credit-review-v1`, `ppo-credit-pilot-v1`.

Cluster technical check 852491: reported 21,493 live samples, 2,576 updates, about 41.9 s training on H200 NVL, CUDA reload and native parity. Integration only, not a performance study. `runs/cluster-receipt-852491`; raw files pending.

### Continuous behavioral cloning followed by PPO

Classical teacher labels; student uses local observations, route/static support and **no traffic-conflict filter**. Data: **62,744 unique training transitions**, 16,108 validation; forty BC epochs, **2,480 supervised updates, 2,509,760 repeated examples**. These are not RL samples.

Both PPO rates share the supervised parent and seed 61420. Interrupted attempts and checkpoint identities are preserved.

| Stage | Additional live RL | RL updates | Arrival | Clean | Flight (s) | Intrusion (s) | Restricted (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BC initial | 0 | 0 | 99% | 34.5% | 1088.405 | 99.400 | 1.215 |
| PPO lr 3e-4 final | 101,628 | 268 | 93% | 57% | 1564.100 | 80.180 | 2.870 |
| PPO lr 3e-5 final | 101,083 | 2,350 | 98% | 33% | 1070.620 | 121.200 | 0.845 |

Default clean gain +22.5 points (paired-world interval about +8 to +36), but arrival -6 points, flight about +44%, restricted exposure higher. A real **conditional learning signal**, not an overall strong solution. Lower rate allowed all planned updates but did not solve performance. Default applied 268/2,340 possible updates due to early KL stopping; lower rate applied 2,350/2,350.

Evidence: `runs/imitation-ppo-lr-recovery-v1/paired-results.json`. Corrected supervised-initial/extra-RL curve labels: `runs/imitation-ppo-control-review-v1/curve`; original artifacts remain.

Fixed-policy component swap, same twenty worlds:

| Heading / speed | Arrival | Clean | Flight (s) | Intrusion (s) | Restricted (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both initial | 99% | 34.5% | 1088.405 | 99.400 | 1.215 |
| Both trained | 93% | 57% | 1564.100 | 80.180 | 2.870 |
| Trained heading / initial speed | 94.5% | 53.5% | 1564.130 | 94.660 | 3.280 |
| Initial heading / trained speed | 99.5% | 35% | 1082.285 | 103.510 | 1.175 |

Heading accounts for most gain **and delay**; speed is not the main cause. This outcome-driven diagnostic is not independent candidate selection. `runs/imitation-ppo-component-swap-v1/results.json`.

### Categorical BC/PPO — latest completed pilot

Twenty headings (observed route bearing or a five-degree grid) × three speed increments. Initial logits prefer route bearing and speed +1 with probability 0.9 each: an explicit navigation prior. All 78,852 train/validation commands are representable.

Technical PPO/MAPPO runs each used 4,228 live samples and 74 updates, with matching initial actors and native parity. A categorical SA deployment check is technical only. The full categorical pilot used BC seed 62300, forty epochs and the same supervised budget; PPO seed 62320 then received **100,734 live, 83,586 padding, 184,320 counted transitions, 2,234 optimizer steps**.

| Stage | Arrival | Clean | Flight (s) | Intrusion (s) | Restricted (s) | Outside (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untrained navigation prior | 100% | 25.5% | 1006.285 | 104.450 | 0.355 | 0 |
| BC initial | 86% | 25% | 1269.145 | 197.120 | 1.295 | 0 |
| First checkpoint at/above 50k | 66.5% | 23.5% | 1766.080 | 250.910 | 1.015 | 0 |
| Final | 74.5% | 28.5% | 1510.075 | 70.400 | 5.595 | 1.280 |
| Classical | 99.5% | 94.5% | 1082.130 | 2.510 | 0.270 | 0 |

Completed 2026-09-17 11:11:48 UTC. Failed every screening check except intrusion time. BC damaged arrivals; PPO's conflict-time reduction accompanies misses and restricted/outside exposure. **Do not scale this recipe unchanged.**

Offline overall heading error improved (about 2.32 degrees categorical versus 4.65 continuous), but intervention-state error worsened (11.93 versus 6.85). Lower average teacher-state loss did not establish robust closed-loop behavior. Evidence: `runs/categorical-imitation-error-diagnostic-v1/diagnostic.json`; `runs/categorical-imitation-ppo-pilot-v1/{complete.json,comparison,curve}`.

Final model SHA-256: `2a513e9f4f5274ef3d219ba8c69fb7bcb32c6fddc53840d6c78733f872a400a8`. Source SHA-256: `d72575b87b1e63f5cdd8fb0c77fe7db5d31017f7fbcb72f27dc2c13971275903`. All 180 final native/vector checked metrics match: transport validation, not quality.

### Latest implementation without a performance result

`atc_rl/pretrained.py` supports verified supervised **actor-only** transfer into MAPPO, retaining a fresh central critic/optimizer. PPO retains supervised policy tensors. Metadata records scope, supervised counts and critic identity; guards reject setting/tensor mismatches and supervised policies mislabeled as untrained references.

Focused tests: 30 passed. Latest full RL suite: **309 passed in 48.95 seconds**. Twelve new cases are in `tests/rl/test_pretrained_mappo.py`. These are not simulator integration results.

A 4k-live MAPPO BC-transfer protocol exists at `runs/pretrained-mappo-development-v1/protocol.json`; **no integration runner or experiment has been executed**. No BC-to-MAPPO performance claim is supported. Frozen 159-file bundle SHA-256: `992e59457c8ba2747487511cc7321603706f21ecdb9dbfeeddbd91d2e3df1fe5`.

### Other completed diagnostics and deferred attempts

- Goal-region versus point-goal routing: classical safety/arrival unchanged; one aircraft's flight shortened 126 s, only 0.630 s on average (paired interval [-1.890, 0]). Learned-policy transfer gave both arms 100% arrival/98% clean, with flight 937.645 versus 937.680 s. No meaningful gain or further RL training; refinement paused. `docs/competition/GOAL_REGION.md`, `runs/goal-region-v1`.
- Intervention-penalty reward experiment: fixed-policy checks completed; first coordinator failed on source provenance. At priority change, the retry coordinator stopped, while its already-running unshaped child finished 25,000 counted / 13,875 live / 11,125 padding samples. Shaped training and queued evaluation are deferred, so there is no learned penalty comparison. `INTERVENTION_REWARD.md`, `runs/intervention-reward-v1`.
- Batched forecast kernel: exact kernel and native metric parity checks passed; interleaved timing pairs disagreed (active-time ratios 1.541 and 0.914), so no reliable end-to-end speedup or promotion is claimed. Keep the failed initial replay too. `COMPUTE.md`, `runs/forecast-batch-v1`.
- Tail/slot analysis: descriptive quantiles, misses and conditional conflict severity are preserved, without treating aircraft IDs as demographic groups or claiming fairness. Means include timeouts. `ROBUSTNESS.md`, `runs/development-tails-v1`.
- Local Linux/WSL setup exhausted disk/memory headroom and was stopped before native Linux validation. Installation attempts, metadata and failure records are preserved; dependency payloads are excluded from publication. Do not repeat that installation beside evaluations; the working university environment supersedes the need for it.
## Failures and incomplete work to retain

- Bootstrap **851227**: dependencies/CUDA passed, eleven tests passed; two reload tests failed because auto-device restoration selected CUDA while inputs stayed CPU. Fixed with `AircraftPolicy.load(path, device=model.device)`. Corrected bootstrap **851402** succeeded. Reuse the environment; do not reinstall because the old job failed.
- Integer-form heading commands could be misinterpreted by the parser. Decimal transport is a recorded revision; old source/results remain. Do not relabel old-format results.
- Per-aircraft deterministic native inference is authoritative. Batched floating-point inference can eventually produce different trajectories; transport tests must mirror deployment.
- BC-to-PPO reload needed compatibility for equivalent normalized float32/float64 Box storage. Fixes preserve identity and reject different bounds/categorical spaces; see `pretrained-space-fix-v1` and recovery evidence.
- Interrupted runs retain partial counters/logs. Recovery uses verified checkpoints and records duplicated work, not fictitious independent training.
- The static-support cluster study is prepared, but **no completion receipt is present locally**. Readiness is not completion; check existing cluster jobs before resubmitting.
- Corrected official SA scoring completed: 1,000 seed-42 scenarios, 98.5% arrival, 70.4% clean, 1175.885 s flight, 17.386 s intrusion, 1.040 s restricted, 0.087 s outside. Historical SAC/controller candidate, not latest PPO. `runs/official-decimal-v1/sa/summary.json`; `judge_verified` false. No paired 1,000-world classical control establishes the learning effect.
- Corrected official **MA** has a protocol, log and stale `running` marker (PID 23900), but **no final CSV or summary**. PID absent at publication. Treat as incomplete/unverified; do not restart or overwrite automatically.
- No relevant local Python experiment processes were present in the final inspection. Categorical, component-swap and local gSDE runs finished normally. Remote jobs require cluster access/receipts to confirm.

## University cluster

Interactive SSH through the jump host:

```bash
ssh -J shri@sjump.comp.nus.edu.sg shri@xlogin.comp.nus.edu.sg
```

The owner enters passwords in SSH prompts; do not request/store credentials. Project root `$HOME/cs4246-rl` (observed `/home/s/shri/cs4246-rl`). Existing interpreter:

```bash
export ATC_PYTHON="$HOME/cs4246-rl/onpolicy-v1/.venv-cluster/bin/python"
"$ATC_PYTHON" --version
```

Login Python 3.12.3; no `module` command. Downloads have worked. Account `allusers`, QoS `normal`. Observed `gpu` limit three hours, supports `--gres=gpu:h200-141:1` and named A100/H100 variants. `gpu-long` limit three days, but H200 was not listed. Recheck allocations when changing GPU/partition.

Established jobs request **one GPU, sixteen CPUs, 64 GB, eight simulator worlds**. Eight CPUs serve simulators with headroom for learner/coordination. Set OMP/MKL/OpenBLAS threads to one. Small policy networks do not automatically benefit from more GPUs; simulation throughput matters. Native evaluation is CPU-heavy, normally two CPUs/eight GB on `normal`. Do not run heavy simulation on login nodes.

Bootstrap, CUDA, reload and training passed. Operational references: `CLUSTER_RL.md`, `BASELINE_RESULTS_DOWNLOAD.md`, `GSDE_CLUSTER.md`, `SAC_CLUSTER_CHECK.md` in `docs/competition`, plus `jobs/CLUSTER.md`. Download completed baseline/gSDE/SAC evidence before relaunching. Do not interfere with unrelated `persona-sft` jobs (historically 852435/852436/852437).

Remaining access dependency: interactive authentication for downloads and genuinely new submissions. Allocation details are known; do not ask again for generic GPU/account/Slurm information. Exact remote status of the prepared static-support study is unknown.

## Public-fork research

`docs/competition/PUBLIC_WORK.md` and dated research records preserve the survey: 38 public forks, about five clearly developed efforts. Forks are not verified submissions; private repositories/email/Discord entries remain unknown. No verified public ranking supports a claim that we lead.

Inspected CGCooke commit `a458870c711679e4fedb764abc1d58c8f4ffdb61` contains a Rust training simulator with author-claimed parity, not independently established here. Committed E18 MA results from 10,000 aircraft records report about 99.6% arrivals, 983.641 s flight, 10.578 s intrusion, 40.622 s restricted, 12.443 s outside. E27's roughly 99.93% arrival was self-reported; its referenced CSV was not independently inspected. Their scalar is not an official competition formula. Another heading-only SAC SA result is not comparable to MA. These are research leads, not a reproduced leaderboard.

## Recommended next experiment and sprint

Do **not** launch more unchanged PPO/gSDE scaling or assume a central critic fixes a weak actor/objective. The clearest problem is that learned heading interventions trade successful, efficient navigation for selected clean/conflict improvements. Cloning also struggles on consequential intervention states.

1. **Consolidate evidence:** download completed cluster archives; verify hashes, raw CSVs, curves and source manifests. Investigate classical Linux/Windows differences. No new training needed.
2. **Diagnose continuous BC/PPO:** use completed heading/speed swaps and per-world traces to distinguish long detours, failure to rejoin, missed arrivals and reduced conflict exposure. Stratify teacher interventions/failures. Do not open reserved streams.
3. **One pilot hypothesis:** an auxiliary imitation loss during PPO may retain route-following while learning conflict maneuvers. Compare existing continuous BC+PPO against BC-initialized PPO with a modest, predeclared demonstration-retention schedule. Keep observations, actions, support, reward, simulator, seed, live budget and evaluation fixed; count extra supervised minibatches separately. This is proposed, **not implemented or demonstrated**. Do not simultaneously combine attention, curriculum, reward and action-space changes.
4. **Controls:** untrained navigation prior, BC-only, compatible PPO-only, existing BC+PPO, new regularized arm and fixed classical. Record full native metric vectors, fixed-budget curves, interventions and failure traces. Keep traffic-conflict filtering off initially to expose learning; evaluate guidance/static/traffic support separately later. Predeclare acceptable arrival/safety/efficiency tradeoffs; preserve old studies' original screens.
5. **Replicate only a promising pilot:** three cluster training seeds with adequate CPU workers. Compare fresh SAC and genuine MAPPO empirically at comparable live budgets; report sample/update/compute differences. Validate BC-to-MAPPO simulator transfer before a full study. Do not assume MAPPO wins.
6. **Lock before unseen evaluation:** reserved 20301/20302 streams and larger scenario panels, paired uncertainty and adverse tails. Avoid tuning against heldout results. Then run the exact native competition protocol on a frozen candidate.
7. **Decide with the owner:** useful infrastructure and honest negative results exist, but strong competitive learning is not demonstrated. Decide if measured effect, novelty and generalization justify this as the course project. No competition/course submission has been performed.

Suggested remaining allocation: one day evidence/diagnosis, two days one controlled implementation/pilot, two days replication/comparison if warranted, one day locked evaluation and continuation decision. This is a proposed schedule, not a promise of winning performance. Verify the actual remaining calendar; the earlier “ten days” message is not a current deadline.

Alternative hypotheses if needed: on-policy demonstration aggregation for covariate shift; explicit arrival/safety objectives with fixed native evaluation; conflict-focused training curricula returning to the original distribution; permutation-aware traffic encoding. These are future ideas, not implemented results.

## Course requirements

Competition-project track and multi-agent topic fit the intent, conditional on the actual deliverable. Recorded milestones from the provided guideline: one-page proposal (10pt, one-inch margins), 28 September 18:00; presentation around the week of 10 November (15 minutes plus 5 questions); report around mid-November, up to ten pages (12pt, one-inch margins), Vancouver references. The draft guideline had a year inconsistency; verify final date in Canvas. Competition 30 November does not supersede course deadlines.

Report actual learning, support attribution, seeds, unseen results, limitations and real member/tool contributions. Do not fabricate novelty, rankings, roles or independent execution. Old report drafts preserve earlier controller work and would need rewriting around eventual RL evidence; polishing remains paused.

## Starting point for the next developer

Read this handoff, `EXPERIMENT_INDEX.md`, latest continuous/categorical BC/PPO comparisons, local gSDE aggregate and cluster receipts. Restore required archives. Use a new experiment directory, frozen source and protocol. Never edit old runs, benchmark or scoring in place.

Latest full RL test command:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.venv\Scripts\python.exe -m pytest tests/rl -q -p no:cacheprovider
```

Use `python -m atc_rl.train --help`, `python -m atc_rl.evaluate --help` and an immutable experiment's runner/protocol to prepare a new run. These references are not instructions to resubmit old cohorts. Dependencies: `jobs/requirements-onpolicy.txt`; reuse cluster environment. Historical scripts embed absolute machine paths; adapt copies, not archived protocols. Never load historical replay pickles during inspection.
