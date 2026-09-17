# Air traffic control experiments

All evaluations use the original scenario generator, flight dynamics, metric sampling,
and scoring methods. Training hooks may change rewards; evaluation uses the same
observation layout as training. The neutral policy issues zero heading/speed changes.

## Local setup

Create a Python 3.12 environment and install the repository with `pip install -e .`.
Simulator runtime files are stored in runs/simulator. The existing `uv.lock` is the upstream dependency reference. Every training run saves
its actual installed versions and configuration. Do not assume runs with different
simulator or library versions are identical.

Run these commands from the repository root:

```sh
python -m atc.evaluate --env sa --episodes 20 --out runs/neutral-sa
python -m atc.evaluate --env ma --episodes 20 --out runs/neutral-ma
python -m atc.train --env ma --algorithm ppo --workers 1 --steps 250000 --run-dir runs/ppo-baseline
python -m atc.train --env ma --algorithm sac --workers 1 --steps 250000 --run-dir runs/sac-baseline
python -m atc.evaluate --env ma --algorithm sac --model runs/sac-baseline/model.zip --episodes 100 --out runs/sac-baseline/validation
```

The baseline reward is unchanged. `--recipe balanced` is an experimental comparison:
reach bonus 25 and discount 0.997; it is not a proven improvement. Use the same recipe
flag when evaluating so the reference total_reward remains interpretable.

`--recipe goal_relative` uses the balanced reward and discount but interprets the
heading output as an offset of up to 90 degrees from the current goal bearing.
The resulting command is still clipped to the original 45-degree turn limit;
speed control, dynamics, scenarios, observations and scoring remain unchanged.
This is an action-parameterization experiment, not a proven safety improvement.
The same recipe is required at inference.

Select checkpoints by completion, safety, and flight-time metrics. Training reward is
not a competition score. Use seed 2026 for development comparisons;
reserve seed 42 for final evaluation. Complete all development runs before using the
1000-episode official sequence. The custom evaluator records scenario and aircraft IDs
so confidence intervals can be clustered by scenario. It adds clean-completion metrics
without modifying any official metric.

The two allowed hooks in scripts/evaluate_competition.py now load the supplied model
and its saved wrapper configuration through atc.submission. The rest of that harness
is unchanged. Its integration is checked separately on development scenarios; a full
1000-episode official run remains required after candidate selection. The custom
evaluator's --official switch does not substitute for that submission requirement.

## Parallel training

Single-agent worlds use separate spawned processes. Multi-agent training follows the
upstream shared-policy design with a fixed population wrapper and one process per world when
workers > 1. Ten agent views of one world are not ten independent simulations.
Completed aircraft enter absorbing zero-reward slots until the world ends. The wrapper
preserves termination versus truncation and resets when the last aircraft leaves; reported
SB3 training steps include those padded transitions. SAC replay keeps only live aircraft
transitions and stops bootstrapping on arrival; PPO currently retains absorbing padding. Objective evaluation runs directly
on the environment and captures each aircraft's final metrics when it leaves.

## University Slurm cluster

Log in interactively using the provided jump host. Never place passwords in scripts.
From a cluster checkout with dependencies installed, set ATC_PYTHON to its Python
executable, then submit jobs/train_baseline.slurm. Pass the cluster's actual partition,
account and GPU allocation flags to sbatch as required; these are not yet known.
An array --array=0-2 runs three seeds. ATC_ALGORITHM selects ppo or sac (default sac).
The job template is prepared locally; cluster execution still needs verification.

## Paired policy comparison

Use `python -m atc.compare runs/neutral-ma runs/sac-baseline/validation --out runs/comparison.json`
when both evaluations have identical environment, scenario count and seed. The report
uses a paired bootstrap over whole scenarios, since ten aircraft in one airspace are
correlated. Intervals are pointwise and do not measure variability across training seeds.
Do not treat repeated development checkpoint selection as an independent final test.

## Resuming and job limits

`--resume path/model.zip` continues model/optimizer state with a fresh scenario seed.
For SAC, also supply `--resume-replay path/replay.pkl` when available. Without replay,
the trainer collects a new warmup before making updates. `--steps` counts additional
transitions when resuming. This is not an exact continuation of simulator or random
number generator state. Keep the original configuration alongside the checkpoint.

`--save-replay` saves SAC experience with every checkpoint and at completion. A full
million-transition buffer with float32 replay observations is about 1 GB, so allow
space for multiple files. `--max-wall-seconds` stops learning and saves before the
Slurm deadline; a termination signal also requests a clean save. A forced kill or
node failure can still interrupt saving. The job template reserves time for saving
and leaves evaluation to a separate job. Its 3-hour allocation and GPU resource flags
must be checked against the selected partition.

Array seeds start at 1000, 1100, 1200, so eight-worker streams do not overlap.
Worker seed ranges must avoid 42, 2026 and 2027. Use 2026 for development and leave
2027 untouched until the finalist check. Match the original recipe when resuming.

## Public reward reference

`--recipe public_weights` selects the reward coefficients and discount reported in
CGCooke's public champion configuration. The original direct-turn and speed mapping
is retained. This is a reference baseline using our collector, not a reproduction of
that fork's Rust simulator or its reported scores. See PUBLIC_WORK.md for attribution
and limitations. Test replay budgets separately: with W multi-agent workers,
`--gradient-steps 4*W` corresponds to roughly 0.4 updates per SB3 vector transition.

## Route-guided experiment

`--recipe route_guided` computes polygon visibility paths with a 6 km obstacle
clearance (falling back to 3, 1, then 0 km only if disconnected). The actor receives
three route-reference features and chooses a heading offset up to 45 degrees around
the visible route target plus the original speed action. Scenarios, flight dynamics,
turn-command limit, and official scoring are unchanged. Geometric clearance is not
a flight-dynamics safety guarantee; learned deviations can leave the planned path.

Without a model, this recipe evaluates the classical route-following reference. That
reference cannot establish RL performance. Compare learned policies against both it
and the straight-flight neutral policy. Use `atc.diagnose` to replay and trace a chosen
multi-agent development scenario; it replays the complete prefix before recording.

`--recipe route_guided_inset` additionally keeps the visibility path 6 km inside the
sector, reducing that margin with the obstacle-clearance fallback when needed.
For either route recipe, a newly trained SAC actor starts at zero mean deviation,
log standard deviation -2, and warmup action noise standard deviation 0.15. This
preserves a known navigation reference at initialization without freezing the learned
policy's control authority. PPO does not use this initialization. Resume runs retain
the saved actor. These settings and their source are captured per run.

## Static-clearance filter experiment

For route recipes, `--guard-static` checks candidate turns against a 45-second
bounded-turn projection, using a 2 km static-obstacle/sector margin. It chooses the
closest projected-feasible offset to the learned command, or returns to the navigation
reference if no candidate is feasible. This approximate predictor is not BlueSky's
flight model and provides no formal safety guarantee. Actual violations are still
measured by the original scorer. All commands retain the original turn and speed limits.

Evaluation can explicitly toggle the filter to measure its effect on the same learned
checkpoint. Otherwise it uses the saved training setting. Training with a changed filter
requires fresh replay, since the meaning of an action changes; model weights can be
continued, but old replay from the other mapping is rejected. Guarded runs must still
beat the matching classical reference on separation and task completion.

## Cluster source transfer and smoke job

Follow jobs/CLUSTER.md for the interactive SSH/SCP handoff. `python -m atc.transfer
pack --out runs/cluster/UNUSED_NAME.zip` packages the current source and documentation,
including uncommitted changes. Extract over the recorded base checkout, then run
`python -m atc.transfer verify` before installing or submitting. The source manifest
is separate from model/replay artifacts. A GPU smoke job exercises both required SA
and stretch MA training/evaluation before any large Slurm arrays are launched.

Route revision 2 retains waypoint progress after evasive deviations. Old route replay
from revision 1 must not be used with this changed mapping; weights-only continuation
with fresh replay is allowed. Evaluation records both training and current route
revisions. The fix did not improve completion in the first failing scenario, so it is
not evidence of a better policy.


On Windows, if the system pytest temporary directory is inaccessible, run tests with
`python -m pytest -q --basetemp runs/pytest-temp`. This directory is reserved for
pytest temporary files and may be cleared by pytest; do not store results there.


## Predictive traffic observations

`--recipe public_cpa` is an observation-only ablation of public_weights. It appends
five features in each existing distance-sorted traffic slot: presence, time to the
closest point within 180 seconds, distance at that point relative to the 5 NM minimum
(capped at four), predicted separation-entry time, and a predicted-conflict flag.
It derives these from current horizontal relative position/velocity. Missing slots
are explicitly masked, including the case of two aircraft at the same position.

The constant-velocity prediction is an approximation. It does not inspect future
scenario state or change commands, rewards, scenario generation, or scoring. A full
20-scenario neutral MA replay matched all eight objective metrics exactly for every
aircraft. Both SA and two-worker MA SAC collection/update/save smoke tests pass.
No trained performance improvement is established by those checks.

## Inference and original-harness checks

Learned MA evaluation now predicts one aircraft at a time, matching the original
harness. Earlier development results used batched inference; small numerical action
differences produced different rollouts for the same checkpoint. Use --batch-inference
with atc.evaluate or atc.diagnose only to reproduce those historical runs. New result
metadata records inference_mode. Final checkpoint selection must use per-aircraft
inference consistently. Single-agent evaluation is unchanged.

`python -m atc.check_harness --reference EVALUATION_PREFIX --model MODEL.zip
--episodes 2 --out runs/harness-check.json` compares the original rollout loops with
an existing development evaluation. It uses seed 2026 in that development process;
the source harness retains seed 42 and the 1000-episode reporting protocol. The check
writes differences and original-harness records for diagnosis. It is not an official
result and does not consume the reserved test stream.

After freezing a compatible candidate, the original command accepts its model path:
`python -m scripts.evaluate_competition --env ma --model MODEL.zip --out RESULTS.csv`.
Retain config.json next to the checkpoint. The adapter rejects a mismatched track,
unknown recipe, stale route mapping or incompatible observation/action space. Run
SA and MA separately with their respective checkpoints. No official run has been
performed yet.


## Warm-starting conflict features

`python -m atc.extend_observations --model SOURCE/model.zip --out-dir NEW_DIRECTORY`
adds CPA inputs to a plain goal_relative or public_weights SAC checkpoint. The
initially zero PredictionResidual preserves the original policy's input and weights.
Add --freeze-predictions for the matched control. Both variants reset optimizers and
start with empty replay; do not supply the original lower-dimensional replay when
resuming. Historical model.num_timesteps is retained. The conversion configuration
records the source hash and explicitly reports zero training after extension.

Resume either branch with atc.train, the target recipe from its config.json, a fresh
training seed and identical budgets. --steps counts additional transitions. The
trainer first collects fresh warmup experience, then updates the model. The saved
configuration records whether the prediction projection is trainable. Use actual
completed counts to compare branches. Conversion and parity checks alone do not
establish a performance gain.

## Multi-agent to single-agent transfer

`python -m atc.transfer_track --model MA_SOURCE/model.zip --out-dir NEW_DIRECTORY`
copies a public_weights or public_cpa SAC checkpoint unchanged into an SA evaluation
configuration. It retains all ten scripted world intruders but observes the nearest
nine, matching the source MA input size. The converter checks actual traffic count,
scenario routes, spaces and a finite action after reset. No SA training has occurred.

Evaluate with --env sa and the public_sa_transfer or public_cpa_sa_transfer recipe
recorded in the output config.json. This tests generalization to scripted intruders;
it is not an SA-trained baseline or evidence that nine slots are optimal. Keep the
source checkpoint and the transfer configuration together for the original harness.


## Static projection for direct-control policies

For direct-turn recipes such as public_weights, --guard-static now checks the learned
heading command against a 90-second bounded-turn prediction, with a 2 km sector and
polygon-obstacle margin. It preserves the command exactly when that prediction is
clear. Otherwise revision 4 chooses the closest feasible turn to the learned command
within the original +/-45 limit, using goal bearing as a tie-breaker. If no candidate is feasible, it minimizes predicted violation and distance
from the permitted domain; this is recorded as an infeasible case, not a guarantee.
The original learned speed command, observations, reward and scorer remain in use.
Revision 2 stops predicted paths at the first goal-circle entry, matching aircraft
removal on arrival. Revision 1 incorrectly assessed hypothetical travel beyond the
goal. Old revision-1 filter results remain historical evidence; do not reuse replay
across this mapping change. Trajectory diagnostics now record the actual selected
turn and filter intervention alongside the actor's proposed action. Revision 3
also excludes a tangent touch or outward departure at exactly the capture radius
from predicted arrival, matching the scorer's strict-inside condition.

Revision 2 minimized deviation from the actor during intervention. A failed-flight
trace showed this could produce a long sector-boundary detour. Revision 3 tests
progress toward the goal when an intervention is already needed. Clear learned
commands remain unchanged. This requires a fresh performance comparison; it is not
evidence of a better policy by itself.

This is an approximate action filter. It assumes constant speed and a 1.5 degree/s
turn rate, replans at the existing decision times, and does not predict other aircraft.
Measure arrival, all objective safety metrics and intervention counts before deciding
whether it helps. Evaluation JSON records the revision, horizon, margin, action count,
interventions and infeasible cases. Goal-relative controls do not support this filter;
route-relative recipes retain their separate earlier filter.

Evaluate the same checkpoint with and without --guard-static. Use --controller goal
without a model for a classical goal-tracking comparison: it commands the shortest
bounded turn toward the goal and leaves the speed action at zero. Combining that
controller with the same filter helps distinguish learned traffic avoidance from the
filter's navigation benefit. Classical controls are diagnostic baselines, not RL entries.

Changing filter settings requires fresh replay. A submission variant must retain the
checkpoint plus a config.json that records the enabled filter and exact parameters;
the original-harness adapter rejects mismatches. Checkpoint conversions preserve a
validated direct filter configuration. An evaluation-time flag does not edit the
source checkpoint's training configuration.


The revision-3 goal-priority screen repaired one failed arrival but introduced two
others (96% completion over five scenarios). Revision 4 restores the conservative
nearest-command choice and retains the corrected goal-entry prediction. Training
with this filter is the next experiment; post-hoc application alone has not preserved
sufficient completion. Earlier variants remain in their recorded source snapshots.


## Configurable SAC learner

New models accept --net-arch 256x256, --learning-rate and --tau. Omitting them retains
our initial 64x64, 0.0003 and 0.005 defaults. Existing --batch-size, --learning-starts,
--gradient-steps and --buffer-size control the remaining SAC training settings.
Resumes retain the checkpoint architecture, learning rate and tau; omit those three
new-model options when using --resume. The run configuration records actual loaded
values, including the actual replay capacity, so requested and saved settings are
not confused.

The E26 trial-13 learner settings recovered from the public source are documented in
docs/competition/PUBLIC_WORK.md. jobs/CLUSTER.md gives a matching learner experiment
using our synchronous BlueSky collector. Matching learner parameters alone does not
reproduce the public E27 simulator, training schedule or reported performance.


## Joint traffic and static action correction

For direct-control recipes, --guard-traffic enables revision 1 of a joint correction
and includes static filtering. The learned policy still proposes heading and speed.
Candidate commands use a 90-second bounded-turn/acceleration forecast, with 5-second
segments, a 2 km static margin and a 1 km traffic margin beyond the scored 5 NM limit.
The traffic check considers closest approach and overlap within each entire segment.
Predicted goal capture removes an aircraft from later traffic checks.

Aircraft are processed in the environment's stable action order. Each decision starts
with constant-heading forecasts of current traffic; subsequent aircraft consider
already selected maneuvers for earlier aircraft. Among jointly feasible candidates,
the correction selects a nearby learned command. If no jointly feasible candidate
exists, it prioritizes static feasibility and minimizes predicted traffic exposure.
An empty feasible set is recorded explicitly. Only horizontal commands are sent to
the original simulator. These approximations do not guarantee real-world or simulated
separation; the unchanged one-second scoring remains authoritative.

```powershell
.venv\Scripts\python.exe -m atc.evaluate --env ma --algorithm sac --recipe public_weights --model runs/public-joint-v1-391k/model.zip --episodes 20 --seed 2026 --out runs/joint-policy-evaluation
.venv\Scripts\python.exe -m atc.evaluate --env ma --recipe public_weights --controller goal --guard-traffic --episodes 20 --seed 2026 --out runs/joint-classical-evaluation
```

The saved public-joint-v1-391k configuration applies the correction to the unchanged
391,520-transition policy; no additional training was performed for that variant.
The MA-to-SA conversion preserves the filter configuration and still simulates all
ten scripted intruders. Feature-extension conversion also preserves these settings.
The completed twenty-scenario learned MA pilot has no intrusions or sector exits,
89.5% clean completion and 96.5% arrival. The required SA transfer reaches 85% arrival.
Navigation failures prevent overall selection; detailed comparisons are in the
results report.

For a separate training experiment, pass --guard-traffic to atc.train or set
ATC_GUARD_TRAFFIC=1 for jobs/train_baseline.slurm. Resume inherits the saved settings;
changing the traffic filter or its revision requires fresh replay. Saved configurations
record the complete prediction settings, and the original-harness loader rejects
mismatches. Route-relative and goal-relative action recipes are not supported by this
correction. See docs/competition/PILOT_RESULTS.md for measured outcomes.


## Route-bearing input with direct controls

The experimental public_route_input recipe replaces only cos_drift and sin_drift
with the bearing to the furthest visible route waypoint when the final goal is
blocked. The original final-goal distance, all other observation fields, vector
layout and direct heading/speed mapping are preserved. If the final goal is visible,
its original observation bearing is preserved exactly. This differs from the older
route-relative action recipes.

The polygonal planner tries obstacle clearances of 6, 3, 1 and 0 km, with matching
sector inset up to 6 km. When no remaining waypoint is visible after a deviation,
it attempts a new route from the aircraft's current position. A failed replan is
recorded; geometric routing does not guarantee feasible flight dynamics. The
separate joint action correction still uses its existing prediction and margins.

public_route_input_sa_transfer keeps nine observed traffic slots for the MA-trained
policy while simulating all ten scripted intruders. Saved route-input metadata is
required by the submission loader and preserved by MA-to-SA conversion. Changes to
that definition require fresh replay. Unit tests cover unchanged fields, direct
action dispatch, route recovery, unreachable paths, reset state and metadata.
Simulator checks cover both tracks and preserve the 124-element policy input.

```powershell
.venv\Scripts\python.exe -m atc.evaluate --env ma --algorithm sac --recipe public_route_input --model runs/public-route-input-v1-joint-391k/model.zip --episodes 20 --out runs/route-input-policy-evaluation
.venv\Scripts\python.exe -m atc.evaluate --env ma --controller goal --recipe public_route_input --guard-traffic --episodes 20 --out runs/route-input-classical-evaluation
```

The initial deployment variant reuses the original 391,520-transition weights with
no route-input training. Performance must be established separately; changing an
observation supplied to an existing model can reduce its effectiveness.


The completed 20-scenario route-input comparison records 97.5% MA arrival and
95.5% clean completion for the unchanged learned weights, versus 100% for both
metrics with the matching classical controller. The learned SA transfer reaches
90% arrival; the classical SA controller reaches 100%. Both learned integrations
exactly match all nine original-harness metrics on two-scenario checks. The
classical controller is a diagnostic foundation and does not qualify as an RL
entry. Full metrics and paired intervals are in docs/competition/PILOT_RESULTS.md.

## Bounded route-residual learning

public_route_residual retains the route input and adds learned adjustments to the
classical follower's command. Its heading command is the clipped classical turn
plus an adjustment of at most 15 degrees, clipped again to the original 45-degree
turn range. Its speed output uses the original normalized speed command. Joint
traffic/static correction then checks this direct command and may change it further.
The 15-degree bound applies to the proposed learned adjustment, not to that later
safety correction or to the aircraft's realized trajectory.

The reference turn comes from the last observation sent to the policy. Zero output
therefore preserves the classical controller's numeric command exactly. All original
observations and rewards remain as in public_route_input. Native MA has 124 inputs;
native SA has 131 and observes all ten scripted intruders. The optional
public_route_residual_sa_transfer recipe exposes nine intruders solely to reuse an
MA checkpoint; it still simulates all ten.

New SAC models start with exactly zero deterministic adjustment and log standard
deviation -2. Warmup explores Gaussian residuals with standard deviation 0.15.
The trainer saves initial-model.zip before collection so the zero-adjustment
reference can be evaluated separately from trained checkpoints. An initial model
is an untrained reference and must not be reported as a learned improvement.
Training uses fresh replay; the submission loader checks the residual definition,
and resume rejects replay collected under a different mapping. The evaluator
records how often and how much the policy changes the navigation command.

```powershell
.venv\Scripts\python.exe -m atc.train --env ma --algorithm sac --recipe public_route_residual --guard-traffic --workers 1 --steps 100000 --seed 2900 --gradient-steps 4 --learning-starts 5000 --batch-size 256 --buffer-size 100000 --checkpoint-every 25000 --max-wall-seconds 3600 --save-replay --run-dir runs/sac-ma-route-residual-v1-100k-seed2900
.venv\Scripts\python.exe -m atc.train --env sa --algorithm sac --recipe public_route_residual --guard-traffic --workers 1 --steps 40000 --seed 2901 --gradient-steps 1 --learning-starts 1000 --batch-size 256 --buffer-size 50000 --checkpoint-every 10000 --max-wall-seconds 3600 --save-replay --run-dir runs/sac-sa-route-residual-v1-40k-seed2901
```

Run directories must be new; these names identify the first local pilots and should
not be reused. For Slurm, the existing launcher accepts ATC_RECIPE=public_route_residual
and ATC_GUARD_TRAFFIC=1. A larger training budget requires its own outcome evaluation;
functional checks do not establish policy quality.


## Stronger classical speed reference and verified failure traces

The classical goal controller defaults to a zero speed command. To test whether a
learned policy's efficiency gain exceeds a simple acceleration heuristic, evaluate
`--controller goal --goal-speed-action 1` with the same route input and joint filter.
The value is a constant normalized command in [-1, 1], not a target airspeed: +1
requests the original maximum speed increment each decision. The existing filter
and aircraft performance envelope still determine the executed motion. This option
is only valid without a model and is recorded in the evaluation metadata. Pure
classical controllers remain comparison references, not RL competition entries.

```powershell
.venv\Scripts\python.exe -m atc.evaluate --env ma --controller goal --goal-speed-action 1 --recipe public_route_input --guard-traffic --episodes 20 --out runs/fast-classical-ma-evaluation
```

For a selected MA or SA failure, `atc.diagnose --skip-prefix-rollouts --reference PREFIX`
can advance earlier scenario resets without simulating their trajectories. It
requires the same checkpoint hash, track, seed and inference mode as a saved
evaluation and checks all nine final metrics exactly before accepting the trace.
It also checks that the selected rollout did not advance the scenario RNG. A
mismatch fails explicitly; use the default full-prefix replay to investigate.
This shortcut was validated for residual checkpoint 25k, scenario index 12; its
validity for each subsequent use is checked against that use's own reference.

```powershell
.venv\Scripts\python.exe -m atc.diagnose --model runs/candidate-ma-route-residual-v1-25k/model.zip --episode 12 --skip-prefix-rollouts --reference runs/sac-ma-route-residual-v1-100k-seed2900/validation-25k-individual-20 --out runs/NEW_DIAGNOSTIC_DIRECTORY
```


## Clearance and path-length route selection

`public_route_choice` and `public_route_choice_residual` retain the corresponding
route-input and residual-control behavior while comparing feasible paths at
6, 3, 2, 1 and 0 km clearance. Among paths with at least 2 km clearance, select the
widest whose geometric length is within 15% of the shortest. Lower clearances are
considered only when no path at or above 2 km exists. This is geometric guidance;
the existing joint filter and original aircraft dynamics still determine motion.

Both recipes have `_sa_transfer` variants that expose nine traffic slots while
simulating all ten scripted intruders. Saved metadata records the route-choice
revision, levels, minimum preferred clearance and detour ratio. Evaluation and
submission reject stale metadata; changed routing requires fresh replay. Existing
`public_route_input` and `public_route_residual` recipes keep their original behavior.

```powershell
.venv\Scripts\python.exe -m atc.evaluate --env ma --controller goal --goal-speed-action 1 --recipe public_route_choice --guard-traffic --episodes 20 --out runs/NEW_ROUTE_CHOICE_CLASSICAL_RUN
.venv\Scripts\python.exe -m atc.train --env ma --algorithm sac --recipe public_route_choice_residual --guard-traffic --steps 25000 --seed 3000 --learning-starts 5000 --gradient-steps 4 --batch-size 256 --buffer-size 100000 --checkpoint-every 25000 --save-replay --run-dir runs/NEW_ROUTE_CHOICE_TRAINING_RUN
```

These commands illustrate configuration, not established performance. Changing
routing for an already trained policy is recorded as an ablation with zero additional
training; it does not demonstrate that the new routing was learned. Selected SA
failure-case replays improve arrival but include a new conflict; consult the matched
evaluation evidence in docs/competition/PILOT_RESULTS.md before selecting a model.

SA diagnostics record all scripted-intruder trajectories and process queued route
setup during skipped prefixes. Every selected rollout still has to match all nine
saved reference metrics exactly. Scenario indices are zero based.


## Larger speed-command experiment

`public_route_choice_speed20` and `public_route_choice_speed20_residual` use a
20-knot speed increment for normalized actions of magnitude one, instead of the
original 20/3 knots. Both have `_sa_transfer` variants. Heading commands remain
within 45 degrees and decisions remain ten seconds apart. The original A320
acceleration and speed limits still govern executed motion. The joint forecast
reads the same speed-action increment that the simulator command uses.

```powershell
.venv\Scripts\python.exe -m atc.evaluate --env ma --algorithm sac --recipe public_route_choice_speed20_residual --model runs/route-choice-speed20-v1-ma25k-ma/model.zip --episodes 20 --out runs/NEW_SPEED20_EVALUATION
```

This named ablation keeps the retained learned bytes unchanged; it has no training
with the larger command range. Four selected SA cases improve in some safety
metrics, but the first twenty scenarios do not show a clear general improvement.
Keep it separate from retained candidates. The physical speed mapping changes both
the policy's nominal speed request and the correction's available speed commands;
it is not an isolated braking-only change.

Configuration records `speed_command_increment_knots`. Evaluation now validates the
saved configuration before loading a model, as submission already does. Missing or
stale speed metadata is rejected; replay resume also rejects changed mapping. Old
recipes retain their original constructor defaults and metadata. Actual environment
checks confirm the 20-knot executor, 45-degree heading limit, ten-second decisions,
5 km capture radius, 5 NM separation and ten actual SA intruders. Training/resume and
negative-metadata evidence is under runs/route-choice-speed20-v1-smoke-audit.


## Residual learning from the faster speed reference

`public_route_choice_fast_residual` and its `_sa_transfer` variant retain the
original physical speed increment of 20/3 knots. A policy speed output `u` now
produces `clip(1 + 2*u, -1, 1)`. Zero output therefore matches the faster classical
reference exactly; `u=-0.5` holds speed and `u=-1` retains full deceleration.
Positive outputs saturate at the reference. Heading adjustments remain bounded
by 15 degrees, and the existing joint filter still processes the composed command.
This is an action-composition experiment, distinct from the 20-knot experiment.

The configuration records speed reference 1 and residual scale 2. Evaluation,
submission and replay resume require matching metadata. Original recipes retain
their prior mapping. Residual speed-adjustment counts measure departure from the
chosen reference; `absolute_speed_action` still measures the composed command.

New navigation SAC runs accept `--critic-warmup-updates N`, default zero. This holds
actor parameters fixed for the first N gradient updates while critics, target
critics and the entropy temperature learn. It is separate from `--learning-starts`,
which collects replay before any gradient updates. Resume inherits the saved count
and progress; an explicit warmup override on resume is rejected. Training summaries
record total critic, held-actor and actor update counts. The current MLP learner uses
one target update per gradient update.

The Slurm template accepts `ATC_CRITIC_WARMUP_UPDATES` for new models. The first
matched experiment uses 25,000 MA vector transitions, seed 2900, 5,000 replay-only
transitions and either zero or 2,000 held-actor updates. Equal environment/critic
budgets deliberately imply fewer actor updates in the warmup arm. Checkpoints at
25,000 transitions precede the final four gradient updates, matching earlier
comparisons. Consult PILOT_RESULTS.md for measured outcomes before selecting a model.


## Five-second control experiment

`public_route_choice_interval5` and `public_route_choice_fast_residual_interval5`
use five-second decisions. Both have `_sa_transfer` variants; all ten scripted
intruders remain in the SA world. Speed commands retain the original 20/3-knot
increment, heading commands retain their 45-degree limit, and the original
one-second scoring, separation distance, waypoint radius and episode limit remain.
More frequent decisions also allow more frequent speed and heading changes, so
this experiment does not isolate observation latency.

The first learned comparison copies the existing 25k checkpoint byte for byte.
It is a post-training control-rate ablation, with zero additional training.
Its five-second configuration records the source training configuration and the
original ten-second training interval. Do not report it as trained at five seconds.

```powershell
.venv\Scripts\python.exe -m atc.evaluate --env ma --algorithm sac --recipe public_route_choice_fast_residual_interval5 --model runs/fast-reference-interval5-v1-ma25k-ma/model.zip --episodes 20 --out runs/NEW_INTERVAL5_EVALUATION
```

Saved configuration includes `decision_interval_seconds=5`. Evaluation, submission
and replay resume reject missing or stale interval metadata. Original recipes keep
their prior defaults and metadata. The real simulator timing is checked in
runs/fast-reference-interval5-v1-ma25k-ma/runtime-audit.json; training and replay
resume are checked in runs/decision-interval-smoke-v1/completion-audit.json.

New training can use the same named recipe, but its discount remains defined per
decision. Changing the decision interval with the same gamma changes discounting
per simulated second. Equal transition budgets also cover less simulated time at
five seconds. These differences must be reported in any training comparison.


## Arrival-reward experiment

`public_route_choice_fast_residual_reach250` changes only `reach_reward` from
37.41716380974046 to 250. Its `_sa_transfer` variant keeps all ten actual intruders
and observes nine. Control timing remains ten seconds; safety penalties, gamma,
observations, route selection, residual mapping and filters remain unchanged.
Use fresh training and replay. This is one explicit reward-balance ablation,
not an established better configuration.

A fixed-policy SA replay verifies identical trajectory bytes and all eight objective
metrics under both reward settings. Only the terminal arrival bonus changes.
The motivation is a selected case where the old undiscounted reward favors a timeout
with fewer conflicts over an arrival with more conflicts; this does not describe
all possible trajectories or the discounted SAC optimum. The planned matched
25k training comparison is recorded in
runs/arrival-reward250-sa28-probe-v1/configuration-audit.json. Evaluate objective
metrics separately from training reward, which is not comparable between recipes.


## Verified replay previews

`atc.record` writes a GIF clip for each selected development scenario, using the
original simulator renderer and the exact saved model/configuration. It checks all
nine final metrics against an existing evaluation before saving a clip, and checks
that rendering did not advance the scenario RNG. It supports both tracks and runs
offscreen. Frame subsampling limits the animation size without truncating a rollout.

```powershell
.venv\Scripts\python.exe -m atc.record --reference runs/fast-reference-interval5-v1-ma25k-ma/validation-20 --model runs/fast-reference-interval5-v1-ma25k-ma/model.zip --episodes 0 1 2 3 4 --out runs/NEW_FIVE_SCENARIO_PREVIEW
```

Use a new output directory. `--episodes` uses zero-based evaluation indices;
displayed scenario labels start at one. Each clip shows simulation time, arrivals
and safety metrics, followed by a two-second result card. `recording.json` records
model/GIF hashes, final metrics, frame spacing, playback speed and comparison results.
These are selected development previews, not population scores or a final submission
video. The first five scenarios reproduce all nine metrics exactly on both MA and SA.
Verified combined previews are in runs/preview-fast-reference-interval5-v1-{ma,sa}/.
See each reel-audit.json for file hashes, frame counts and playback duration.
