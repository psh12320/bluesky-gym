# Heading command serialization

The primary MA held-out evaluation logged a heading-parser error in scenario
index 46. A separate, read-only diagnostic reproduced the error with the frozen
model and compiled geography backend. At simulated time 570 s, aircraft KL008
received this heading command:

```text
HDG KL008 7.843909918392455e-05
```

The number is finite. The installed BlueSky parser rejects exponent notation in
heading text, although the original HeadingAction can emit it for a small heading.
The ordinary decimal representation 0.00007843909918392455 parses successfully
and converts to exactly the same floating-point value. No non-finite observations
or policy actions occurred in the diagnostic's 1,925 heading commands.

This is a command-formatting defect, not evidence of an invalid neural-network
output. The diagnostic skips prefix rollouts while retaining all prefix scenario
resets. The completed primary 200-scenario CSV now confirms exact parity for all nine
metrics and every aircraft in this diagnostic case; see full-prefix-audit.json.

## Opt-in correction

atc/heading_transport.py provides DecimalHeadingAction and attach_decimal_heading.
The action class retains the original continuous/discrete normalization, relative
heading calculation, heading bound and action-space definition. It expands the
number to plain decimal text before queuing HDG. The speed action, simulator,
scenario generator, metric functions, policy weights and control interval are not
changed by this adapter.

The correction is explicitly opt-in. Existing recipes and the competition harness
still use the original heading class; the running frozen evaluations and training
replications therefore retain their declared behavior. A corrected deployment
needs a separately recorded transport revision and evaluation. It must not be
substituted silently into the already frozen results.

Seven targeted tests pass. They cover the exact rejected heading; 2,135 boundary,
random and subnormal values roundtripped through BlueSky's parser; non-finite value
rejection; and preservation of original continuous/discrete numeric heading targets.
Those were initially focused checks. The combined suite was subsequently run as
recorded below.

The corrected known-case replay has completed in
runs/heading-parser-ma47-decimal-v1/. All 1,925 heading commands parse, with zero
non-finite observations or actions. Every one of the eight physical metrics is
exactly unchanged for every aircraft compared with the original-format diagnostic.
The largest per-aircraft reward change is 0.000006109414542265768, so trajectory/
reward arithmetic is not bitwise identical. This is a bug-fix check on an inspected
scenario, not a new held-out performance result or proof that all scenarios are
unchanged by the transport correction.

## Evidence and next decisions

- Original-format observation and exact rejected string:
  runs/heading-parser-ma47-v1/diagnostic.json and invalid-commands.json.
- Opt-in formatter and meaningful numeric/parser checks:
  atc/heading_transport.py and tests/test_heading_transport.py.
- Corrected replay: runs/heading-parser-ma47-decimal-v1/.
- Primary MA results remain in runs/heldout-2027-ma-interval5-v1/; their interpretation
  must include the logged rejected command.

The correction now has the separately versioned deployment described below. Its
changed execution behavior still requires a complete evaluation and source record.
The old results and original frozen protocol remain preserved.

## Versioned deployment entry

atc/deployment.py now loads a deployment.json manifest, verifies the model hash,
checks the environment configuration and geography backend, and attaches the
corrected formatter. The original frozen files remain unchanged. The copied
candidates are in runs/decimal-transport-candidates-v1/{sa,ma}/ and contain the
same model bytes as the original selected candidates, with zero additional training.

The corrected entry replaces only the original harness's make_env and load_policy
hooks. It requires the full 1000-scenario protocol and rejects changed seed/count
constants. Use python -m atc.deployment with the original --env, --model and --out
arguments. These deployments deliberately have no legacy config.json file, so
accidentally using the old loader fails instead of silently ignoring the correction.

Twelve deployment validation tests pass. The original rollout loops with the new
hooks match every one of the nine recorded metrics exactly on two development
scenarios per track (2 SA and 20 MA aircraft records). These small checks do not
replace the still-required full corrected evaluation. Evidence is in
runs/decimal-transport-candidates-v1/integration-sa.json and integration-ma.json.

## Packaged reproduction check

output/candidates/decimal-heading-v1.zip contains 146 files, including the source,
two selected models, explicit deployment manifests and check scripts. Its SHA-256
is da020cf4079af110fcfcb7598b13f58e84859f8b5a0e549d9a96ad9ab5c86b2c.
Archive integrity and every listed file hash were verified.

A fresh extraction imported its own copies of atc, core, bluesky_gym and bluesky_zoo.
Both tracks reproduced all nine metrics exactly on their first two development
scenarios, using the compiled geography backend required by the manifests. This
validates the extracted files with the existing Python dependencies; installation
in a clean dependency environment has not been tested. Full corrected scoring
is running for both MA and SA. The external package-audit.json records both checks without
rewriting the immutable archive.

The full corrected-protocol runner is prepared at
runs/official-decimal-v1/run_official.py. Both track preflights pass: they verify
the archive, extracted checks, model manifests, current executable sources,
original harness constants and all code outside the two allowed integration hooks.
The MA full run started at 2026-09-15 13:28:24 UTC in
runs/official-decimal-v1/ma/. SA started at 13:37:09 UTC in
runs/official-decimal-v1/sa/. Both are active; do not restart or overwrite them. The runner preserves separate source,
protocol, logs and complete metrics, and refuses to overwrite an existing run.

The original MA harness CSV numbers finalized aircraft records, not scenarios
or named aircraft. The runner groups every ten consecutive records into one
scenario for aggregate clean completion; it labels positions as completion slots,
without inventing stable aircraft identities.

## Corrected five-scenario recordings

Both corrected deployments have now completed the first five development scenarios
with the original simulator renderer. All nine metrics match the prior development
records exactly for every aircraft: five SA records and fifty MA records. Each
scenario's random-generator state is unchanged during its rollout. These checks
cover the actual versioned loader and corrected heading action. They do not
replace the full scoring runs or establish population-level performance.

Evidence is in runs/preview-decimal-v1/completion-audit.json and the track-specific
recording manifests, protocols and GIFs. The corrected SA reel is byte-identical
to its original-format source reel; its MP4 is saved under
output/videos/decimal-heading-v1/. Both track MP4s are complete, preserve
source durations, decode without errors, and have inspected in-flight and final
frames. Their audit files distinguish encoded-file hashes from decoded frames.

For university execution, jobs/evaluate_candidate.slurm and
jobs/EVALUATE_CANDIDATE.md provide the fixed-archive, separate-workspace CPU job.
Its Bash and embedded Python syntax pass locally; cluster execution is unverified
and no job has been submitted. Each allocation runs its own short reproduction
check before the full original scoring loop.

## Combined regression suite

All 108 tests now pass together in 25.64 s, including the original 89 tests, the
seven heading-transport tests and twelve deployment tests. The suite has zero
failures, errors or skips. Its JUnit output and execution-source hashes are saved
in runs/decimal-transport-candidates-v1/regression-results.xml and
regression-audit.json. Every packaged execution file and test still matches the
immutable candidate archive, and both selected model hashes are unchanged.

This checks software behavior and interactions in the local dependency environment.
It does not imply that complete performance evaluations or the remaining training
replications have finished.
