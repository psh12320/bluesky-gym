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
resets. Its complete per-aircraft metrics must still be compared with the primary
200-scenario CSV when that run completes. Do not assume skipped-prefix parity.

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
These are additional focused tests, not a new claim that the full prior suite ran.

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

After the original runs finish, decide how to integrate the verified correction
into a separately versioned deployment. Any changed execution behavior must retain
its own complete evaluation and source record. Neither old results nor the original
frozen protocol should be overwritten.

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
