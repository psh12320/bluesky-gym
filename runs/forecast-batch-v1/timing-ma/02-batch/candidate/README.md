# Corrected heading deployment: candidate v1

This is an inspectable deployment candidate, not a completed competition submission.
It contains source, two selected trained models, versioned deployment manifests and
specific integration evidence. Full corrected evaluations are still pending.

The learned weights are unchanged. Heading transport expands scientific notation
to ordinary decimal because the installed simulator rejects the former. The original
and corrected known-case diagnostics are included, including the small reward change.
The source retains the original simulator/scoring files and the permitted integration
hooks. Historical experiment documents describe results beyond the evidence packaged here.

## Verify and inspect

From the extracted directory, run `python verify_contents.py`. This checks every
listed file against the package manifest. Model hashes and transport settings are in
`models/sa/deployment.json` and `models/ma/deployment.json`.

With the recorded Python dependencies available, run:

```sh
python check_development.py --env sa
python check_development.py --env ma
```

These checks import the extracted source, run two development scenarios per track,
and compare every metric with the included reference. They do not use seed 42.
Use Python 3.12. The package's runtime-packages.txt records the actual Windows CPU
environment, including a local editable-install path; it is evidence, not a portable
cross-platform lockfile. Dependency reinstallation has not been validated by this
small source/model extraction check.

For an installable checkout, clone the upstream competition branch, check out base
commit 00930013219af4c17e509c3efc84a8becd0f3546, and overlay `source/` into that fresh
checkout. The upstream build obtains its version from Git. Install that checkout in
a compatible environment and retain the package's model/manifest directories.

## Full scoring entry

From `source/` (or the overlaid checkout), with the model path adjusted as needed:

```sh
python -m atc.deployment --env sa --model ../models/sa/model.zip --out ../results/sa.csv
python -m atc.deployment --env ma --model ../models/ma/model.zip --out ../results/ma.csv
```

This uses the original 1000-scenario, seed-42 scoring loops and replaces only the two
allowed integration hooks. Use a new output file and retain the exact source,
manifest, installed versions and console output. These commands run local scoring;
they do not submit anything or establish a judge-verified ranking. The legacy loader
cannot load these manifest-only packages, which prevents silent omission of the fix.

The development checks and diagnostic replay have narrower scope than a full
validation. The source and model package remains provisional until the corrected
full evaluations and final report are complete.
