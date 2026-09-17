# Shared PPO/MAPPO deployment in the competition harness

The atc_rl.deployment module supplies the two permitted environment/policy hooks for checkpoints trained by atc_rl.train. It keeps the original scenario generator, scoring methods, aircraft counts, dynamics and episode limits. The classical and SAC hooks in atc.submission remain separate.

The callable actor accepts one aircraft's local observation. Its critic input is zeroed, and deterministic inference uses the saved actor. The environment reconstructs the checkpoint's route input, predictive-feature mask, static/traffic filters, heading mapping and five-second action interval. Training-only reward scaling and potential shaping are disabled during evaluation.

For single-agent transfer, all ten scripted intruders remain simulated. Nine nearest traffic slots are exposed to match the multi-agent actor's input size. This is a transfer evaluation, not evidence that the policy was trained in the single-agent environment. The remaining-time observation is identical to the training collector's finite-horizon clock.

## Development evaluation

Run from the source root, retaining the original config.json, checkpoints.json and provenance.json beside the selected model:

```powershell
.\.venv\Scripts\python.exe -m atc_rl.competition --env ma --model <checkpoint-path> --episodes 20 --seed 20260 --out <new-result-directory>
.\.venv\Scripts\python.exe -m atc_rl.competition --env sa --model <checkpoint-path> --episodes 20 --seed 20260 --out <different-new-result-directory>
```

The entry point calls the original scripts.evaluate_competition run loops with deployment hooks installed in memory. Development runs use a separate seed and are explicitly marked non-official. It writes durable per-aircraft metrics, the original harness CSV, scenario hashes, checkpoint identity, source hashes and population/action-support checks. Only complete evaluations receive summary.json.

Seed 42 requires explicit --official, --seed 42 and --episodes 1000, plus a trained checkpoint. Reserved unseen streams 20301/20302 require --heldout. Candidate selection and learning-contribution checks should precede those evaluations.

## Judge-facing integration

The original harness's two marked edit points can call:

```python
from atc_rl.deployment import make_env, load_policy
```

Call load_policy before make_env, as the harness does. The source file of the harness is not modified by the development entry point. Environment and inference dependencies are checked against the checkpoint's original provenance before loading.
