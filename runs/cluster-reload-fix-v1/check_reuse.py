from pathlib import Path
import json
import subprocess

repo = Path.cwd()
work = repo / 'runs/cluster-reload-fix-v1'
script = work / 'check_reuse.sh'
script.write_text('''#!/bin/bash
set -euo pipefail
JOB=$(cygpath -u "$1")
WORK=$(cygpath -u "$2")
mkdir "$WORK/mock-bin" "$WORK/old-environment"
cat > "$WORK/mock-bin/python3" <<'MOCK'
#!/bin/bash
printf '%s\\n' "system-python $*" >> "$ATC_TEST_LOG"
if [[ "$*" == *'pip install'* || "$*" == *'-m venv'* ]]; then exit 91; fi
MOCK
cat > "$WORK/old-environment/python" <<'MOCK'
#!/bin/bash
printf '%s\\n' "reused-python $*" >> "$ATC_TEST_LOG"
if [[ "$*" == *'pip install'* || "$*" == *'-m venv'* ]]; then exit 92; fi
MOCK
cat > "$WORK/mock-bin/nvidia-smi" <<'MOCK'
#!/bin/bash
printf '%s\\n' 'nvidia-smi' >> "$ATC_TEST_LOG"
MOCK
chmod +x "$WORK/mock-bin/python3" "$WORK/mock-bin/nvidia-smi" "$WORK/old-environment/python"
export PATH="$WORK/mock-bin:$PATH"
export ATC_PYTHON="$WORK/old-environment/python"
export ATC_TEST_LOG="$WORK/reuse-commands.txt"
export SLURM_JOB_ID=900001
export SLURM_SUBMIT_DIR="$WORK/fresh-source"
mkdir "$SLURM_SUBMIT_DIR"
bash "$JOB" > "$WORK/reuse-output.txt" 2>&1
if grep -E 'pip install| -m venv' "$ATC_TEST_LOG"; then exit 93; fi
grep -q 'reused-python -m pip check' "$ATC_TEST_LOG"
grep -q 'reused-python -m pytest tests/rl' "$ATC_TEST_LOG"
grep -q 'reused-python -m atc_rl.train' "$ATC_TEST_LOG"
grep -q 'reused-python -m atc_rl.evaluate' "$ATC_TEST_LOG"
test -f "$SLURM_SUBMIT_DIR/runs/cluster-check-900001/packages.txt"
printf '%s\\n' preserved > "$SLURM_SUBMIT_DIR/runs/cluster-check-900001/sentinel"
if bash "$JOB" > "$WORK/duplicate-output.txt" 2>&1; then exit 94; fi
test "$(cat "$SLURM_SUBMIT_DIR/runs/cluster-check-900001/sentinel")" = preserved
export ATC_PYTHON="$WORK/missing/python"
export SLURM_JOB_ID=900002
if bash "$JOB" > "$WORK/missing-output.txt" 2>&1; then exit 95; fi
grep -q 'ATC_PYTHON must name an existing executable' "$WORK/missing-output.txt"
test ! -e "$SLURM_SUBMIT_DIR/runs/cluster-check-900002"
unset ATC_PYTHON
mkdir "$SLURM_SUBMIT_DIR/.venv-cluster"
if bash "$JOB" > "$WORK/existing-env-output.txt" 2>&1; then exit 96; fi
grep -q 'Existing environment preserved' "$WORK/existing-env-output.txt"
if grep -E 'pip install| -m venv' "$ATC_TEST_LOG"; then exit 97; fi
printf '%s\\n' 'Reuse branch passed; no installs; existing outputs and environment preserved; missing interpreter rejected.'
''', encoding='utf-8', newline='\n')
job = repo / 'runs/onpolicy-cluster-reload-fix-v1-verify/jobs/bootstrap_onpolicy.slurm'
command = ['C:/Program Files/Git/bin/bash.exe', str(script), str(job), str(work)]
result = subprocess.run(command, text=True, capture_output=True)
record = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
          'scope': 'Stubbed shell control-flow check; not a CUDA or Slurm execution.'}
(work / 'reuse-shell-check.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
assert result.returncode == 0