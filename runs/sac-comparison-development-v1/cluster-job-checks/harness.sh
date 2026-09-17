#!/bin/bash
set -euo pipefail
chmod +x "$1/python-check" "$1/sbatch"
export PATH="$1:$PATH"
export ATC_PYTHON="$1/python-check"
export ATC_TEST_DIR="$3"
cd "$3"
bash "$2"
