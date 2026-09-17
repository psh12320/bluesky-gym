export PATH="$PWD/bin:$PATH"
export ATC_PYTHON="$PWD/bin/python3"
chmod +x bin/*
bash jobs/submit_ppo_baseline_v1.sh
