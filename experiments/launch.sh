#!/usr/bin/env bash
# usage: experiments/launch.sh ppo|sac NAME [runner options...]
# Starts train_<algorithm>.py in the background, logging to experiments/logs/NAME.log.
cd "$(dirname "$0")/.." || exit 1
mkdir -p experiments/logs
export TORCHINDUCTOR_CACHE_DIR=$PWD/experiments/cache/inductor TRITON_CACHE_DIR=$PWD/experiments/cache/triton
nohup uv run python "train_$1.py" --name "$2" "${@:3}" < /dev/null > "experiments/logs/$2.log" 2>&1 &
echo "$2 pid $!"
