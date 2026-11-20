#!/usr/bin/env bash
# usage: experiments/stop.sh NAME
# Stops that run and deletes its folder without promoting it; copy its best.pt first to keep it.
cd "$(dirname "$0")/.." || exit 1
pkill -f "train_(ppo|sac)\.py --name $1( |\$)"
sleep 15
rm -rf runs/ppo_current/"$1" runs/sac_current/"$1"
echo "stopped $1"
