#!/usr/bin/env bash
# Copy runs/ and videos/ from the training host's checkout into this one, over plain ssh.
#
#   scripts/pull_runs.sh            copy everything; keep files only this machine has
#   scripts/pull_runs.sh --mirror   first delete the local runs/ and videos/, for an exact copy
#
# Files with the same path are overwritten by the host's version. The host packs both
# folders with tar and streams them through ssh; nothing else needs to be installed.
set -euo pipefail

REMOTE="${REMOTE:-arhum@131.215.96.183}"
REMOTE_REPO="${REMOTE_REPO:-Documents/rl_racer}"  # relative to the remote home directory

mirror=false
for argument in "$@"; do
    case "$argument" in
        --mirror) mirror=true ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
if "$mirror"; then
    rm -rf runs videos
fi
ssh "$REMOTE" "cd $REMOTE_REPO && tar -czf - runs videos" | tar -xzvf -
