#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
export JEV_USE_IPC=1
exec ./scripts/run.sh
