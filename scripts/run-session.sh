#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
if [ -x .venv311/bin/python ]; then PY=.venv311/bin/python; else PY=.venv/bin/python; fi
exec "$PY" -m embodied_arm.runtime.start_session "$@"
