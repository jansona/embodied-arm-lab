#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python
    elif [ -x .venv311/bin/python ]; then PYTHON=.venv311/bin/python
    else PYTHON=python3; fi
fi
# Preview by default; live workers require explicit --live --config.
exec "$PYTHON" -m embodied_arm "$@"
