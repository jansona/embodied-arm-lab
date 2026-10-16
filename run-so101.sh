#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${PYTHON:-.venv311/bin/python}
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
# Preview-only by default. This script never starts paid inference or loads keys.
exec "$PYTHON" -m uvicorn app_so101:app --host 127.0.0.1 --port 8766
