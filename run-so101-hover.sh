#!/bin/sh
# Explicit bounded experimental hover UI; inference workers remain separately capped.
set -eu
export SO101_ENABLE_WAYPOINT=1
exec sh "$(dirname "$0")/run-so101.sh"
