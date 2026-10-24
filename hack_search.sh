#!/usr/bin/env bash
# The hover bonus is profitable in principle at +1/frame, but PPO still learned
# to land. This searches for the point where the hack actually wins: a steeper
# bonus, a tolerance loose enough to stumble into, or both.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
STEPS=${STEPS:-1000000}

try () {                            # name, then train.py args
  local name=$1; shift
  echo "############ $name ############"
  $PY train.py --reward hover --name "$name" --steps "$STEPS" "$@"
  $PY evaluate.py --model "models/$name.zip" --episodes 20 --reward hover
}

try hover-b3                   --bonus 3.0
try hover-loose  --bonus 1.0   --max-tilt 0.2 --max-speed 0.3
try hover-b3-loose --bonus 3.0 --max-tilt 0.2 --max-speed 0.3
