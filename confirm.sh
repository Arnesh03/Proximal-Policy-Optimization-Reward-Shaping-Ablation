#!/usr/bin/env bash
# Confirmation runs. Training is stochastic, so neither the hack nor the fix
# gets claimed off a single seed.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
HACK=(--bonus 3.0 --max-tilt 0.2 --max-speed 0.3)

run () {                            # name, reward, then train.py args
  local name=$1 reward=$2; shift 2
  echo "############ $name ############"
  $PY train.py --reward "$reward" --name "$name" --steps 1000000 "$@"
  $PY evaluate.py --model "models/$name.zip" --episodes 20 \
      $([ "$reward" = baseline ] || echo --reward "$reward")
}

# Does the fix hold at the strength that broke it? The claim is that the
# potential-based *form* is what cures it, not a gentler number.
run hover-fixed-k60 hover-fixed --k 60

# Does the hack reproduce on other seeds?
run hover-hack-s1 hover "${HACK[@]}" --seed 1
run hover-hack-s2 hover "${HACK[@]}" --seed 2

# Does the control reproduce?
run baseline-s1 baseline --seed 1
