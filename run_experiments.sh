#!/usr/bin/env bash
# The whole project, start to finish. Each run is ~1M frames.
#
#   ./run_experiments.sh            # the four reward variants, seed 0
#   ./run_experiments.sh seeds      # extra seeds, because one run proves nothing
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
STEPS=${STEPS:-1000000}

train_and_score () {               # reward, name, extra train args...
  local reward=$1 name=$2; shift 2
  echo "############ $name ############"
  $PY train.py --reward "$reward" --name "$name" --steps "$STEPS" "$@"
  $PY evaluate.py --model "models/$name.zip" --episodes 20 \
      $([ "$reward" = baseline ] || echo --reward "$reward")
  $PY record.py --model "models/$name.zip" --name "$name" --episodes 3
}

if [ "${1:-main}" = seeds ]; then
  for seed in 1 2; do
    train_and_score baseline "baseline-s$seed" --seed "$seed"
    train_and_score hover    "hover-s$seed"    --seed "$seed" \
        --bonus 3.0 --max-tilt 0.2 --max-speed 0.3
  done
  exit 0
fi

train_and_score baseline    baseline     # step 3: does PPO land at all?
train_and_score timefee     timefee      # step 5: a reward of your own, harmless
train_and_score hover       hover        --bonus 3.0 --max-tilt 0.2 --max-speed 0.3
                                         # step 6: break it on purpose. At the
                                         # default +1/tight tolerance PPO still
                                         # lands; these are the settings where the
                                         # hack actually wins (see hack_search.sh).
train_and_score hover-fixed hover-fixed --k 60
                                         # step 6: and fix it, at matched strength --
                                         # the cure is the potential-based *form*,
                                         # not a smaller number.

$PY plot_curves.py baseline timefee hover hover-fixed --out logs/curves.png
