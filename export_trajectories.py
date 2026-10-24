"""Export real episodes to JSON so a web UI can replay them exactly.

Nothing here is an approximation: every frame is Box2D's own output, with the
lander's and legs' world vertices taken straight off the bodies, so the browser
only has to draw polygons it is handed.

    python export_trajectories.py --out ui/trajectories.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from rewards import build_wrapper

ENV_ID = "LunarLander-v3"

# name, model, reward wrapper, wrapper kwargs, label, blurb
CLIPS = [
    ("baseline", "baseline", "baseline", {},
     "Built-in reward",
     "The game's own scoring, untouched. The control."),
    ("timefee", "timefee", "timefee", {},
     "Built-in + time fee",
     "A charge of 0.03 per frame, so dawdling costs something. Still lands."),
    ("hacked", "hover-b3-loose", "hover",
     {"bonus": 3.0, "max_tilt": 0.2, "max_speed": 0.3},
     "+3/frame for upright & slow",
     "Sounds like what you want. Pays for being somewhere nice, so the rocket "
     "stops and collects."),
    ("fixed", "hover-fixed", "hover-fixed", {"k": 20.0, "fee": 0.03, "gamma": 0.999},
     "The same idea, paid correctly",
     "Pays the change in uprightness rather than the level, plus a time fee. "
     "Nothing to farm by waiting."),
]


def verts(body) -> list[list[float]]:
    """A body's fixture vertices in world coordinates, rounded for size."""
    shape = body.fixtures[0].shape
    return [[round(c, 3) for c in body.transform * v] for v in shape.vertices]


def capture(model_path: str, reward: str, kwargs: dict, seed: int) -> dict:
    model = PPO.load(model_path, device="cpu")
    env = build_wrapper(reward, **kwargs)(gym.make(ENV_ID))

    obs, _ = env.reset(seed=seed)
    core = env.unwrapped

    # The terrain is a chain of segments; each sky polygon's first two points
    # are the ground points under it.
    ground = [list(core.sky_polys[0][0])] + [list(p[1]) for p in core.sky_polys]

    frames, base_total, shaped_total = [], 0.0, 0.0
    terminated = truncated = False
    last = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        frames.append(
            {
                "l": verts(core.lander),
                "g": [verts(leg) for leg in core.legs],
                # Box2D reorders a polygon's vertices when it builds the hull,
                # so the body transform is exported rather than inferred.
                "p": [round(core.lander.position[0], 4),
                      round(core.lander.position[1], 4)],
                "t": round(float(core.lander.angle), 5),
                "a": action,
                "o": [round(float(v), 4) for v in obs],
                "br": round(base_total, 2),
                "sr": round(shaped_total, 2),
            }
        )
        obs, total, terminated, truncated, info = env.step(action)
        base_total += info["base_reward"]
        shaped_total += info["base_reward"] + info["extra_reward"]
        last = info["base_reward"]

    env.close()
    outcome = ("never lands" if truncated else
               "lands" if last >= 99 else
               "crashes" if last <= -99 else "ends")
    return {
        "ground": [[round(c, 3) for c in p] for p in ground],
        "pad": [round(core.helipad_x1, 3), round(core.helipad_x2, 3),
                round(core.helipad_y, 3)],
        "frames": frames,
        "trueReturn": round(base_total, 1),
        "shapedReturn": round(shaped_total, 1),
        "outcome": outcome,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="ui/trajectories.json")
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    out = {"seed": args.seed, "fps": 50, "world": [20.0, 13.333], "clips": {}}
    for key, model, reward, kwargs, label, blurb in CLIPS:
        path = f"models/{model}.zip"
        if not Path(path).exists():
            print(f"  skipping {key}: no {path}")
            continue
        clip = capture(path, reward, kwargs, args.seed)
        clip.update(label=label, blurb=blurb, model=model, reward=reward)
        out["clips"][key] = clip
        print(f"  {key:9s} {len(clip['frames']):4d} frames  "
              f"true {clip['trueReturn']:7.1f}  shaped {clip['shapedReturn']:7.1f}  "
              f"{clip['outcome']}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")))
    print(f"\n  wrote {dest}  ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
