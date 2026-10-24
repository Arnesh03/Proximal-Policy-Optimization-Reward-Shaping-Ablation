"""A side-by-side strip of what each policy actually does with its time.

The score tells you something is wrong; this tells you what.

    python contact_sheet.py baseline="lands it" hover-hacked="hovers forever" \
        --out logs/contact_sheet.png

Each row is one episode of one model, sampled at even intervals, so a policy
that never descends is obvious at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

ENV_ID = "LunarLander-v3"


def rollout(model_path: str, seed: int) -> tuple[list, float, str]:
    """One deterministic episode; returns its frames, score and outcome."""
    model = PPO.load(model_path, device="cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    frames, total, last = [], 0.0, 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        last = reward
    env.close()

    outcome = ("never lands" if truncated else
               "lands" if last >= 99 else
               "crashes" if last <= -99 else "ends")
    return frames, total, outcome


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clips", nargs="+", help="name=label, name being models/<name>.zip")
    p.add_argument("--out", default="logs/contact_sheet.png")
    p.add_argument("--cols", type=int, default=5, help="frames sampled per row")
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    specs = []
    for clip in args.clips:
        name, _, label = clip.partition("=")
        specs.append((name, label or name))

    fig, axes = plt.subplots(
        len(specs), args.cols,
        figsize=(2.6 * args.cols, 2.0 * len(specs)),
        squeeze=False,
    )

    for row, (name, label) in enumerate(specs):
        frames, score, outcome = rollout(f"models/{name}.zip", args.seed)
        # Even spacing across the whole episode, first and last frame included.
        picks = [round(i * (len(frames) - 1) / (args.cols - 1)) for i in range(args.cols)]
        for col, idx in enumerate(picks):
            ax = axes[row][col]
            ax.imshow(frames[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_edgecolor("0.8")
            if row == 0:
                ax.set_title(f"frame {idx}", fontsize=9, color="0.3")
            else:
                ax.set_xlabel(f"frame {idx}", fontsize=9, color="0.3")
        axes[row][0].set_ylabel(
            f"{label}\n{outcome}, score {score:.0f}",
            fontsize=10, rotation=0, ha="right", va="center", labelpad=14,
        )

    fig.suptitle("Same 1,000-frame budget, two very different ideas of success",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
