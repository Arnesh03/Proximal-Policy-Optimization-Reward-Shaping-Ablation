"""Step 4: watch it land, and keep the clip.

Writes an mp4 and a gif of the policy flying the RAW environment.

    python record.py --model models/baseline.zip --name baseline --episodes 3

Gymnasium's RecordVideo wrapper is the stock way to do this; this script
collects the rgb_array frames itself and hands them to imageio instead, which
gets both an mp4 and a gif out of a single pass and has one less dependency to
go wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

ENV_ID = "LunarLander-v3"
FPS = 50  # the environment's own frame rate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True, help="output basename under videos/")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--gif-every", type=int, default=2, help="keep 1 frame in N for the gif")
    p.add_argument("--max-frames", type=int, default=3000, help="stop after this many frames")
    args = p.parse_args()

    out_dir = Path("videos")
    out_dir.mkdir(exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")

    frames: list[np.ndarray] = []
    summaries: list[str] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total, n, last = 0.0, 0, 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            last = reward
            n += 1
            if len(frames) >= args.max_frames:
                break

        outcome = ("timeout" if truncated else
                   "landed" if last >= 99 else
                   "crashed" if last <= -99 else "ended")
        summaries.append(f"  episode {ep + 1}: {outcome:8s} {n:4d} frames  score {total:7.1f}")
        if len(frames) >= args.max_frames:
            break

    env.close()

    mp4 = out_dir / f"{args.name}.mp4"
    gif = out_dir / f"{args.name}.gif"
    imageio.mimsave(mp4, frames, fps=FPS, macro_block_size=None)
    imageio.mimsave(gif, frames[:: args.gif_every], duration=1000 / (FPS / args.gif_every))

    print("\n".join(summaries))
    print(f"\n  {mp4}  ({len(frames)} frames)")
    print(f"  {gif}")


if __name__ == "__main__":
    main()
