"""Step 2: watch it fail.

Sends random actions to the lander so you can see what untrained flailing
looks like, and prints the eight numbers the AI actually gets to see.

    python random_agent.py                 # with a window
    python random_agent.py --no-render     # headless, just the numbers
"""

from __future__ import annotations

import argparse

import gymnasium as gym

LABELS = [
    "x      left/right of pad",
    "y      height above pad",
    "vx     sideways speed",
    "vy     up/down speed",
    "angle  tilt (radians)",
    "omega  spin rate",
    "leg1   left leg down",
    "leg2   right leg down",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--no-render", action="store_true", help="run without a window")
    p.add_argument("--print-every", type=int, default=25, help="frames between printouts")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make("LunarLander-v3", render_mode=None if args.no_render else "human")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total, frames = 0.0, 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            frames += 1

            if frames % args.print_every == 0:
                print(f"\n  frame {frames:4d}   running score {total:8.2f}")
                for label, value in zip(LABELS, obs):
                    print(f"    {label:28s} {value:7.3f}")

            if terminated or truncated:
                how = "crashed / ended" if terminated else "ran out of time"
                print(f"\nepisode {ep + 1}: {how} after {frames} frames, score {total:.1f}")
                break

    env.close()
    print("\nThat is what random looks like. Scores around -200 to -500 are normal.")


if __name__ == "__main__":
    main()
