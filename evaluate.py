"""How you know it works: the average true score over many attempts.

Runs the RAW environment by default -- no reward wrapper -- so your own bonus
cannot flatter the number. Lunar Lander counts 200+ as solved, so you want the
mean above that.

    python evaluate.py --model models/baseline.zip --episodes 20

Pass --reward to also see the shaped score the same policy would have been
paid during training. For a hacked reward the two numbers are miles apart, and
the outcome breakdown says why: no landings, every episode timing out.
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from rewards import LEG1, LEG2, VX, VY, REWARDS, Y, build_wrapper

ENV_ID = "LunarLander-v3"
SOLVED = 200.0


def run(model, env, episodes: int, seed: int, deterministic: bool) -> dict:
    """Fly `episodes` attempts and collect what happened in each."""
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, frames, last_reward = 0.0, 0, 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            last_reward = reward
            frames += 1

        # LunarLander replaces the final frame's reward with exactly +100 for
        # coming to rest safely or -100 for a crash, so this read is exact.
        if truncated:
            outcome = "timeout"
        elif last_reward >= 99:
            outcome = "landed"
        elif last_reward <= -99:
            outcome = "crashed"
        else:
            outcome = "ended"

        rows.append(
            {
                "return": total,
                "frames": frames,
                "outcome": outcome,
                "final_height": float(obs[Y]),
                "final_legs": float(obs[LEG1] + obs[LEG2]),
                "final_speed": float(np.hypot(obs[VX], obs[VY])),
            }
        )
    return rows


def report(title: str, rows: list[dict], solved_threshold: bool = True) -> float:
    returns = [r["return"] for r in rows]
    mean = statistics.fmean(returns)
    sd = statistics.stdev(returns) if len(returns) > 1 else 0.0

    print(f"\n{title}")
    print("-" * len(title))
    print(f"  episodes      {len(rows)}")
    print(f"  mean return   {mean:8.1f}  +/- {sd:.1f}")
    print(f"  worst / best  {min(returns):8.1f} / {max(returns):.1f}")
    print(f"  mean frames   {statistics.fmean(r['frames'] for r in rows):8.1f}")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    breakdown = "  ".join(
        f"{k} {v}/{len(rows)}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    print(f"  outcomes      {breakdown}")

    stuck = [r for r in rows if r["outcome"] == "timeout"]
    if stuck:
        height = statistics.fmean(r["final_height"] for r in stuck)
        legs = statistics.fmean(r["final_legs"] for r in stuck)
        speed = statistics.fmean(r["final_speed"] for r in stuck)
        print(f"  at the 1000-frame cutoff ({len(stuck)} episodes):"
              f"  height {height:.2f}   legs down {legs:.1f}/2   speed {speed:.2f}")
        # Two very different ways to never finish, and they need telling apart.
        if height > 0.25 and legs < 0.5:
            print("      -> HOVERING: airborne, legs up, going nowhere")
        elif legs > 1.5:
            print("      -> REFUSING TO SETTLE: on the pad but still firing, so"
                  " the lander never falls asleep and the episode never ends")
        else:
            print("      -> neither cleanly; look at the clip")

    if solved_threshold:
        verdict = "SOLVED" if mean >= SOLVED else "not solved"
        print(f"  verdict       {verdict}  (threshold {SOLVED:.0f})")
    return mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--reward", default=None, choices=sorted(REWARDS),
                   help="also score with this wrapper on, for comparison")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the most likely one")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    deterministic = not args.stochastic

    env = gym.make(ENV_ID)
    true_rows = run(model, env, args.episodes, args.seed, deterministic)
    env.close()
    true_mean = report(f"TRUE reward (wrapper off) -- {args.model}", true_rows)

    if args.reward:
        wrapped = build_wrapper(args.reward)(gym.make(ENV_ID))
        shaped_rows = run(model, wrapped, args.episodes, args.seed, deterministic)
        wrapped.close()
        shaped_mean = report(
            f"SHAPED reward ('{args.reward}' wrapper on) -- what training paid",
            shaped_rows,
            solved_threshold=False,
        )
        print(f"\n  shaping gap   {shaped_mean - true_mean:+8.1f}"
              "   <- how much the wrapper inflates the score")
    print()


if __name__ == "__main__":
    main()
