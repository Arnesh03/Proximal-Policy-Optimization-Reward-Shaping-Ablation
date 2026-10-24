"""Step 3 and 5: train PPO, with or without a reward wrapper of your own.

    python train.py --reward baseline     --steps 1000000 --name baseline
    python train.py --reward timefee      --steps 1000000 --name timefee
    python train.py --reward hover        --steps 1000000 --name hover
    python train.py --reward hover-fixed  --steps 1000000 --name hover-fixed

Two numbers scroll past while it runs:

    rollout/ep_rew_mean          the TRUE score -- Monitor sits inside the
                                 reward wrapper, so this is the game's own
                                 reward, comparable across every run here
    rollout/shaped_ep_rew_mean   what PPO is actually chasing, i.e. the true
                                 score plus your extra terms

When those two diverge -- the shaped one climbing while the true one stalls or
sinks -- you are watching reward hacking happen.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rewards import REWARDS, build_wrapper, knobs

ENV_ID = "LunarLander-v3"


class ShapedReturnLogger(BaseCallback):
    """Logs the shaped return alongside SB3's own (true) ep_rew_mean."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.shaped = deque(maxlen=window)
        self.base = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "shaped_return" in info:
                self.shaped.append(info["shaped_return"])
                self.base.append(info["base_return"])
        return True

    def _on_rollout_end(self) -> None:
        if self.shaped:
            self.logger.record("rollout/shaped_ep_rew_mean", float(np.mean(self.shaped)))
            self.logger.record("rollout/true_ep_rew_mean", float(np.mean(self.base)))
            self.logger.record(
                "rollout/shaping_gap",
                float(np.mean(self.shaped)) - float(np.mean(self.base)),
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reward", default="baseline", choices=sorted(REWARDS))
    p.add_argument("--steps", type=int, default=1_000_000, help="frames to learn for")
    p.add_argument("--name", default=None, help="output name (default: the reward name)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    # Reward-wrapper knobs. Passing one the chosen wrapper does not take is
    # an error, not a silently ignored flag.
    p.add_argument("--fee", type=float, default=None, help="per-frame charge")
    p.add_argument("--bonus", type=float, default=None, help="hover bonus per frame")
    p.add_argument("--max-tilt", type=float, default=None, help="hover: tilt tolerance")
    p.add_argument("--max-speed", type=float, default=None, help="hover: speed tolerance")
    p.add_argument("--k", type=float, default=None, help="hover-fixed: potential scale")
    # PPO hyperparameters: the RL Baselines3 Zoo settings tuned for this env.
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.98)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--no-eval", action="store_true", help="skip periodic clean-env eval")
    p.add_argument("--eval-freq", type=int, default=25_000, help="steps between evals")
    args = p.parse_args()

    name = args.name or args.reward
    log_dir = Path("logs") / name
    log_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    wrapper_kwargs = {
        name: value
        for name, value in (
            ("fee", args.fee),
            ("bonus", args.bonus),
            ("max_tilt", args.max_tilt),
            ("max_speed", args.max_speed),
            ("k", args.k),
        )
        if value is not None
    }
    # The potential-based fix needs the same discount the learner uses.
    if "gamma" in knobs(args.reward):
        wrapper_kwargs.setdefault("gamma", args.gamma)

    wrapper = build_wrapper(args.reward, **wrapper_kwargs)

    env = make_vec_env(
        ENV_ID,
        n_envs=args.n_envs,
        seed=args.seed,
        wrapper_class=wrapper,
        monitor_dir=str(log_dir / "monitor"),
    )

    model = PPO(
        "MlpPolicy",  # the standard small net: two hidden layers of 64
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        seed=args.seed,
        verbose=1,
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))

    callbacks: list[BaseCallback] = [ShapedReturnLogger()]
    if not args.no_eval:
        # Deliberately the raw env -- no wrapper -- so "best" means best at
        # the real game, not best at collecting your bonus.
        eval_env = Monitor(gym.make(ENV_ID))
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(log_dir / "best"),
                log_path=str(log_dir),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=5,
                deterministic=True,
                verbose=0,
            )
        )

    print(f"\ntraining '{name}'  reward={args.reward} {wrapper_kwargs or ''}  "
          f"steps={args.steps:,}  envs={args.n_envs}  seed={args.seed}\n")

    started = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - started

    out = Path("models") / f"{name}.zip"
    model.save(out)
    env.close()

    meta = {
        "name": name,
        "reward": args.reward,
        "wrapper_kwargs": wrapper_kwargs,
        "steps": args.steps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "wall_clock_seconds": round(elapsed, 1),
        "model": str(out),
    }
    (log_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\ndone in {elapsed / 60:.1f} min -> {out}")
    print(f"now: python evaluate.py --model {out} --episodes 20")


if __name__ == "__main__":
    main()
