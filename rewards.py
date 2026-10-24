"""Reward wrappers for the Lunar Lander experiments.

A wrapper sits between the game and PPO. Each step it takes the built-in
reward, adds or subtracts its own terms, and passes the total on. The game's
code is untouched.

Observation layout (LunarLander-v3), eight floats per frame:

    0  x       left / right of the pad   (0 = on the pad's centre line)
    1  y       height above the pad
    2  vx      sideways speed
    3  vy      up / down speed
    4  angle   tilt in radians           (0 = upright)
    5  omega   how fast it is spinning
    6  leg1    left leg on the ground    (0 or 1)
    7  leg2    right leg on the ground   (0 or 1)

Every wrapper here also stashes the *unshaped* env return in
``info["base_return"]`` when an episode ends. That is what lets the training
log show the real score next to the one PPO is chasing -- the gap between the
two is reward hacking, caught in the act.
"""

from __future__ import annotations

import inspect
import math

import gymnasium as gym

# Observation indices, named so the reward terms read like prose.
X, Y, VX, VY, ANGLE, OMEGA, LEG1, LEG2 = range(8)


class ShapedReward(gym.Wrapper):
    """Base class: env reward plus whatever ``extra_reward`` returns."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_obs = None
        self.base_return = 0.0
        self.shaped_return = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs
        self.base_return = 0.0
        self.shaped_return = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        extra = self.extra_reward(self._prev_obs, action, obs, terminated, truncated)
        self._prev_obs = obs

        self.base_return += reward
        total = reward + extra
        self.shaped_return += total

        # Per-step, so a replay can show the two rewards diverging live.
        info["base_reward"] = reward
        info["extra_reward"] = extra

        if terminated or truncated:
            info["base_return"] = self.base_return
            info["shaped_return"] = self.shaped_return
        return obs, total, terminated, truncated, info

    def extra_reward(self, prev_obs, action, obs, terminated, truncated) -> float:
        return 0.0


class Baseline(ShapedReward):
    """Changes nothing. Here so every run logs both returns the same way."""


class TimeFee(ShapedReward):
    """A small charge every frame, so dawdling costs something.

    The built-in reward has no such term: a lander that takes 600 frames to
    settle is scored the same as one that takes 200. ``fee`` is per frame, so
    at 0.03 a full 1,000-frame episode pays 30 points -- enough to prefer a
    brisk landing, small enough that it never prefers a crash (-100) over
    flying on.
    """

    def __init__(self, env: gym.Env, fee: float = 0.03):
        super().__init__(env)
        self.fee = fee

    def extra_reward(self, prev_obs, action, obs, terminated, truncated) -> float:
        return -self.fee


class HoverBonus(ShapedReward):
    """The deliberate mistake: pay per frame for *being* in a nice spot.

    "Plus 1 for every frame the rocket is upright and slow" sounds like
    exactly what you want. But an episode is cut off at 1,000 frames, so a
    rocket that simply hangs in the air can collect up to 1,000 points from
    this rule while paying at most ~300 in fuel. Landing ends the episode and
    the payments stop. Hovering beats landing, and PPO will find that out.
    """

    def __init__(
        self,
        env: gym.Env,
        bonus: float = 1.0,
        max_tilt: float = 0.1,
        max_speed: float = 0.1,
    ):
        super().__init__(env)
        self.bonus = bonus
        self.max_tilt = max_tilt
        self.max_speed = max_speed

    def extra_reward(self, prev_obs, action, obs, terminated, truncated) -> float:
        upright = abs(obs[ANGLE]) < self.max_tilt
        slow = math.hypot(obs[VX], obs[VY]) < self.max_speed
        return self.bonus if (upright and slow) else 0.0


class HoverBonusFixed(ShapedReward):
    """The fix: pay for *getting* better, not for sitting somewhere good.

    Two changes to ``HoverBonus``:

    1. The bonus becomes potential-based. Define a "how good is this attitude"
       number phi(s) = -k * (|tilt| + speed) and pay the *change* in it,
       gamma * phi(s') - phi(s), the way the built-in reward already pays the
       change in its own position score. A closed loop -- fly up, come back --
       nets out to zero, so there is nothing to farm by waiting. (This is
       Ng, Harada & Russell's 1999 result: shaping of this form cannot change
       which policy is optimal, only how fast it is found.)
    2. A per-frame fee, so waiting costs something regardless.

    phi is forced to 0 at a terminal state, which is what makes the guarantee
    hold: otherwise ending the episode would look like losing phi's worth of
    future bonus.
    """

    def __init__(
        self,
        env: gym.Env,
        k: float = 20.0,
        fee: float = 0.03,
        gamma: float = 0.999,
    ):
        super().__init__(env)
        self.k = k
        self.fee = fee
        self.gamma = gamma

    def phi(self, obs) -> float:
        return -self.k * (abs(obs[ANGLE]) + math.hypot(obs[VX], obs[VY]))

    def extra_reward(self, prev_obs, action, obs, terminated, truncated) -> float:
        phi_next = 0.0 if terminated else self.phi(obs)
        return self.gamma * phi_next - self.phi(prev_obs) - self.fee


REWARDS = {
    "baseline": Baseline,
    "timefee": TimeFee,
    "hover": HoverBonus,
    "hover-fixed": HoverBonusFixed,
}


def knobs(name: str) -> set[str]:
    """The tuning arguments a given wrapper actually accepts."""
    params = inspect.signature(REWARDS[name].__init__).parameters
    return set(params) - {"self", "env", "args", "kwargs"}


def build_wrapper(name: str, **kwargs):
    """Return a callable ``env -> wrapped env`` for ``make_vec_env``."""
    if name not in REWARDS:
        raise SystemExit(f"unknown reward {name!r}; pick one of {sorted(REWARDS)}")
    unusable = set(kwargs) - knobs(name)
    if unusable:
        raise SystemExit(
            f"reward {name!r} takes no {sorted(unusable)}; "
            f"it accepts {sorted(knobs(name)) or 'nothing'}"
        )
    cls = REWARDS[name]
    return lambda env: cls(env, **kwargs)
