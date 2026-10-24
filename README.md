# Lunar Lander: teaching a rocket to land, then breaking it on purpose

A reinforcement-learning project in four parts: get PPO to land the Gymnasium
lander, write a reward of my own, break the agent with a reward that sounds
sensible, and fix it. Runs on a laptop CPU in minutes, no GPU.

The interesting part is not the landing. It is that **an agent optimises
exactly what you pay it for**, which is rarely exactly what you meant.

## What's here

| File | What it does |
|---|---|
| [`rewards.py`](rewards.py) | The reward wrappers — the actual experiment |
| [`random_agent.py`](random_agent.py) | Untrained flailing, plus the 8 numbers the agent sees |
| [`train.py`](train.py) | PPO training, with any reward wrapper |
| [`evaluate.py`](evaluate.py) | Scores a saved policy on the **raw** env |
| [`record.py`](record.py) | mp4 + gif of a policy flying |
| [`plot_curves.py`](plot_curves.py) | Learning curves: paid score vs. true score |
| [`run_experiments.sh`](run_experiments.sh) | The whole sweep, end to end |
| [`hack_search.sh`](hack_search.sh) | Finds the bonus setting where the hack wins |
| [`confirm.sh`](confirm.sh) | Extra seeds, so nothing rests on one run |
| [`contact_sheet.py`](contact_sheet.py) | Side-by-side strip of two policies flying |
| [`export_trajectories.py`](export_trajectories.py) | Dumps real episodes to JSON for the UI |
| [`ui/index.html`](ui/index.html) | Replay console + hand-flying mode |
| [`ui/sim.js`](ui/sim.js) | LunarLander-v3 ported to the browser, for flying it yourself |
| [`ui/craft3d.js`](ui/craft3d.js) | Shaded 3D lunar module with quilted insulation, greebling and LOD |
| [`build_site.py`](build_site.py) | Builds `docs/`, the deployable static site |

## Setup

```bash
uv venv --python 3.12 .venv            # or: python3 -m venv .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
```

Then, in order:

```bash
.venv/bin/python random_agent.py                              # watch it fail
./run_experiments.sh                                          # train all four + score + record
./run_experiments.sh seeds                                    # extra seeds
```

Or one variant at a time:

```bash
.venv/bin/python train.py --reward baseline --steps 1000000 --name baseline
.venv/bin/python evaluate.py --model models/baseline.zip --episodes 20
.venv/bin/python record.py --model models/baseline.zip --name baseline
```

## How the pieces fit

The environment hands the agent **eight numbers** per frame at 50 fps — pad-relative
position, velocity, tilt, spin, and a contact flag per leg. It never sees pixels.
The agent picks one of **four actions**: nothing, left engine, main engine, right
engine. The policy is an MLP of two 64-unit layers (SB3's `MlpPolicy`), trained
with **PPO**, whose whole idea is to keep each update small so one lucky stretch
cannot wreck what has been learned.

Hyperparameters are the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
settings tuned for this env (`n_envs=16`, `n_steps=1024`, `batch_size=64`,
`n_epochs=4`, `gamma=0.999`, `gae_lambda=0.98`, `ent_coef=0.01`).

### The built-in reward

Each frame the env computes a "how good is this position" number — roughly
`-100·distance - 100·speed - 100·|tilt| + 10·(legs down)` — and the frame's
reward is **how much that number improved since last frame**, minus fuel
(0.3 main engine, 0.03 side engine). The final frame's reward is replaced by
`+100` for coming to rest or `-100` for a crash.

Paying the *change* rather than the *level* is the detail that matters. It is
what the broken reward below gets wrong and what the fix restores.

### Honest measurement, by construction

SB3's `make_vec_env` wraps `Monitor` **inside** a custom wrapper, which turns
out to be exactly what this project needs: `rollout/ep_rew_mean` is always the
**game's own** reward, comparable across every run here, while
`rollout/shaped_ep_rew_mean` (logged by a callback in `train.py`) is what PPO is
actually being paid. Those two lines pulling apart *is* reward hacking, visible
live in the training log. `evaluate.py` runs the raw env by default for the same
reason: a bonus of mine must never be able to flatter the number.

## The four rewards

**`baseline`** — the env's reward, untouched. The control.

**`timefee`** — `−0.03` per frame. The built-in reward charges nothing for
dawdling: a lander that takes 600 frames is scored like one that takes 200.
A full 1,000-frame episode pays 30, enough to prefer a brisk landing, small
enough never to prefer a crash (−100) over flying on. It should still land.

**`hover`** — the deliberate mistake: **`+1` for every frame the rocket is
upright and slow**. This sounds like precisely what you want. But an episode is
cut off at 1,000 frames, so a rocket that simply *hangs in the air* collects up
to 1,000 points from this rule while paying at most ~300 in fuel — and landing
*ends* the episode, stopping the payments. Hovering beats landing by a wide
margin, and PPO will find that out.

**`hover-fixed`** — the same intent, paid correctly:

1. **Pay for improvement, not position.** Define `φ(s) = −k·(|tilt| + speed)` and
   pay `γ·φ(s′) − φ(s)` — the *change*, the way the built-in reward already works.
   A closed loop (drift off, come back) nets to zero, so there is nothing to farm
   by waiting. `φ` is forced to `0` at terminal states, which is what makes this
   [potential-based shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
   in Ng, Harada & Russell's sense: it provably cannot change which policy is
   optimal, only how quickly it is found.
2. **Charge for time**, so waiting costs something regardless.

## Results

### Does PPO land at all?

1,000,000 frames each, 20 evaluation episodes on the **raw** environment,
deterministic actions, seed 0. 200+ counts as solved.

| Run | Reward | True score | Outcomes | Solved |
|---|---|---:|---|:--:|
| `baseline` | built-in only | **259.3** ± 20.0 | 20/20 landed | yes |
| `timefee` | built-in − 0.03/frame | **244.3** ± 58.2 | 19 landed, 1 crashed | yes |
| `hover-fixed` | potential-based + fee | **270.3** ± 19.4 | 20/20 landed | yes |

So the control works, a reward term of my own is harmless, and the
potential-based version is, if anything, slightly better — the shaping does
some of the work the agent would otherwise have to discover.

### Breaking it took more than the obvious bonus

The first attempt at the hack — `+1` per frame for `|tilt| < 0.1` and
`speed < 0.1` — **did not work**. It scored 245.6 and landed 20/20.

The incentive was there: 1,000 frames × (1.0 − 0.30 fuel) = **+700**, against
roughly +250 for landing. PPO simply never left the landing optimum. The
training log looked alarming anyway (shaped return ~850 against a true ~150),
but that gap turned out to be mostly an artifact: under stochastic rollouts the
action noise keeps the lander awake so episodes never terminate, and the
**baseline** policy timed out 17/20 the same way. A gap in the log is a hint,
not a verdict.

Making it bite needed a steeper bonus and a tolerance loose enough to stumble
into (`hack_search.sh`):

| Bonus | Tolerance | True score | Outcomes |
|---|---|---:|---|
| +1 | tilt 0.1, speed 0.1 | 245.6 | 20/20 landed |
| +3 | tilt 0.1, speed 0.1 | 184.8 | 16 landed, 2 crashed, 2 timeout |
| +1 | tilt 0.2, speed 0.3 | 114.6 | 15 landed, 5 never settled |
| **+3** | **tilt 0.2, speed 0.3** | **−102.0** | **0 landed, 20/20 timeout** |

The last row is the result. It is paid **+684** and scores **−102** — worse
than doing nothing — and the diagnostic in `evaluate.py` says exactly what it
is doing: at the 1,000-frame cutoff it sits at height **0.94** with **0/2 legs
down** and speed 0.05. Airborne, level, going nowhere, for all twenty episodes.

There are two distinct ways to never finish, and they are worth separating.
At the loose tolerance with +1 the agent *landed* but kept firing so the
lander never fell asleep and the episode never ended. At +3 it stopped
pretending and just hovered.

### One run proves nothing, so here are three

Training is stochastic, so neither the hack nor the fix gets claimed off a
single seed (`confirm.sh`):

| Run | Seed | True score | Outcomes | At the cutoff |
|---|:--:|---:|---|---|
| `hover` +3 loose | 0 | −102.0 | 20/20 timeout | height 0.94, 0/2 legs |
| `hover` +3 loose | 1 | −84.6 | 20/20 timeout | height 0.86, 0/2 legs |
| `hover` +3 loose | 2 | −99.5 | 20/20 timeout | height 1.02, 0/2 legs |

Sixty episodes across three seeds, not one landing between them, hovering
within a few centimetres of the same altitude each time.

The control is just as stable: `baseline` scores **259.3** on seed 0 and
**258.5** on seed 1, landing 20/20 both times.

And the fix holds at the strength that broke the naive version — `hover-fixed`
with `k=60`, an aggressive shaping term, still scores **272.5 ± 20.1** and
lands 20/20. It is the potential-based *form* that cures it, not a smaller
number: that is the whole claim, and it survives being pushed.

![paid versus flown](logs/curves.png)

![what each policy does with its time](logs/contact_sheet.png)

### Watch it rather than read it

[`ui/index.html`](ui/index.html) is a scroll-driven walkthrough: scrolling
scrubs the actual flight frame by frame, through the descent and then through
the hacked run, where the two score counters pull apart as you go. The craft is
a shaded 3D model ([`ui/craft3d.js`](ui/craft3d.js)) — a painter's-algorithm
renderer with no dependency — but the physics underneath is unchanged 2D Box2D,
and the model's roll and leg hinges are driven by the real bodies.

The model is built at three levels of detail and cached, with only the legs
rebuilt per frame, so the quilted insulation, panel plates, propellant tanks,
RCS clusters, handrails and ladder cost nothing on a small canvas: a craft 30
pixels across gets the silhouette, one filling the screen gets everything.
Measured at 75 fps with a worst frame of 14.4 ms.

Below the scenes it has two modes.

**Watch the agents** replays real exported episodes — Box2D's own output, not
an approximation — with the eight observation numbers live, both scores
counting up side by side, and the divergence chart. On the hacked run the paid
counter reaches **2,799** while the true score ends at **−87**.

**Fly it yourself** hands you the same lander. Arrow keys fire the engines, one
per frame, which is the same single choice the agent gets. You pick which
reward function pays you, so you can try to farm the broken one by hovering and
watch your own paid score pull away from your true one.

### Is the browser version really the same game?

[`ui/sim.js`](ui/sim.js) re-runs the actual physics through
[planck.js](https://piqnt.com/planck.js/), a direct Box2D port, with the same
shapes, densities, joints, engine impulses, solver iteration counts and reward
function as `lunar_lander.py`. Checked against the Python environment:

| Check | Python | Browser |
|---|---:|---:|
| Lander mass / inertia | 4.81667 / 0.83331 | 4.81667 / 0.83331 |
| Coast all the way down (20 eps) | −143.5, 71.2 frames | −139.6, 72.5 frames |
| Left engine held (12 eps) | −596.8, 68.7 frames | −604.7, 69.3 frames |
| Main engine held (40 eps) | −876.8 ± 647 | −813.5 ± 537 |
| Δvx per frame, side engine | ∓0.0507 / ±0.0479 | ∓0.0497 / ±0.0512 |

Every case crashed in both. The held-main-engine number has a standard
deviation of ~600, so the gap between the two means is well inside noise — at
n=12 it had looked like a real discrepancy, which is its own small lesson about
reading one number. A score you earn by hand is therefore directly comparable
to the agents' scores, which is why the end-of-flight card shows both.


## Hosting it

The page in `ui/` is written for an Artifact host, which wraps it in a document
skeleton at publish time. Anywhere else it needs that skeleton itself, or the
browser drops into quirks mode with no charset. `build_site.py` adds it:

```bash
python build_site.py --zip
```

That writes `docs/` (776 KB, four files and a `.nojekyll`) and
`lunar-lander-site.zip` with `index.html` at the archive root. It is a plain
static site with no build step — nothing to install, nothing to run server-side.

- **Netlify Drop** — drag the zip onto [app.netlify.com/drop](https://app.netlify.com/drop).
  No account needed; sign in afterwards to keep the URL.
- **GitHub Pages** — push the repo, then Settings → Pages → `main` / `docs`.
- **Vercel / Cloudflare Pages** — point the publish directory at `docs`, no build command.

To update a deployed copy, re-run the command above and drop the new zip.

## Pitfalls I hit (and the ones I didn't)

- **`LunarLander-v2` doesn't exist any more.** Current Gymnasium is `v3`.
  Most tutorials you'll find are a version behind.
- **Box2D / swig.** Older guides tell you to `brew install swig` first. As of
  `box2d==2.3.10` there are arm64 wheels and the install just works; the swig
  dance is only needed if pip falls back to compiling.
- **Python too new.** PyTorch had no wheels for the 3.14 on this machine. Pinning
  the venv to 3.12 fixed it — worth checking before blaming the install.
- **Watch the clip, not the score.** The whole point of the `hover` run: its
  shaped score climbs beautifully. You have to look at the rocket to see the
  problem — or at the outcome breakdown, where it never lands.
- **One run proves nothing.** Training is stochastic. `./run_experiments.sh seeds`
  re-runs `baseline` and `hover` on seeds 1 and 2 before any claim is made.
