"""Draw the learning curves from one or more runs' progress.csv.

    python plot_curves.py baseline timefee hover hover-fixed --out logs/curves.png

For each run it plots the true score (solid) and, when a wrapper was used, the
shaped score PPO was chasing (dashed). The two lines pulling apart is the
picture of reward hacking.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRUE = "rollout/ep_rew_mean"
SHAPED = "rollout/shaped_ep_rew_mean"
STEPS = "time/total_timesteps"


def read_series(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Pull (steps, value) pairs for the columns we care about."""
    series: dict[str, list[tuple[float, float]]] = {TRUE: [], SHAPED: []}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                step = float(row[STEPS])
            except (KeyError, ValueError):
                continue
            for col in (TRUE, SHAPED):
                raw = row.get(col, "")
                if raw not in ("", None):
                    series[col].append((step, float(raw)))
    return series


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run names under logs/")
    p.add_argument("--out", default="logs/curves.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = plt.cm.tab10.colors

    for i, run in enumerate(args.runs):
        csv_path = Path("logs") / run / "progress.csv"
        if not csv_path.exists():
            print(f"  skipping {run}: no {csv_path}")
            continue
        series = read_series(csv_path)
        color = colors[i % len(colors)]

        if series[TRUE]:
            xs, ys = zip(*series[TRUE])
            ax.plot(xs, ys, color=color, lw=2, label=f"{run} — true score")
        if series[SHAPED]:
            xs, ys = zip(*series[SHAPED])
            ax.plot(xs, ys, color=color, lw=1.4, ls="--", alpha=0.8,
                    label=f"{run} — shaped score")

    ax.axhline(200, color="0.35", lw=1, ls=":")
    ax.annotate("solved (200)", xy=(0.01, 200), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=9, color="0.35")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("training frames")
    ax.set_ylabel("mean return over last 100 episodes")
    ax.set_title("Lunar Lander: what PPO was paid vs. how well it actually flew")
    # Outside the axes: the shaped curves run high and the legend would
    # otherwise sit on top of them.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
