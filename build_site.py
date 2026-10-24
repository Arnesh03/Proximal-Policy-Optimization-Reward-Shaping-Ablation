"""Build the deployable site.

The page in ui/index.html is written for the Artifact host, which wraps it in a
document skeleton at publish time. Anywhere else it needs that skeleton itself,
or the browser drops into quirks mode with no charset: a different scrolling
element, and mojibake wherever the server omits a charset header.

    python build_site.py            # -> docs/ (deploy) and ui/_local.html (dev)

docs/ is a plain static folder. It works as-is on GitHub Pages (Settings ->
Pages -> main / docs), Netlify, Vercel or Cloudflare Pages with "docs" as the
publish directory. Nothing needs a build step on their side.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
UI = ROOT / "ui"
ASSETS = ["sim.js", "craft3d.js", "trajectories.json"]

TITLE = "What the Lander Was Paid"
DESCRIPTION = (
    "Four PPO agents fly the same lunar lander under four reward functions. "
    "Watch what each was paid against how well it actually flew, or fly it "
    "yourself on the same physics."
)

SKELETON = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{description}">
<meta name="color-scheme" content="dark light">
<meta name="theme-color" content="#0d1318">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:type" content="website">
<style>
  :root {{
    color-scheme: light;
    padding-top: env(safe-area-inset-top, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
  }}
  body {{ margin: 0; font-family: system-ui, sans-serif; font-size: 14px; }}
  img {{ max-width: 100%; }}
  [hidden] {{ display: none !important; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def build(out: Path) -> None:
    body = (UI / "index.html").read_text(encoding="utf-8")
    page = SKELETON.format(body=body, title=TITLE, description=DESCRIPTION)

    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    for name in ASSETS:
        shutil.copy2(UI / name, out / name)
    # GitHub Pages runs Jekyll otherwise, which skips files beginning with _
    (out / ".nojekyll").write_text("")

    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"  {out}/  ({total / 1024:.0f} KB)")
    for f in sorted(out.iterdir()):
        if f.is_file():
            print(f"    {f.name:22s} {f.stat().st_size / 1024:7.1f} KB")

    # the dev server serves ui/, so keep a wrapped copy there too
    (UI / "_local.html").write_text(page, encoding="utf-8")


def package(out: Path, zip_path: Path) -> None:
    """Zip the built site with index.html at the archive root.

    That is the shape Netlify Drop and friends expect: drop the zip and it is
    unpacked as the site, no folder level in between.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f.name != ".DS_Store":
                z.write(f, f.name)
    print(f"  {zip_path.name}  ({zip_path.stat().st_size / 1024:.0f} KB) "
          f"\u2014 drag this onto app.netlify.com/drop")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs", help="output directory (default: docs)")
    p.add_argument("--zip", action="store_true",
                   help="also write lunar-lander-site.zip for drag-and-drop hosts")
    args = p.parse_args()
    out = ROOT / args.out
    build(out)
    if args.zip:
        package(out, ROOT / "lunar-lander-site.zip")


if __name__ == "__main__":
    main()
