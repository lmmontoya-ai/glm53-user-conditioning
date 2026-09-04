"""Crop helper for the CLI judge backend: writes an enlarged region of a PNG and prints its path."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glm53.judge_plots import crop_png  # noqa: E402


def main() -> int:
    png = Path(sys.argv[1])
    x0, y0, x1, y1 = (float(v) for v in sys.argv[2:6])
    scale = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
    n = len(list(png.parent.glob("crop_*.png")))
    out = png.parent / f"crop_{n + 1:02d}.png"
    out.write_bytes(crop_png(png, x0, y0, x1, y1, scale))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
