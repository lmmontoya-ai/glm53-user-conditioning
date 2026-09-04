"""Stage 7: the three figures, from stage outputs only, as PNG (300 dpi) and SVG."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, print_plan, rel, stage_parser  # noqa: E402

from glm53.io import load_yaml, write_json  # noqa: E402
from glm53.plots import FIGURES, apply_style  # noqa: E402

STAGE = "plots"
INPUTS = [
    REPO_ROOT / "outputs/estimands/estimands.csv",
    REPO_ROOT / "outputs/reproduce/summary.json",
    REPO_ROOT / "outputs/identities/identities.csv",
    REPO_ROOT / "outputs/identities/summary.json",
    REPO_ROOT / "outputs/roles/merged_coding.csv",
    REPO_ROOT / "outputs/decompose/summary.json",
]


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("--figures", nargs="*", default=list(FIGURES), choices=list(FIGURES))
    parser.add_argument("--figure-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_yaml("plots.yaml")
    fig_dir = args.figure_dir or (REPO_ROOT / cfg["output"]["dir"])
    outputs = [fig_dir / f"{cfg[k.replace('fig', 'figure')]['file']}.{fmt}" for k in args.figures for fmt in cfg["output"]["formats"]]
    if args.dry_run:
        print_plan(STAGE, INPUTS, outputs + [args.output_root / f"{k}_data.json" for k in args.figures], {"api_calls": 0})
        return 0
    apply_style(cfg["style"])
    manifest = {}
    for key in args.figures:
        paths, data, caption = FIGURES[key](cfg, fig_dir)
        write_json(args.output_root / f"{key}_data.json", data)
        manifest[key] = {"files": [rel(p) for p in paths], "caption": caption, "data": rel(args.output_root / f"{key}_data.json")}
        print(f"{key}: {', '.join(rel(p) for p in paths)}")
    write_json(args.output_root / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
