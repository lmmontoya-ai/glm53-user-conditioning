"""Stage 8: judge rounds for each figure with a Claude model and a crop tool.

For each figure: up to three rounds. Each round logs the judge output, what changed, and what
was rejected to outputs/judge/<figure>/round_<n>.json. Rounds stop early when the judge
returns no change requests. Number-consistency failures are treated as bugs.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, print_plan, rel, stage_parser  # noqa: E402

from glm53.io import load_yaml, write_json  # noqa: E402
from glm53.judge_plots import judge  # noqa: E402
from glm53.plots import FIGURES, apply_style  # noqa: E402

STAGE = "judge"


def render(cfg, key: str, fig_dir: Path, data_dir: Path) -> tuple[Path, Path, str]:
    apply_style(cfg["style"])
    paths, data, caption = FIGURES[key](cfg, fig_dir)
    data_path = data_dir / f"{key}_data.json"
    write_json(data_path, data)
    png = next(p for p in paths if p.suffix == ".png")
    return png, data_path, caption


def main() -> int:
    parser = stage_parser(STAGE, __doc__)
    parser.add_argument("--figures", nargs="*", default=list(FIGURES), choices=list(FIGURES))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--backend", choices=("api", "cli"))
    parser.add_argument("--model", default=None, help="claude-opus-5 for the api backend, opus for the cli backend")
    parser.add_argument("--round-start", type=int, default=1, help="continue numbering from this round (after manual fixes)")
    args = parser.parse_args()
    cfg = load_yaml("plots.yaml")
    fig_dir = REPO_ROOT / cfg["output"]["dir"]
    backend = args.backend or ("api" if os.environ.get("ANTHROPIC_API_KEY") else "cli")
    model = args.model or ("claude-opus-5" if backend == "api" else "opus")
    if args.dry_run:
        print_plan(
            STAGE,
            [fig_dir / f"{cfg[k.replace('fig', 'figure')]['file']}.png" for k in args.figures],
            [args.output_root / k / f"round_{n}.json" for k in args.figures for n in range(args.round_start, args.round_start + args.rounds)],
            {"api_calls": f"up to {len(args.figures) * args.rounds} judge conversations (each may call crop several times)", "backend": backend, "model": model},
        )
        return 0
    data_dir = REPO_ROOT / "outputs" / "plots"
    for key in args.figures:
        for n in range(args.round_start, args.round_start + args.rounds):
            png, data_path, caption = render(cfg, key, fig_dir, data_dir)
            verdict = judge(png, data_path, backend=backend, model=model, caption=caption)
            record = {
                "figure": key,
                "round": n,
                "judged_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "png": rel(png),
                "data_json": rel(data_path),
                "judge_output": verdict,
                "changes_applied": [],
                "rejected": [],
                "status": "awaiting_changes" if verdict.get("change_requests") else "no_change_requests",
            }
            write_json(args.output_root / key / f"round_{n}.json", record)
            print(f"{key} round {n}: {len(verdict.get('change_requests', []))} change requests; consistent={verdict.get('number_consistency', {}).get('consistent')}")
            print(json.dumps(verdict.get("change_requests", []), indent=2, ensure_ascii=False))
            # Changes are applied by editing configs/plots.yaml or src/glm53/plots.py between rounds;
            # the round record is then completed with `changes_applied` and `rejected` before the
            # next round is started with --round-start. This script stops here to leave that step
            # explicit rather than editing plotting code automatically.
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
