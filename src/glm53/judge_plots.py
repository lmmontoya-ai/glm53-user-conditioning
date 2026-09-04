"""Figure review rounds by a Claude judge with a crop tool.

Each round sends the figure PNG, the stage JSON the figure was built from, and the rubric.
The judge may call `crop` to enlarge a region. It returns structured JSON. Two backends:
`api` (Anthropic SDK, tool-use loop) and `cli` (local Claude Code CLI, which views the PNG
with its file reader and crops through a restricted shell command).
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


RUBRIC = """You are reviewing a scientific figure for a research repository. Answer each rubric item from the figure image and its title alone first, then check against the attached JSON, which contains every number the figure was built from.

1. lesson: Can a reader state the lesson within ten seconds from the figure and title alone? Quote the lesson they would state.
2. number_consistency: Are all plotted numbers consistent with the attached JSON? List any mismatch (plotted value versus JSON value). Use the crop tool to read small text.
3. intervals_visible: Are confidence intervals visible and not visually suppressed?
4. overclaiming: Does any element overclaim (title, annotation, color emphasis, axis truncation)?
5. legibility: Legibility at the printed size (about 16 cm wide): font sizes, overlapping labels, colorblind-safe palette, contrast.
6. single_change: What single change would most improve the figure?

Return only a JSON object with this shape:
{"lesson": {"can_state_in_ten_seconds": true|false, "quoted_lesson": "..."},
 "number_consistency": {"consistent": true|false, "mismatches": [{"element": "...", "plotted": "...", "json": "..."}]},
 "intervals_visible": {"visible": true|false, "notes": "..."},
 "overclaiming": {"overclaims": true|false, "elements": ["..."]},
 "legibility": {"acceptable": true|false, "issues": ["..."]},
 "single_change": "...",
 "change_requests": [{"id": "short-slug", "description": "...", "severity": "bug|style"}]}
`change_requests` must be empty when the figure needs no change."""

CROP_TOOL = {
    "name": "crop",
    "description": "Return an enlarged crop of the figure. Coordinates are fractions of width and height in [0, 1].",
    "input_schema": {
        "type": "object",
        "properties": {
            "x0": {"type": "number"},
            "y0": {"type": "number"},
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "scale": {"type": "number", "description": "enlargement factor, default 2"},
        },
        "required": ["x0", "y0", "x1", "y1"],
        "additionalProperties": False,
    },
}


def crop_png(png: Path, x0: float, y0: float, x1: float, y1: float, scale: float = 2.0) -> bytes:
    """PNG bytes of the region [x0, x1] by [y0, y1] (fractions), enlarged by `scale`."""
    image = Image.open(png).convert("RGB")
    w, h = image.size
    box = (int(max(0, min(x0, x1)) * w), int(max(0, min(y0, y1)) * h), int(min(1, max(x0, x1)) * w), int(min(1, max(y0, y1)) * h))
    region = image.crop(box)
    region = region.resize((max(1, int(region.width * scale)), max(1, int(region.height * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    region.save(buffer, format="PNG")
    return buffer.getvalue()


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("judge returned no JSON object")
    # Take the first complete JSON object; judges sometimes append prose after it.
    value, _end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("judge JSON is not an object")
    return value


def judge_api(png: Path, data_json: Path, *, model: str, caption: str) -> dict[str, Any]:
    """One judge round through the Anthropic API with a crop tool loop."""
    import anthropic

    client = anthropic.Anthropic()
    image_b64 = base64.standard_b64encode(png.read_bytes()).decode()
    content: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
        {"type": "text", "text": f"Figure file: {png.name}\nCaption under the figure: {caption}\n\nJSON the figure was built from:\n{data_json.read_text(encoding='utf-8')}\n\n{RUBRIC}"},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    crops = 0
    for _ in range(12):
        response = client.messages.create(model=model, max_tokens=8000, tools=[CROP_TOOL], messages=messages)
        if response.stop_reason == "refusal":
            raise RuntimeError("judge refused")
        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return _parse_json(text) | {"_crops": crops, "_backend": "api", "_model": model}
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "crop":
                crops += 1
                args = dict(block.input)
                data = crop_png(png, float(args["x0"]), float(args["y0"]), float(args["x1"]), float(args["y1"]), float(args.get("scale", 2.0)))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(data).decode()}}]})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("judge did not finish within the tool-call budget")


def judge_cli(png: Path, data_json: Path, *, model: str, caption: str) -> dict[str, Any]:
    """One judge round through the local Claude Code CLI with only file reading and the crop command allowed."""
    import shutil

    claude = shutil.which("claude")
    if claude is None:
        raise RuntimeError("claude CLI not found on PATH")
    with tempfile.TemporaryDirectory(prefix="glm53-judge-", ignore_cleanup_errors=True) as work:
        work_path = Path(work)
        local_png = work_path / png.name
        local_png.write_bytes(png.read_bytes())
        (work_path / "figure_data.json").write_text(data_json.read_text(encoding="utf-8"), encoding="utf-8")
        # Standalone crop helper: same crop as the API tool, no repository imports.
        (work_path / "crop.py").write_text(CROP_HELPER, encoding="utf-8")
        prompt = (
            f"Read the figure image at {local_png} with your Read tool. The JSON it was built from is at {work_path / 'figure_data.json'}; read it too.\n"
            f"Caption under the figure: {caption}\n\n"
            f"To enlarge a region run the shell command: python crop.py {png.name} X0 Y0 X1 Y1 [SCALE] (fractions of width and height in [0,1]); it prints the path of a PNG you then read. Use it to check small text.\n\n"
            + RUBRIC
        )
        env = dict(os.environ)
        env["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + env.get("PATH", "")
        command = [claude, "-p", prompt, "--output-format", "json", "--model", model, "--tools", "Read,Bash", "--allowedTools", "Read,Bash(python crop.py:*)", "--add-dir", str(work_path)]
        completed = subprocess.run(command, cwd=work, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, env=env)
        if completed.returncode != 0 or not completed.stdout.strip():
            raise RuntimeError(f"claude exit {completed.returncode}: {completed.stderr[-1200:]}")
        payload = json.loads(completed.stdout)
        crops = len(list(work_path.glob("crop_*.png")))
    return _parse_json(str(payload.get("result", ""))) | {"_crops": crops, "_backend": "cli", "_model": model, "_cost_usd": payload.get("total_cost_usd")}


CROP_HELPER = '''import sys
from pathlib import Path
from PIL import Image
png = Path(sys.argv[1]); x0, y0, x1, y1 = (float(v) for v in sys.argv[2:6]); scale = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
im = Image.open(png).convert("RGB"); w, h = im.size
box = (int(max(0, min(x0, x1)) * w), int(max(0, min(y0, y1)) * h), int(min(1, max(x0, x1)) * w), int(min(1, max(y0, y1)) * h))
r = im.crop(box); r = r.resize((max(1, int(r.width * scale)), max(1, int(r.height * scale))), Image.LANCZOS)
n = len(list(png.parent.glob("crop_*.png"))); out = png.parent / f"crop_{n + 1:02d}.png"; r.save(out); print(out)
'''


def judge(png: Path, data_json: Path, *, backend: str, model: str, caption: str) -> dict[str, Any]:
    if backend == "api":
        return judge_api(png, data_json, model=model, caption=caption)
    return judge_cli(png, data_json, model=model, caption=caption)
