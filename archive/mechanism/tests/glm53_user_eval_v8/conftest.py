from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def codebooks() -> dict:
    return json.loads(
        (ROOT / "pipelines/glm53_user_eval/v8/configs/proxy_codebooks_v1.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def schedule() -> dict:
    return json.loads(
        (ROOT / "pipelines/glm53_user_eval/v8/configs/causal_schedule_v1.json").read_text(
            encoding="utf-8"
        )
    )
