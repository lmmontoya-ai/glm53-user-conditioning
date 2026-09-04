from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glm53_user_eval.v11.administrative_closure import close_v11


def main() -> None:
    audit_root = ROOT / "artifacts/glm53_user_eval/v11/offline_audit"
    decision, evidence = close_v11(
        repo_root=ROOT,
        amendment_path=ROOT
        / "pipelines/glm53_user_eval/v11/configs/amendment_v11_administrative_closure_v1.yaml",
        diagnostic_path=ROOT
        / "pipelines/glm53_user_eval/v11/status/ai_diagnostic_review/reported_summary.json",
        decision_path=audit_root / "decision.json",
        evidence_path=audit_root / "administrative_closure_evidence.json",
    )
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "decision_sha256": evidence["decision_sha256"],
                "evidence_path": (
                    "artifacts/glm53_user_eval/v11/offline_audit/"
                    "administrative_closure_evidence.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
