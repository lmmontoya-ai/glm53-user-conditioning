# V21 exploratory Hua continuation

V20 stopped because one component of its local parity gate had the wrong sign.
The overall local interaction, both codebooks, retained magnitude, and output
quality checks passed. After seeing those aggregate results, the user asked to
complete the untouched intervention ladder.

V21 does that without changing V20. It reuses the 1,404 immutable V20 baseline
rows and scores only the five intervention arms plus twenty sign-flip controls.
Its results are exploratory and cannot be described as a successful V20
confirmatory test.

The run completed on 2026-09-03 with the machine decision
`no_resolved_deployment_attenuation`. See [status/FINAL_REPORT.md](status/FINAL_REPORT.md)
for the scientific result, verification, cost, and cleanup record.

Run locally before launch:

```powershell
uv run python pipelines/glm53_user_eval/v21/run.py validate-prereg
uv run python pipelines/glm53_user_eval/v21/run.py plan-paid
uv run pytest -q tests/glm53_user_eval_v21
```
