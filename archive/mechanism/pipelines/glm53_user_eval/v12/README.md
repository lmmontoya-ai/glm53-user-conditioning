# V12 Fact-Extracted Semantic Validation

V12 validates the unchanged 576-row V11 source bank by extracting four facts from each scenario and deriving the class in deterministic code. It does not treat AI review as human evidence and does not permit manual override.

The sequence is:

```powershell
uv run python pipelines/glm53_user_eval/v12/run.py validate-prereg
uv run python pipelines/glm53_user_eval/v12/run.py plan
uv run python pipelines/glm53_user_eval/v12/run.py smoke
uv run python pipelines/glm53_user_eval/v12/run.py run-primary
uv run python pipelines/glm53_user_eval/v12/run.py analyze-primary
uv run python pipelines/glm53_user_eval/v12/run.py plan-verifier
uv run python pipelines/glm53_user_eval/v12/run.py run-verifier
uv run python pipelines/glm53_user_eval/v12/run.py analyze-verifier
uv run python pipelines/glm53_user_eval/v12/run.py verify
uv run python pipelines/glm53_user_eval/v12/run.py decide
uv run python pipelines/glm53_user_eval/v12/run.py build-evidence
```

`smoke`, `run-primary`, and `run-verifier` require the annotated preregistration tag to point to the current clean commit. Runs are atomic and resumable. `--max-new-rows` supports bounded execution without exposing interim scientific aggregates.

Only exact-FP8 source extraction is unlocked by a pass. Local parity remains a separate gate before prompt recruitment. CoT transfer and steering remain locked by default.
