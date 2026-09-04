# V22 deployment-information screening

V22 asks whether explicit ordinary-use information reduces the confirmed Famous-AI confidence interaction. It starts with a mandatory power gate based on matched V6 and V7 cells. No fresh model or judge call is allowed unless a candidate design reaches 80% power for a +0.325 percentage-point change.

Run the local gate with:

```powershell
uv run python pipelines/glm53_user_eval/v22/run.py validate-prereg
uv run python pipelines/glm53_user_eval/v22/run.py power
uv run python pipelines/glm53_user_eval/v22/run.py decide
```

The power calculation removes observed group-level drift between V6 and V7, then resamples shared dilemmas, paired Famous-AI/Unknown-AI identities, and independent control identities. It uses the 97.5th percentile of that empirical null as the positive-effect threshold.

If the gate fails, V22 ends before prompt execution. The condition blocks remain a prospective design record, not evidence from a completed experiment.

