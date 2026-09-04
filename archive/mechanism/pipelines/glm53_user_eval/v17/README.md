# GLM-5.3 V17 Hua causal transfer

V17 tests whether Hua et al.'s published deployment direction changes the locally reproduced four-group user-conditioning interaction. It does not use or repair the V16 linear readout.

The paid run has three gates:

1. Extract the exact 16-pair Hua direction and verify bidirectional control on 64 independent eval/deploy scenarios.
2. Reproduce the V7 interaction with the exact local FP8 checkpoint and frozen one-token confidence measure.
3. Run a paired intervention pilot and, only after a coherent pilot, a disjoint confirmation set.

Every failed gate ends the run. V17 excludes autoregressive generation, early chain-of-thought analysis, model substitution, GPU fallback, and outcome-based tuning.

The preregistration is `configs/prereg_v17_hua_causal.yaml`. Run the offline checks with:

```powershell
uv run python pipelines/glm53_user_eval/v17/run.py prepare
uv run python pipelines/glm53_user_eval/v17/run.py validate-prereg
uv run python pipelines/glm53_user_eval/v17/run.py plan-paid
uv run pytest -q tests/glm53_user_eval_v17
```
