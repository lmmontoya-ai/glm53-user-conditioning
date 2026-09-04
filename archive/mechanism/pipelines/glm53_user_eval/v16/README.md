# GLM-5.3 V16

V16 tests three questions in order on the exact official FP8 checkpoint.

1. Can a frozen linear readout distinguish latent evaluation from deployment on the V15 text bank, including its untouched controls?
2. Does the local checkpoint reproduce the V7 user effect under the frozen one-token confidence test?
3. If both checks pass, do famous AI identities shift the frozen readout more than AI affiliation and ordinary fame predict?

The 64 fresh V15 controls are stored separately from development features. Their expected classes are not available to fitting or layer selection. V16 stops after the first failed machine gate. It does not generate chain of thought and it does not steer activations.

Run the weight-free preparation with:

```powershell
uv run python pipelines/glm53_user_eval/v16/run.py audit-tokenizer
uv run python pipelines/glm53_user_eval/v16/run.py validate-prereg
uv run python pipelines/glm53_user_eval/v16/run.py plan-paid
```

The paid command is available only from the preregistered clean commit on a two-B300 Secure Cloud Pod:

```bash
uv run python pipelines/glm53_user_eval/v16/run.py paid-supervisor \
  --model-path /runpod-local/GLM-5.3-Flash/04c4e9e95c5da8862dced7e5056455116f83a7e0 \
  --confirm-spend
```
