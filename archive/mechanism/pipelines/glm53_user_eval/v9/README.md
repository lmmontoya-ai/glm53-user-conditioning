# GLM-5.3 v9: paper-faithful eval-awareness probe

V9 corrects the representation surface used in v8. V8 remains an immutable,
stopped study. V9 masks governed eval/deploy cue spans after rendering the exact
GLM chat template, then trains the prior paper's sequence-linear probe on the
mean of the retained prompt tokens.

The first paid run only extracts 448 prompt examples across all 45 layers. It
also saves all retained prompt-token bags. Probe fitting and the 100-repetition
pair-preserving null run offline. Steering is a separate later gate.

Commands:

```powershell
python pipelines/glm53_user_eval/v9/run.py validate-prereg
python pipelines/glm53_user_eval/v9/run.py validate-tokenizer
python pipelines/glm53_user_eval/v9/run.py plan
```

On the exact-checkpoint GPU host:

```bash
python pipelines/glm53_user_eval/v9/run.py extract \
  --model-path /workspace/mats-glm53/models/GLM-5.3-Flash/04c4e9e95c5da8862dced7e5056455116f83a7e0
```

After the feature bundle is copied locally:

```powershell
python pipelines/glm53_user_eval/v9/run.py analyze-readout
```

