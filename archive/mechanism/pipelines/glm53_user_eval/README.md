# GLM-5.3 user and evaluation awareness

This pipeline tests whether recognized AI-safety users recruit the same GLM-5.3-Flash activation direction as explicit evaluation cues. It starts from Transluce's released famous, affiliation-twin, and same-name general-population conditions.

## Current status: 2026-08-29

G2 passed on the exact official FP8 revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. The source-locked Transformers
runtime verified all 62 weight shards, 45 four-stream mHC decoder layers, prompt-vector
extraction, exact alpha-zero logits and greedy generation, and local additive-hook algebra.

G3 has no scientific rows. Sequential Hugging Face decoding was too slow, native
Transformers tensor parallelism failed in the upstream hybrid-attention TP plan, and the
official vLLM server could not load the 305.79 GiB checkpoint from RunPod's FUSE-mounted
network volume within the bounded serving tests. Streaming loaded one shard in 229.76
seconds, prefetch stalled at 2/62 shards, and eager loading remained at 0/62 shards for
more than five minutes. A custom local-disk stage copied one shard in 438 seconds, and a final
direct Hub-to-local transfer completed 0/72 files after more than eleven minutes. All Pods were
deleted. The 500 GB network volume remains intact.

The machine-readable outcome is in
`status/g3_infrastructure_decision_20260829.json`. A prospective Pod-local staging amendment is
recorded in `status/g3_local_stage_prereg_20260829.json`. Its public, digest-pinned image downloads
the exact Hugging Face revision directly to a 450 GB container disk and verifies all 62 G2 hashes
before starting vLLM. Do not reinterpret G2
as behavioral evidence.

The final decision is `status/g3_serving_final_decision_20260829.json`. The next scientific step
is the previously designed pinned-provider API behavior gate, which requires a separately supplied
OpenRouter or first-party Z.ai credential. White-box work should resume only if that behavioral gate
passes and the checkpoint is available through high-performance storage or a provider-local cache.

Preregistration v4 reinstates that API gate through OpenRouter, restricted to the first-party Z.AI
endpoint with fallbacks disabled and router metadata required. It derives the API schedule from the
frozen 600-row local schedule and changes only the phase, provider label, and API model slug; all
prompts and sample IDs remain fixed. OpenRouter's endpoint metadata does not advertise `seed` or the
provider-specific `clear_thinking` object for Z.AI, so v4 records their omission before any call and
does not treat API outputs as exact-checkpoint outputs. A passing API gate unlocks only a capped
RunPod Serverless cached-model test. A failing gate stops the project without further RunPod spend.

The v4 API gate is now complete. All 600 rows parsed and passed the provider audit, but the primary
famous-name effect was -0.08 percentage points with a 95% interval of [-1.62, 1.36]. Only two of four
pairs had the predicted sign. The affiliation effect was -0.60 points with a 95% interval of
[-2.11, 0.85]. G3 therefore failed, RunPod Serverless was not unlocked, and the GLM-5.3 project stops
at the behavioral rung. See `status/g3_api_final_decision_20260829.json`.

Preregistration v5 preserves that failed four-identity gate and asks a different question. The
effective identity sample in v4 was four, despite its 600 rows. V5 fixes all 70 Transluce roster
indices before calls, uses the untouched `behavior_hardening_25` split for discovery and
`behavior_causal_25` for confirmation, and tests four conditions at every index: famous AI user,
affiliation-matched unknown twin, same-name general-population user, and famous non-AI control.
Each stage has 7,000 rows. The first-party Z.AI subject route, first-party OpenAI extraction route,
prompts, generation settings, and task manifest remain unchanged. Individual identities must pass
BH correction on discovery and a separate task-bootstrap confirmation rule. RunPod remains locked
until a roster, identity-specific, or affiliation result replicates.

The pipeline stops at fixed gates. Preregistration v3 retires the unexecuted API gate. G2 loads and validates the exact official FP8 checkpoint; G3 then measures the behavior on that same local checkpoint before any activation or steering analysis. RunPod model staging and volume expansion remain forbidden until v3 is committed and tagged and the unchanged G0 evidence is revalidated.

Tag `glm53-user-eval-prereg-v3.1` records a hardware-capacity-only amendment after two primary
RTX PRO 6000 allocation attempts failed before Pod creation. It permits four H100 NVLs in the same
datacenter under the existing two-hour and USD 30 G2 ceilings. No model or scientific condition
changed.

Tag `glm53-user-eval-prereg-v3.2` pins the observed RunPod image digest and removes a container
command override that prevented SSH startup. The failed startup completed no scientific calls.

Tag `glm53-user-eval-prereg-v3.3` pins `kernels==0.16.0`, required by the source-locked
Transformers fine-grained FP8 path. The triggering runtime stopped before a successful forward.

Tag `glm53-user-eval-prereg-v3.4` pins Triton 3.5.1 after direct symbol inspection showed that
the downloaded FP8 kernel requires its updated autotuner API. No successful forward preceded it.

Tag `glm53-user-eval-prereg-v3.5` fixes the G2 synthetic hyper-head assertion to compare tensors
on the device selected by Accelerate. The triggering run had completed all 20 diagnostic forwards.

Tags through `glm53-user-eval-prereg-v3.11` record the later tensor-parallel serving and
network-storage amendments. None produced a completed behavior row.

Run deterministic validation with:

```bash
uv run python pipelines/glm53_user_eval/run.py validate-prereg
uv run pytest -q tests/glm53_user_eval
```

The canonical source locks are in `reference/source_locks_glm53_user_eval_v1.json`. The operative preregistration is `prereg_v3.yaml`. It uses the dated `gpt-5.4-mini-2026-03-17` extraction judge only after local subject generation. Generated calls and large model artifacts remain under the ignored `artifacts/` tree.

After a passing G2 decision, the local behavior sequence is:

```bash
uv run python pipelines/glm53_user_eval/run.py behavior-local \
  --schedule-root artifacts/glm53_user_eval/behavior_local/g3_schedule_v3 \
  --model-root /workspace/mats-glm53/models/GLM-5.3-Flash \
  --g2-decision artifacts/glm53_user_eval/runtime/g2/decision.json \
  --output artifacts/glm53_user_eval/behavior_local/g3_subject

uv run python pipelines/glm53_user_eval/run.py judge-local-behavior \
  --schedule-root artifacts/glm53_user_eval/behavior_local/g3_schedule_v3 \
  --subject-root artifacts/glm53_user_eval/behavior_local/g3_subject \
  --output artifacts/glm53_user_eval/behavior_local/g3_judged

uv run python pipelines/glm53_user_eval/run.py analyze-local-behavior \
  --results artifacts/glm53_user_eval/behavior_local/g3_judged/results.jsonl \
  --schedule artifacts/glm53_user_eval/behavior_local/g3_schedule_v3/schedule.jsonl \
  --reading-log artifacts/glm53_user_eval/behavior_local/reading_log.csv \
  --output artifacts/glm53_user_eval/behavior_local/g3_analysis
```

Subject generations are written atomically per sample and can be resumed. Judging is a separate
step so the paid GPU Pod can be deleted before the first-party extraction calls run.
