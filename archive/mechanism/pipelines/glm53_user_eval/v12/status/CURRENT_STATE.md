# V12 Final State

V12 is terminal with decision `fact_extracted_semantic_validation_failed_stop_all_experiments`.

The 576 V11 prompts, labels, factors, splits, generators, tasks, and hashes were unchanged. The primary OpenAI-routed judge saw only each scenario and extracted four facts. Deterministic code derived the class. No human-review claim, AI-review substitution, or manual override was used.

Primary results:

- 576/576 rows completed on the pinned OpenAI route.
- Overall factor accuracy: 74.66%.
- Clean binary derived-label accuracy: 38.54%.
- Final-counterfactual accuracy: 35.94%.
- Mixed-purpose acceptance: 100%.
- No-status acceptance: 100%.
- Neutral-control acceptance: 93.75%.
- Exact evidence-span validity: 75.77%.
- Primary API cost: $1.232181.

The errors were concentrated in resolving the binary scenarios' opaque actor/outcome relations. Among the 480 binary rows, the judge derived 233 `ambiguous`, 233 `deploy`, 12 `eval`, and 2 `no_status` outcomes. False positives dominated the two operational factors: 182 for `real_external_recipient` and 224 for `operational_consequence` among the 512 factor-scored rows. The judge often treated entities mentioned in the reference mapping as active recipients or consequences even when the routing statements assigned the reply elsewhere.

The independently worded second pass covered all 420 primary mismatches plus 64 deterministic matches. Its factor accuracy was 75.28%, with 188 primary/verifier disagreements. The import-isolated verifier reproduced the primary decision exactly.

No GLM-5.3 forward or RunPod compute occurred. Exact-FP8 source extraction, local parity, recruitment, CoT transfer, and steering remain unauthorized. Under the preregistered stop rule, no V13 dataset or validator redesign is permitted before the application deadline.

Final infrastructure cleanup is complete. RunPod volume `a9diryunoj` was deleted after the scientific evidence and raw API bundles were committed and pushed. RunPod now reports zero Pods, zero Serverless endpoints, zero network volumes, and $0/hour spend. The deleted volume is not recoverable. Credential rotation remains a user action.
