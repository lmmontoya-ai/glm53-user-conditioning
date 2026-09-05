# V12 Fact-Extracted Semantic Validation

Decision: `fact_extracted_semantic_validation_failed_stop_all_experiments`

The frozen 576-row V11 text bank was not edited. A blinded judge extracted four facts, and deterministic code derived the semantic class. No human-review claim or manual override was used.

Overall factor accuracy: 0.7466
Clean binary derived-label accuracy: 0.3854
Final counterfactual accuracy: 0.3594
Mixed-purpose acceptance: 1.0000
No-status acceptance: 1.0000
Neutral-control acceptance: 0.9375
Evidence-span validity: 0.7577

Independent second-pass rows: 484
Primary/verifier factor disagreements: 188

Only exact-FP8 source extraction is unlocked on a pass. Local parity remains a separate gate before user-context recruitment. CoT transfer and steering remain out of scope.
