# GLM-5.3 Transluce interaction confirmation v7

This branch contains a fresh, full-population confirmation of the exploratory v6 AI-specific fame interaction. The v6 run is discovery evidence only. No v7 score aggregation is permitted until the complete technical and manual audits pass.

Primary estimand:

`(Famous-AI - Unknown-AI) - (Famous-non-AI - GenPop)`

The scientific generation contract is mechanically identical to v6. Only run identity, output paths, analysis, and decision logic differ.

## Final result

- Status: `confirmed_target_sized_interaction`
- Interaction: `-0.65024890176303` percentage points
- Primary 95% crossed-bootstrap interval: `[-1.2099960877594156, -0.09734531995242601]`
- Independent verifier interval: `[-1.215275412547941, -0.08507385889218903]`
- White-box green light: `true`
- Scientific rows: `34,400` (`34,338` valid; `99.8198%`)
- API cost: `$28.93733862`

Both fixed dilemma splits were negative, every leave-one-out interaction remained negative,
and the largest leave-one-out shift was `0.03883` pp. The matched public-versus-constructed
address estimate was `0.02267` pp with a 95% interval of `[-0.38127, 0.42555]`.

The 40-row score-blind transcript review was completed by the Codex coding research agent;
human researcher review remains explicitly pending. The preregistered non-gating
statsmodels robustness fit was unavailable in the pinned Transluce environment and did not
affect the primary bootstrap, independent verifier, decision, or resource gate.

Compact committed evidence lives under
`artifacts/glm53_user_eval/reports/transluce_interaction_v7/`. The 9 MB
`raw_scores.jsonl` and raw Inspect logs remain local, immutable, and SHA-256-covered by
`final_evidence.json`, following the preregistered compact-evidence policy.
