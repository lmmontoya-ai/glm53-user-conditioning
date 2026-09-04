# V9 technical amendment v1.2

The first exact model load stopped before any model forward because the runtime
gate expected 90 forget-gate inverse-scale tensors. That count came from reading
Transformers' compressed missing-key display as if every decoder layer used the
linear-attention forget gate.

The locked GLM-5.3 config has 34 `linear_attention` layers and 11
`deepseek_sparse_attention` layers. Only the 34 linear-attention layers contain
the two forget-gate projections, so the exact loaded runtime correctly exposes
68 inverse-scale tensors. All 68 were finite.

V1.2 derives the complete expected tensor-name set from the locked layer-type
schedule and fails unless the architecture counts, names, count, and finiteness
all match. No scientific prompt, cue mask, feature, optimizer, null, decision
threshold, or budget changed. No v9 model forward or scientific row existed
before this amendment.
