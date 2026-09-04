# GLM-5.3 API behavior gate: final result

The frozen 600-row Transluce-style behavior gate completed through OpenRouter with every GLM turn
restricted to first-party Z.AI and every extraction turn restricted to first-party OpenAI. All rows
parsed, all raw call artifacts were finalized, reasoning was observed in every row, and no provider
fallback occurred.

The hypothesized user-awareness effect did not replicate. Recognized AI-safety names changed stated
behavioral confidence by -0.08 percentage points relative to affiliation-matched unknown twins. The
95% two-way cluster-bootstrap interval was [-1.62, 1.36]. Pair-level effects were -0.76, -1.34,
+1.06, and +0.72 points, so only two of four pairs had the predicted sign. The same-name affiliation
contrast was -0.60 points with a 95% interval of [-2.11, 0.85]. Neither contrast met the frozen
1.5-point clean-effect gate.

The route therefore fails G3 before any representation or steering work. RunPod Serverless remains
locked, and the project stops without trying new identities, prompts, or sampling settings. The
OpenRouter subject-and-judge calls cost approximately $0.54.

This result applies to the pinned first-party Z.AI API route. That endpoint did not advertise support
for `seed` or the provider-specific `clear_thinking` object, so preregistration v4 omitted them before
the first successful row and treated the API route as a different serving stack from the official
local FP8 checkpoint. The prior G2 result still shows that white-box extraction and intervention are
technically valid on the exact checkpoint. It does not supply behavioral evidence, and this API null
does not authorize further exact-weight work under the project gate.

Manual qualitative review was not used to complete the gate because three numerical requirements
already failed. It could not change the stop decision. An independent calculation from the immutable
rows reproduced the -0.08 name effect, -0.595 affiliation effect, and two-of-four sign count.
