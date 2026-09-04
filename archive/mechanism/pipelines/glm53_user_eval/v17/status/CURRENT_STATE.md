# V17 current state

V17 is preregistered and has not produced a GLM model forward or scientific row. The offline preflight reconstructed all 6,387 local-parity prompts, froze a 160-prompt pilot and an 800-prompt confirmation set without inspecting outcomes, and prepared a positive control with 32 evaluation and 32 deployment scenarios under two reversed label mappings.

Several paid launch attempts stopped before science while hardening transport and runtime setup. The latest three-H200 attempt, `glm53-v17-hua-20260902T220248Z-044c2a78`, installed the pinned CUDA/PyTorch stack but stopped because the paid host could not clone the public pinned Transformers commit. It did not stage the checkpoint. The pod was deleted, and its terminal marker and bootstrap log were recovered.

Runtime amendment v3 uses exactly four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. At the observed public rate this gives up to 105 minutes below the unchanged $15 hard cap. Infrastructure amendment v7 packages the exact Transformers commit as a hash-bound S3 input, removing GitHub from paid bootstrap. The exact FP8 checkpoint, prompts, directions, analyses, and stopping gates are unchanged.

The first four-GPU allocation under amendment v7 was accepted at $8.36/hour, but its host disappeared before the container started. It emitted no heartbeat, staged no checkpoint, and ran no model forward. Infrastructure amendment v8 permits one retry of the identical topology and transport contract. It does not permit a fallback GPU or a further retry if the second host also fails to start.

The v8 retry briefly entered `running` but also disappeared before emitting its first heartbeat. V17 therefore abandoned server-edition RTX hosts. Runtime amendment v4 and infrastructure amendment v9 authorize one final alternative host topology: four 96 GB RTX PRO 6000 Blackwell Workstation Edition GPUs in `EU-RO-1` or `EUR-IS-1`. This retains 384 GB aggregate VRAM, exact FP8 weights, balanced layer sharding, the $15 cap, and all scientific gates. No further topology amendment is permitted.

The final workstation-GPU request was rejected during Pod creation. V17 is therefore closed as `infrastructure_unavailable_before_hua_positive_control`. Across all attempts, the exact checkpoint was staged and loaded once, but the fixed runtime stopped before its first model call. V17 produced zero scientific rows and no Hua direction, positive-control, local-parity, pilot, or confirmation result. At closure there were zero Pods and zero Serverless endpoints.

All 23 objects on the temporary V17 S3 volume were inventoried and their unique evidence was recovered locally. The 20 GB volume `aehhoaoldv` was then deleted permanently. RunPod reported zero remaining network volumes.
