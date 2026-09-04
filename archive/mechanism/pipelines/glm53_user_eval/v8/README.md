# GLM-5.3 user/evaluation mechanism study v8

This project tests whether the confirmed v7 four-population confidence interaction recruits an independently trained evaluation/deployment readout. It uses the official FP8 GLM-5.3-Flash checkpoint and a one-token A-K confidence proxy. It does not reuse the API confidence response as model input.

Every command checks the preceding machine decision. The study stops at the first failed load-bearing gate. A readable direction is not treated as causal evidence, and successful steering would establish dependence only under the tested intervention.

The task partition uses `mechanistic_anchor_manifest_v1.json`. The older `behavior_causal_25` contains dilemmas outside v7's first 100, so those tasks have no immutable v7 first-turn transcript. The mechanism manifest was selected before v7 from hashes and is fully covered by the parent logs.

Rows whose immutable v7 call contains an empty first assistant turn are retained
and marked proxy-ineligible. They are never reconstructed or rerun. M1 fails if
their rate exceeds the preregistered 0.5% ceiling.

Preregistration amendment v1.19 preserves the v1.4 audit enforcement and every
scientific choice. Every four-H200 allocation request failed before RunPod
created a Pod. V1.12 then allocated two B300s, but an erroneous `sleep
infinity` Docker argument replaced the image's startup command and prevented
SSH from starting. That Pod was deleted before model work. V1.13 permits one
same-topology retry with the Docker argument removed. Two B300s provide 576 GB
nominal VRAM. The Pod has no network-volume mount. It downloads the exact
public revision to local NVMe, verifies all 62 shards, and uses
`balanced_low_0` with auto-detected capacity.
V1.13 bootstrap then failed closed when `uv run` re-synchronized the pinned
Transformers overlay back to the project lock. V1.14 calls the prepared
virtualenv interpreter directly after the overlay install. GPU commands must
not use `uv run` under this runtime.
V1.14 then showed that `balanced_low_0` is unsuitable for two GPUs because it
filled GPU 1 while reserving GPU 0. V1.15 uses `balanced` for one same-Pod load
retry. The exact checkpoint then loaded, but the first calibration forward
failed before logits or activations because CUDA 12.8 ptxas does not recognize
the B300 `sm_103a` target. V1.16 pins PyTorch 2.13.0 with CUDA 13.0 and its
resolved Triton 3.7.1 dependency. Its kernel smoke passed, but model imports
then found the older project `torchvision` incompatible with PyTorch 2.13.
V1.17 pins the matching official `torchvision 0.28.0` CUDA 13 build and checks
the Transformers processor import before model loading. That check then found
the old `torchaudio` binary. Because no stable torchaudio 2.13 wheel exists,
V1.19 removes that unused optional audio package and validates both processor
and model imports before loading. The active text-only runtime remains PyTorch
2.13.0 and torchvision 0.28.0 on CUDA 13. No scientific row or completed M2
report existed before this change.
OOM benchmark
candidates fail individually so a smaller valid batch can still be selected.
V1.4 closed the offline audit findings before any v8
model forward. M0 now verifies the exact bytes of the governed eval/deploy
dataset. M2 through M8 decisions name and hash every input. Pilot alpha
selection fails when no deployment-directed value passes the measured positive
control and probability-mass rules. The causal candidate is frozen before M5 as
either the shared eval direction or a held-out-valid name/affiliation direction.

The GPU supervisor cannot issue a final passing M8 decision. It stops at
`M8_manual_audit_pending` after hardening and independent recomputation. A human
must review the seeded packet, after which `finalize-m8` verifies the exact
review IDs and all pending input hashes. `build-evidence` records honest terminal
failures as well as a completed M8 result.

The active ordinary-Pod launch and bootstrap paths require tag
`glm53-user-eval-v8-preregistered-v1.19`. Retired Serverless files retain their
v1.7 tag as immutable infrastructure history and must not be used for v1.8.
