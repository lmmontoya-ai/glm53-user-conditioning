# GLM-5.3 v8 recovery resumed with four H200s

Date: 2026-08-31

The user explicitly authorized one more exact-FP8 recovery using four H200
GPUs. This supersedes the operational stop recorded on 2026-08-30. The stop
record and its evidence remain immutable history.

The scientific contract is unchanged. V1.11 changes only the hardware and
model transport:

- four H200 GPUs in Secure Cloud;
- no attached network volume;
- exact public model revision downloaded to local NVMe with Hugging Face Xet;
- `balanced_low_0` with auto-detected capacity across four devices;
- maximum rate of USD 18.36/hour;
- maximum runtime of 3.25 hours and compute spend of USD 59.67;
- at least USD 15 reserved after compute, plus USD 2 for storage;
- machine-gated stop at the first failed M2 through M8 decision.

No v1.11 model call or paid resource may begin before the v1.11 preregistration
tag exists remotely.
