# V1.14 balanced-low-0 load failure

V1.14 staged and rehashed the exact GLM-5.3-Flash FP8 checkpoint on Pod
`6f8skqxvd5n9xd`. Transformers loaded all 1,872 weight entries, but
`balanced_low_0` reserved GPU 0 and filled GPU 1 during MoE conversion. GPU 1
held 267.12 GiB and the loader failed on a 2.25 GiB allocation, followed by a
4.50 GiB finalization allocation.

No model forward, activation, proxy score, or scientific row ran. The full log
is stored at `status/evidence/recovery_20260831/supervisor_v114.log` with SHA-256
`aa07b5152b0ef87de4308e9f34dee13098ec3bf060ce7c0cb1c9d941c4406045`.

V1.15 changes only the two-GPU placement policy to `balanced` and permits one
load retry on the same Pod. The checkpoint, runtime versions, and scientific
contract remain unchanged.
