# V18 two-B300 execution result

V18 successfully allocated exactly two NVIDIA B300 SXM6 AC GPUs and loaded the
official FP8 GLM-5.3-Flash checkpoint. The local copy contained all 62 expected
weight shards and 328,337,455,672 weight bytes. A complete rehash passed. The
runtime found all 68 required FP8 scale tensors, and every tensor was finite.

The no-op intervention check also passed: adding an exactly zero intervention
changed no full-vocabulary logit. This confirms that the exact checkpoint,
software build, multi-GPU loader, and intervention hook were functional.

The run stopped at the next fail-closed check. Scoring four prompts together
changed the allowed next-token logits by as much as 1.25, far above the frozen
0.002 tolerance relative to scoring the same prompts one at a time. Although
the top-token metadata agreed, the numerical discrepancy means batched and
single-row results cannot be treated as equivalent.

The measured batch-of-four rate was 0.442 prompts per second. At that rate, the
full preregistered ladder was projected to require 52,299 seconds, including
the frozen 30% headroom—about 14.5 hours, versus the 110-minute V18 limit.

Therefore V18 produced no Hua direction, no eval/deploy positive-control rows,
no local behavioral-parity rows, and no causal user rows. It supports no claim
about whether Hua-style steering transfers to GLM-5.3.

The terminal state is:

```text
runtime_calibration_failed_before_hua_positive_control
```

The final paid attempt ran for roughly ten minutes. The account balance fell
from $111.27 at that launch to $108.50 after cleanup. All Pods and Serverless
endpoints are gone, the temporary 20 GB evidence volume was deleted after its
hash-verified artifacts were copied locally, and current RunPod spend is zero.

