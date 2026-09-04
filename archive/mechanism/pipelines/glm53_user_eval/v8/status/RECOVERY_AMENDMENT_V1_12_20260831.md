# GLM-5.3 v8 recovery amendment v1.12

RunPod rejected every v1.11 request for four H200 GPUs before creating a Pod.
The requests covered every datacenter that reported H200 stock. No compute was
allocated, the account balance did not change, and no model forward or
scientific row ran.

The user authorized trying four H200s or more. V1.12 permits one two-B300
Secure Cloud Pod. Two B300s provide 576 GB nominal GPU memory, compared with
564 GB for four H200s. The live aggregate rate at amendment time was
$15.78/hour, below the v1.11 four-H200 rate of $18.36/hour.

The Pod remains unmounted and stages the exact public FP8 revision to local
NVMe. The checkpoint, prompts, identities, tasks, codebooks, estimands,
thresholds, and machine gates do not change. M2 still stops the run if the
exact checkpoint, runtime algebra, measured throughput, or budget gate fails.
