# V9 current state

V9 was opened because v8 did not reproduce the prior paper's probe surface.
V8 used unmasked final-prompt-token activations. V9 uses cue-masked prompt means,
retains the prompt-final result as a leakage-prone baseline, saves token bags for
the frozen token-MIL analysis, and separates readout qualification from steering.

No v9 model forward or paid v9 target execution had occurred when this document
was created.

The pinned tokenizer preflight passed all 448 rows: 416 were fully masked and
32 neutral decoys correctly had no cue. There were no partial, ambiguous,
missing, empty, or no-overlap masks.

At the final local preflight, RunPod reported a $58.6749472482 balance and no
active Pods. Two-B300 Secure Cloud capacity was listed as available at
$15.78/hour aggregate. The first paid run is capped at 1.75 hours and $30, with
at least $25 reserved. It may only extract the frozen 448-row feature bundle.
