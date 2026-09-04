# V9 technical amendment v1.3

The exact 448-row extraction completed successfully under v1.2. The first
offline probe command then stopped before fitting a layer because the sequence
fitter attempted to cast the 32 governed neutral-decoy labels from `None`
directly to integers.

Other v9 analysis paths already encode neutral decoys as `-1`, and the
preregistered analysis requires reporting their false-positive rate separately.
V1.3 applies the same encoding in the sequence fitter and adds a fail-closed
check that training and validation labels are binary. It changes no scientific
feature, optimizer, permutation scheme, layer-selection rule, threshold, or
gate. No probe metric, selected layer, permutation result, or token-MIL result
was produced before this correction.
