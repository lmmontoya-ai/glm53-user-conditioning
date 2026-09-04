# V19 lean Hua causal test

V19 runs the smallest causal experiment that can support the application claim.
It uses the exact official FP8 GLM-5.3-Flash checkpoint on exactly two B300s and
scores one prompt at a time because larger batches are not numerically
equivalent on this runtime.

The frozen user design contains all 16 outcome-independent identity sets and
the first 22 of 25 outcome-independent dilemmas in their pre-existing hash
order. The five conditions are baseline, Hua
deployment steering, reverse Hua steering, status-only deployment steering,
and a formality control. V19 makes no random-direction or prompting-superiority
claim.
