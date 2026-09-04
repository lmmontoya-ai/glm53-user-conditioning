# V22 final report

## Outcome

V22 stopped before any new GLM or judge call. The proposed experiment does not have enough power under its locked design.

The decision is `stop_before_scientific_calls_insufficient_power`.

V22 tested the design on the only direct repeat data available. It matched every non-anonymous V6 and V7 result by population, identity, and dilemma. The difference between those two independent calls estimates how much the same cell varies between runs. The calculation removed the realized V6-to-V7 group shifts, then resampled dilemmas and identities using the planned crossed bootstrap.

The smallest meaningful effect was fixed at +0.325 percentage points, which is half of the V7 interaction. The required power was 80%.

## Power results

| Dilemmas | Power at +0.325 pp | Power at +0.500 pp | Power at +0.650 pp | Effect needed for 80% power |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 5.6% | 8.3% | 11.3% | 2.307 pp |
| 50 | 8.6% | 15.1% | 22.7% | 1.473 pp |
| 75 | 11.2% | 21.0% | 32.5% | 1.194 pp |
| 100 | 13.7% | 26.8% | 41.6% | 1.029 pp |

The 100-dilemma design is the strongest allowed design, but it reaches only 13.7% power for the effect V22 was meant to detect. Its empirical null interval is [-0.713, +0.723] pp. Adding a true +0.325 pp effect would usually leave the lower endpoint below zero.

An independent implementation used a different seed and 20,000 draws. It reproduced the stop decision. Its 100-dilemma power estimates were 13.5%, 26.8%, and 41.9% for the three simulated effects.

## What adequate power would cost

With all 100 dilemmas and equal independent repeats in both conditions, the empirical model estimates:

| True attenuation | Repeats per condition and cell | Baseline + deployment rows | Estimated cost |
| ---: | ---: | ---: | ---: |
| +0.325 pp | 11 | 616,000 | $518.18 |
| +0.500 pp | 5 | 280,000 | $235.54 |
| +0.650 pp | 3 | 168,000 | $141.32 |

These figures exclude the neutral-text condition, manipulation checks, retries, and transcript judging. The $518 estimate for the preregistered effect is therefore a lower bound for the complete proposed study.

## Why the experiment stopped

The behavioral interaction is small compared with ordinary run-to-run changes in confidence. Pairing identities and dilemmas helps, but the available 100 dilemmas do not supply enough independent information. A single baseline plus deployment run would cost about $47 and would still miss a real half-attenuation effect roughly 86% of the time.

Running that study would likely produce another wide interval. The user explicitly required an empirical power gate and asked not to accept another imprecise result. V22 followed that rule.

## What was not run

V22 made zero fresh subject-model calls, zero judge calls, and zero manipulation-check calls. It spent $0 on API or cloud compute. The neutral and operational prompt families were drafted and frozen, but no model saw them. The transcript-annotation track was not started because it cannot repair the failed primary power gate and a personal review by Luis cannot be performed by automation.

## Scientific state

V7 remains the strongest result: a fresh confirmatory interaction of -0.650 pp with its 95% interval below zero. V21 remains inconclusive about mechanism. V22 adds no behavioral outcome. It establishes that this particular screening-off experiment, with 70 identities and at most 100 dilemmas, cannot answer the proposed question at the required precision with one call per cell.

