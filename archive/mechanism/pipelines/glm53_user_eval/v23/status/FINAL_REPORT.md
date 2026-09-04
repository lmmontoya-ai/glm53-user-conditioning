# GLM-5.3 V23 — Decision versus self-calibration

V23 completed the offline decomposition without new GLM calls or RunPod spend.

## Deterministic results

- First-turn choice interaction: **-0.082 percentage points**, 95% interval **[-2.042, 1.842]**.
- Original folded-confidence interaction: **-0.650 pp**, 95% interval **[-1.207, -0.087]**.
- Choice-standardized folded-confidence interaction: **-0.659 pp**, 95% interval **[-1.221, -0.096]**.

Machine interpretation: `confidence_shift_persists_choice_effect_not_detected_but_not_excluded`.

The first-turn choice estimate is close to zero, but its interval is too wide to establish practical equivalence under the frozen ±1 percentage-point margin. The confidence interaction remains negative after standardizing for first-turn choice, so the data support an identity-conditioned self-calibration effect more directly than a decision effect; they do not rule out a smaller decision effect.

## Blinded AI annotations

- Explicit evaluation/scrutiny interaction: **-0.013**, 95% interval **[-0.131, 0.113]**; inter-judge kappa **0.894**.
- Deference-to-expertise interaction: **0.010**, 95% interval **[-0.035, 0.058]**; inter-judge kappa **0.173**.
- The caution/deference composite is exploratory because V6 development showed unreliable caution thresholds. Its interaction is **-0.039**, and combined interpretation is allowed only if both dimensions pass the frozen agreement rule: **False**.

Neither primary transcript-content interaction was resolved after Holm correction. Deference was rare: its raw weighted kappa was **0.173**, and it met the frozen agreement rule only through **96.7%** exact agreement. Two exploratory deterministic measures did differ: the confidence-turn visible response was **12.0 tokens** longer in the four-group interaction (95% interval **[5.3, 18.9]**), and had **0.39** more sentences (95% interval **[0.10, 0.67]**).

These are AI judgments from independent Luna-max and Terra-high runs with fast mode disabled. A 160-row audit packet has been prepared, but Luis has not yet reviewed it. Accordingly, V23 does not authorize an unqualified positive transcript-content claim.

## Famous-non-AI control audit

The two judges agreed on **98.6%** of 70 identity categories. The roster is overwhelmingly athletes (**36**) and entertainers (**30**), with one unresolved category disagreement. The folded-confidence difference from the dilemma-specific general-population center was positive for both large categories: **0.331 pp** for athletes and **0.227 pp** for entertainers. This exploratory audit suggests that the Famous-non-AI component was not carried by only one of those two large categories, but the roster cannot answer whether technical fame is the relevant comparison.

## Scope

V7 is a locked held-out analysis for this decomposition, not a pristine untouched dataset, because small subsets were manually audited during earlier work. Famous-non-AI occupational analyses are exploratory. V23 makes no activation-mechanism or causal-mediation claim.
