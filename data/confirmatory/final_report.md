# GLM-5.3 Transluce interaction confirmation v7

## Result

The fresh full-population run produced an AI-specific fame interaction of **-0.6502 percentage points** (95% crossed-bootstrap interval **[-1.2100, -0.0973]**). The machine decision is **`confirmed_target_sized_interaction`**. The white-box resource gate is **true**.

The interaction is `(Famous-AI - Unknown-AI) - (Famous-non-AI - GenPop)`, using person-weighted, dilemma-centered folded confidence. The v6 result generated this hypothesis; only fresh v7 calls determine confirmation.

## Execution

- Model: `z-ai/glm-5.3-flash`
- Provider: Novita, FP8, fallbacks disabled
- Reasoning: high
- Population: five complete Transluce groups across all 100 dilemmas
- Scientific rows: 34,400
- Valid scores: 34,338
- Primary bootstrap: 20,000 crossed person/dilemma draws
- Independent verification: passed

## Components

- Famous-AI minus Unknown-AI: -0.3607 pp
- Famous-non-AI minus GenPop: 0.2895 pp
- Unknown-AI minus GenPop: -0.2660 pp
- Matched public minus constructed address: 0.0227 pp

## Limitation

This population contrast is not a randomized 2×2 fame-by-AI experiment. It tests whether the four released identity populations differ in the preregistered pattern; it does not by itself establish evaluation awareness or a causal internal mechanism.
