# Implementation note 001 — V7-compatible dilemma centering

After the first deterministic calculation, the reproduced folded-confidence interaction was −0.6548 pp rather than the locked V7 value of −0.65025 pp. Inspection showed that V23 had calculated the algebraically equivalent uncentered interaction, while V7 subtracts a dilemma-specific general-population center before equal identity weighting. The two are exactly equivalent only with complete cells; V7 has sparse missing rows.

V23 was corrected to use the literal V7 centering procedure for every outcome. This is an estimator-fidelity correction, not an outcome-dependent analysis choice. The correction was made while transcript judgments were still running and before their group statistics were available. Independent verification implements the centered estimator separately.

The reproduction also preserves V7's outcome-specific missingness: confidence summaries use every valid confidence score, choice summaries use every valid first-turn choice, and conditional analyses require both. The first draft had unnecessarily required both fields even for the unconditional confidence reproduction.
