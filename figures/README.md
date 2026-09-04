# Figures

All three figures are produced by `scripts/07_plots.py` from stage outputs only; no number is
typed into the plotting code. Each was reviewed in up to three rounds by a Claude judge with a crop
tool (`scripts/08_judge_plots.py`); the round logs are in `outputs/judge/<figure>/`. The final
round's verdict is quoted under each figure.

## Figure 1: `fig1_estimands`

Four estimands with 95% crossed bootstrap intervals, discovery and confirmatory runs side by side:
unknown twin minus general population, famous AI minus unknown twin, famous non-AI minus general
population, and their interaction. The lesson: being a famous AI figure pushes stated confidence
down while being famous for something else pushes it up, and the difference between the two,
about two thirds of a percentage point, appeared in both runs with intervals that exclude zero. The
component shifts on their own do not exclude zero in the confirmatory run, and the whole effect is
about six hundredths of the row-level standard deviation, which the caption states.

Final judge round (3 of 3; model opus, 7 crops):

> Lesson stated by the judge: "Fame moves stated confidence in opposite directions for AI figures (down) and non-AI figures (up), and that gap — the interaction, about -0.8 pp in discovery and -0.65 pp in the confirmatory run — held up on fresh data."
>
> Numbers consistent with the stage JSON: True. Intervals visible: True. Overclaims flagged: True (Title clause 'AI fame and non-AI fame shift stated confidence in opposite directions': neither component contrast is separated from zero in the confirmatory run. Famous AI - unknown twin is [-0.81, 0.06] (discovery) and [-0.78, 0.05] (confirmatory); Famous non-AI - general is [-0.11, 0.67] (confirmatory). Only discovery Famous non-AI - general excludes zero. The plotted intervals contradict the directional claim the title asserts.; Title clause 'the difference replicated on fresh data': true only for the interaction row (both CIs exclude zero), but placed after the first clause it invites reading the whole opposite-directions story as replicated. It also hides that the point estimate attenuated ~22% (-0.83 -> -0.65 pp).; Axis range as magnitude emphasis: the panel spans only about -1.5 to +0.9 pp while the response-level SD is 10.8 pp, so a 0.06-SD effect fills the full plot width. Zero is included and marked, so this is not bar-style truncation, but the panel carries no in-plot cue to scale; the 0.06-SD context appears only in the caption.; Estimand selection: the JSON contains the direct famous-AI-vs-general-population contrast (confirmatory -0.63 pp, CI [-1.11, -0.15], excludes zero) and the published-address subset, neither of which is shown, so the reader cannot see the evidence that would actually bear on the title's first clause.). Legibility acceptable: True.
>
> Single change proposed: "Retitle to the claim the intervals actually support - that only the interaction is distinguishable from zero and it replicates (e.g. 'The AI-fame vs non-AI-fame gap replicates: -0.83 pp (discovery), -0.65 pp (confirmatory); each component contrast alone is consistent with zero') - since the current title asserts a directional effect for both components that the plotted CIs contradict."

Applied after this round: 1. Rejected: 4 (title-overclaims-components; in-panel-effect-size-context; show-famous-ai-vs-general; caption-and-layout-sizing (font part)). Details in `outputs/judge/fig1/round_3.json`.

## Figure 2: `fig2_identities`

One point per famous-AI identity: its twin-adjusted effect in the discovery run against the same
quantity in the confirmatory run, with the eight most extreme identities named and a dashed y = x
line. The lesson: per-identity effects are only moderately stable across runs (Spearman rho about
0.4 with an interval reaching nearly to zero), so single-identity rankings should not be read as
facts about those people. The identities with the most negative effects in both runs are mostly
people known for AI safety, evaluation, or public criticism of AI; that observation motivated the
blind role-coding protocol and is not yet tested. Points are uniform in color because the merged
human-and-LLM role coding does not exist yet; the plotting code colors by role and draws role
centroids once `outputs/roles/merged_coding.csv` is complete.

Final judge round (3 of 3; model opus, 4 crops):

> Lesson stated by the judge: "Per-identity effects reproduce only weakly and imprecisely across runs (rho = 0.41, 95% CI [0.01, 0.56]): the discovery-vs-confirmatory scatter is a diffuse cloud around y = x, so a given identity's effect in one run barely predicts its effect in the other."
>
> Numbers consistent with the stage JSON: True. Intervals visible: True. Overclaims flagged: False. Legibility acceptable: True.
>
> Single change proposed: "Resolve the tension between the eight name labels and the caption's own warning: either drop the labels or grey them back to a secondary weight, and in any case reposition them so no label abuts an unlabelled point. Naming the eight most extreme points puts typographic emphasis on exactly the estimates the caption says are 'not reliable identity-level effects' — and the extremes are the ones most inflated by regression to the mean, so they are the least likely to reproduce."

Applied after this round: 3. Rejected: 2 (label-collisions; rerender-after-role-coding). Details in `outputs/judge/fig2/round_3.json`.

## Figure 3: `fig3_decomposition`

Three panels on the confirmatory run with a shared axis: the four-group interaction in the yes-rate,
in folded confidence, and in confidence after standardizing for the answer, plus the same-answer
matched estimate as a grey hollow marker. The lesson: the answer itself did not measurably move, but
its interval is wide enough to contain a shift the size of the confidence effect, so the answer
shift is undetected rather than excluded. The confidence shift is essentially unchanged after
adjusting for the answer, so the effect lives in how sure the model says it is, not in what it says
it would do.

Final judge round (3 of 3; model opus, 3 crops):

> Lesson stated by the judge: "The famous-AI confidence gap (about -0.65 pp) is essentially unchanged after adjusting for the yes/no answer (-0.66 pp), while the yes-rate itself shows no detected shift (-0.08 pp) - though its interval is too wide to rule out an answer shift the size of the confidence effect."
>
> Numbers consistent with the stage JSON: True. Intervals visible: True. Overclaims flagged: True (Title clause 'No shift in the predicted answer was detected' reads as a null result, but panel 1's interval spans -2.04 to +1.84 pp and does not exclude an answer shift as large as the confidence effect (-0.65); the panel annotation and caption say this honestly, so the title is the only element out of step with the data.; Title clause 'the confidence shift survives adjustment' is literally true but understates fragility: the adjusted interval's upper bound is -0.10, barely excluding zero.). Legibility acceptable: True.
>
> Single change proposed: "Rewrite the title's first clause so it states the actual result rather than a null: e.g. 'Confidence shift is unchanged by adjustment for the answer; the yes-rate interval is too wide to rule out an answer shift of the same size.' That moves the caveat already annotated in panel 1 into the line most readers will read alone."

Applied after this round: 3. Rejected: 1 (title-null-framing). Details in `outputs/judge/fig3/round_3.json`.
