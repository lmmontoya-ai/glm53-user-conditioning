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

FINAL_JUDGE_FIG1

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

FINAL_JUDGE_FIG2

## Figure 3: `fig3_decomposition`

Three panels on the confirmatory run with a shared axis: the four-group interaction in the yes-rate,
in folded confidence, and in confidence after standardizing for the answer, plus the same-answer
matched estimate as a grey hollow marker. The lesson: the answer itself did not measurably move, but
its interval is wide enough to contain a shift the size of the confidence effect, so the answer
shift is undetected rather than excluded. The confidence shift is essentially unchanged after
adjusting for the answer, so the effect lives in how sure the model says it is, not in what it says
it would do.

FINAL_JUDGE_FIG3
