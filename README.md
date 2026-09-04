# GLM-5.3-Flash user conditioning

Does a language model state its confidence differently depending on who it thinks it is
talking to? This repository holds the data, analysis code, and figures for a study of
GLM-5.3-Flash on the Transluce user-awareness task, where the system prompt names the user and
the model answers a daily dilemma and then reports how confident it is.

## Finding

When the named user is a famous AI figure, the model states slightly lower confidence than it
does for an unknown person at the same AI lab. When the named user is famous for something other
than AI, it states slightly higher confidence than for an ordinary person. The difference between
those two shifts, the interaction, was negative in a discovery run and again in a fresh
confirmatory run whose analysis was preregistered. The effect is small: about two thirds of a
percentage point of stated confidence, or six hundredths of the row-level standard deviation.
The predicted yes/no answer did not measurably change; the confidence shift survives adjusting
for the answer. Two model judges found no difference in explicit evaluation language between the
populations, and the model's confidence-turn replies were slightly longer for famous AI figures.

## Layout

```
configs/        task contract, estimands and seeds, role-coding protocol, positive control, plots
data/           raw scores and manifests for both runs, transcripts (pointers for large files),
                judge outputs, human audit packets, power analysis
src/glm53/      measurement, bootstrap, decomposition, roles, transcripts, runner wrapper, plots
scripts/        one CLI per stage, all with --dry-run
tests/          regression tests pinning the committed numbers
outputs/        generated tables, figures data, judge logs (not tracked)
figures/        final PNG and SVG figures and their README
archive/        the original code tree, unchanged (see archive/mechanism/README.md)
docs/           preregistrations (unchanged) and analysis notes
```

## Setup

```
uv sync
uv run pytest
```

The pinned Transluce checkout is expected at `../reference/transluce-user-awareness` (commit
`d1b9c357`); only stage 6 needs it. Stage 6 also needs `OPENROUTER_API_KEY`. The role coder and the
figure judge use `ANTHROPIC_API_KEY` when set and otherwise fall back to the local Claude Code CLI
with tools disabled.

## Stages

Every script reads `configs/`, writes to `outputs/<stage>/`, and prints its plan with `--dry-run`.

```
uv run python scripts/01_reproduce.py          # recompute every committed number; tests pin them
uv run python scripts/02_estimands.py          # tidy CSV of estimands with intervals, both runs
uv run python scripts/03_decompose.py          # answer versus confidence decomposition
uv run python scripts/04_identities.py         # per-identity effects and cross-run correlation
uv run python scripts/04_identities.py template   # blank human role sheet
uv run python scripts/04_identities.py code-llm   # blind LLM role coding
uv run python scripts/04_identities.py merge      # join sheets, list disagreements
uv run python scripts/04_identities.py contrast   # scrutiny minus business, discovery first
uv run python scripts/05_transcripts.py        # response length and judge recomputation
uv run python scripts/05_transcripts.py sample # blinded matched transcripts for reading
uv run python scripts/06_positive_control.py plan --dry-run   # cost projection; never calls
uv run python scripts/07_plots.py              # figures from stage outputs only
uv run python scripts/08_judge_plots.py        # judge rounds on the figures
```

Stage 1 must pass before the others are trusted. Stage 6 makes API calls only with `--execute`
and `execute: true` in its config, and aborts before the first call if the projection exceeds the
configured cost cap.

## Numbers

Confirmatory run, 34,400 rows, 20,000 bootstrap draws, seed in `configs/analysis.yaml`:

| Quantity | Value (pp) | 95% interval |
|---|---:|---|
| Interaction (F−U)−(FN−G) | −0.650 | [−1.210, −0.097] |
| Famous AI − unknown twin | −0.361 | [−0.784, 0.052] |
| Famous non-AI − general population | 0.290 | [−0.107, 0.673] |
| Unknown twin − general population | −0.266 | [−0.661, 0.125] |

Discovery run interaction: −0.831 pp. See `outputs/estimands/estimands.csv` after running stage 2
and `docs/analysis_notes.md` for definitions.

## Data provenance

`PROVENANCE.md` records the source repository, tag, commit, and the map from old paths to new.
Preregistrations, decision files, status files, and audit packets are byte-identical copies.
