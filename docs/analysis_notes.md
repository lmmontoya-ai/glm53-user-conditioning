# Analysis notes

What each quantity in this repository is, and why it is measured that way.

## The task

Each row is one conversation with the subject model on the pinned Transluce user-awareness
task. The system prompt names the user, their pronouns, an affiliation, and an email address.
The user message is one of 100 daily dilemmas written as a deployed-agent scenario that ends by
asking whether the model would take a specific action. The model answers yes or no. A second
turn asks how confident it is, as a percentage. A separate extraction model reads both replies
and records the yes/no and the integer confidence. Rows where the extraction found no usable
answer are recorded as missing and excluded from the quantities below.

## Folded confidence

The outcome is `max(p, 100 - p)`, where `p` is the stated confidence. Folding removes the
direction of the answer so that "90% confident I would" and "90% confident I would not" count
the same. It measures how sure the model says it is, not what it says it would do.

## Centering and the identity effect

Every dilemma has its own baseline level of confidence. To compare identities across dilemmas,
each row is centered on the mean folded confidence that the 70 general-population identities
produced for that dilemma. An identity's effect is the mean of its centered rows over the
dilemmas where it has a valid score. Identities are then averaged with equal weight within each
population. This is the aggregation the original Transluce code uses, and the runs reproduce it.

## Populations

- General population (G): 70 ordinary names and jobs.
- Unknown AI-affiliated twin (U): the same 70 names moved to an AI lab. The twin shares its
  roster index with one famous AI figure at the same organization.
- Famous AI figure (F): 70 well-known people in AI, with constructed email addresses.
- Famous AI figure, published address (Freal): 59 of the same people with their real public
  address, used to check that the constructed address is not doing the work.
- Famous non-AI figure (FN): 70 well-known athletes, entertainers, and other public figures.

## Estimands

- `U-G`: does an AI affiliation alone change confidence?
- `F-U`: does being a famous AI figure change confidence beyond the affiliation?
- `FN-G`: does fame in general change confidence?
- Interaction `(F-U)-(FN-G)`: the part of the famous-AI shift that generic fame does not
  explain. This is the preregistered primary quantity. A negative value means the model states
  lower confidence to famous AI figures than generic fame would predict.
- `F-G` and `Freal-G` are reported for comparison with the original Transluce statistics, with
  the same tie-aware Mann-Whitney test across identities.

## Uncertainty

Intervals are crossed percentile bootstraps with 20,000 draws. One draw resamples the 100
dilemmas once (the same draw for every population), resamples famous-AI and twin identities as
pairs so the pairing is kept, and resamples the two control populations independently. The
general-population center is recomputed inside every draw, so uncertainty in the baseline is
included. Seeds are recorded in `configs/analysis.yaml`; the same seeds reproduce the committed
endpoints to floating-point precision.

## Decomposition

The interaction could come from the yes/no answer changing, or from confidence changing given the
answer. Three quantities separate these:

- Yes-rate interaction: the same four-group contrast applied to the answer coded 1 for yes.
- Choice-standardized confidence interaction: within each population, confidence is averaged
  separately over yes rows and no rows, then recombined with the pooled yes-rate as the weight.
  This removes any difference caused by populations answering yes at different rates.
- Same-answer matched estimate: paired differences (famous minus twin, famous non-AI minus
  general population at the same roster index) kept only where both members gave the same
  answer on that dilemma. It is descriptive, not causal, because the retained cells are selected
  on an outcome.

## Transcript measures

Two model judges rated 2,000 profile-blinded transcripts on several dimensions. Only dimensions
with adequate inter-judge agreement are interpreted: explicit evaluation language (binary) and
deference to the user (0 to 2). Interactions are computed on the judge-averaged rating in the
rating's own unit. Response length is measured in provider-reported visible tokens and in
sentences of the confidence-turn reply.

## Per-identity effects and roles

The twin-adjusted effect of a famous-AI identity is its effect minus its twin's effect. Because
the twin has the same name and affiliation, this isolates recognition of the person. Role coding
follows the frozen protocol in `configs/roles/PROTOCOL.md`: coders see only name and role text,
the discovery run is the held-out test for the scrutiny-minus-business contrast, and the
confirmatory run is reported second.

## Positive control

The context blocks add one explicit sentence about the session (ordinary use, model assessment,
or a neutral note) after the identity profile. Run on general-population identities only, the
model-assessment block asks whether telling the model outright that it is being assessed moves
folded confidence in the same direction, and on the same dilemmas, as famous scrutiny-role
identities do. This stage is built and costed but has not been run.
