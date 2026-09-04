# GLM-5.3 roster-wide user-awareness result

## Outcome

The roster-wide extension did not establish a reliable recognized-user confidence effect in GLM-5.3-Flash. Across 70 Transluce identity sets and 50 untouched dilemmas, the famous-AI identity condition lowered stated behavioral confidence by 0.409 percentage points relative to the affiliation-matched unknown twin. The two-way cluster-bootstrap 95% interval was -0.965 to +0.168 points.

The preregistered decision is `ambiguous_roster_result_stop_and_report_heterogeneity`. No population, identity-specific, or affiliation positive gate passed. The result is not a clean null under the conservative rule because the combined lower confidence bound remains below -0.50 points.

## Question and method

The experiment tested whether the Transluce user-awareness phenomenon generalizes to GLM-5.3-Flash and supplies a behavioral substrate for a later white-box mechanism study. The central comparison held organizational affiliation fixed while changing whether the user was a recognized AI-world person:

- `famous_coherent`: a recognized AI-world identity;
- `unknown_same_org`: an unknown person assigned the same organization and email domain;
- `unknown_general`: the same unknown name in its ordinary general-population context;
- `famous_nonai_control`: a famous non-AI identity control.

The dependent variable was the model's stated confidence in its preceding yes/no behavioral prediction, using the released Transluce two-turn structure and parsing convention. Confidence was not inverted according to the yes/no answer.

The full 70-pair roster was chosen to test the concern that the earlier four-identity null could reflect an unrepresentative subset or model-specific identity heterogeneity. The 25-task discovery and 25-task confirmation splits were untouched and analyzed separately before pooling.

## Results

| Stage | Rows | Famous - twin effect | 95% CI | Identities with negative sign |
|---|---:|---:|---:|---:|
| Discovery | 7,000 | -0.246 pp | [-0.984, +0.503] | 34/70 |
| Confirmation | 7,000 | -0.573 pp | [-1.292, +0.218] | 38/70 |
| Combined | 14,000 | -0.409 pp | [-0.965, +0.168] | 43/70 |

The discovery stage produced no identity-level candidates after the preregistered Benjamini-Hochberg correction. Consequently, no identity-specific effect was eligible for confirmation. Identity effects correlated only modestly between discovery and confirmation (`r = 0.310`), and only 39/70 retained the same sign. This does not support the hypothesis that a different stable subset of identities carries a strong GLM-5.3 effect.

The same-name affiliation contrast was -0.108 pp in the combined data, 95% CI [-0.616, +0.395]. The generic-fame contrast was -0.117 pp, 95% CI [-0.611, +0.366]. Neither passed its gate.

The earlier preregistered four-identity result remains immutable: -0.08 pp, 95% CI [-1.615, +1.355]. The roster-wide experiment is a new amendment and dataset, not a replacement or post-hoc reinterpretation of that null.

## Data quality and execution

- All 14,000 scheduled rows completed.
- The combined parse rate was 99.793%.
- Missingness differed across conditions by 0.057 percentage points.
- All 14,000 rows were routed to first-party Z.AI, with reasoning observed in every row.
- Eighty raw transcripts were manually reviewed, 40 per stage. Binary and confidence parsing were correct in all 80; 79 were on task. The single off-task row had an empty main response and was correctly marked invalid rather than assigned a fabricated binary label.
- Independent scripts reproduced the discovery, confirmation, and combined headline estimates to floating-point tolerance.
- Subject and judge calls cost $13.599 in total.

## Parallel and resumable runner

The final run used 96 concurrent scientific rows. Each row saved four independent turn checkpoints: main subject response, follow-up subject response, binary judgment, and confidence judgment. Results were appended by one centralized writer under an OS-level lock. A resume scan used `sample_id` and checkpoint presence to execute only missing work.

One confirmation row initially failed before a scientific response with a JSON-decoding error. The unattended controller retried that same sample ID, resumed from the completed main-turn checkpoint, and finalized the remaining turns. The final ledger contains the row exactly once.

During the initial discovery launch, two writers were accidentally started before the writer lock existed. The original 2,811-line ledger is preserved. Reconciliation matched raw response artifacts to recover 1,741 unique canonical rows and excluded 1,070 redundant lines without rerunning or selecting on outcomes. The hardened writer lock prevented recurrence.

## Interpretation

The data do not support proceeding to activation-direction or steering experiments on the hypothesis that recognized AI-safety identities robustly recruit an eval/deploy state in GLM-5.3-Flash. The prerequisite behavioral effect was small, inconsistent across identities, and failed both discovery and confirmation gates.

The result is still informative but should be described precisely: this implementation rules out a large, roster-wide effect more strongly than the four-identity pilot, while leaving uncertainty about sub-percentage-point population effects. It does not show that GLM-5.3 has no user representations, that no individual identity can ever matter, or that the Transluce result was invalid for other models and serving stacks.

## Decision

Stop the GLM-5.3 shared-direction project at the behavioral gate. Do not spend RunPod credit on exact-checkpoint serving, activation extraction, or causal steering for this hypothesis without a new scientific rationale and preregistration.
