# V13 local Codex judge cohort

V13 tests a narrow explanation for the V12 failure: the earlier low-reasoning
judge may have been unable to resolve the frozen scenarios' aliases and routing
relations. It does not change any scenario, expected fact, label, split, or
threshold.

Each of the 576 frozen scenarios is judged independently by GPT-5.6 Luna at max
reasoning and GPT-5.6 Terra at high reasoning through the locally authenticated
Codex CLI. Each invocation is ephemeral, uses an empty read-only workspace, has
agent tools disabled, and writes schema-constrained output. The two models must
pass every V12 threshold separately.

If the cohort fails, V13 creates a row-, pair-, factor-, generator-, and
split-level audit. Any repair happens under a new V14 preregistration. Repaired
final-test examples cannot retain untouched-holdout status.
