# V10 offline construct audit

This package diagnoses the completed v9 readout without making a new model call.
It compares training procedures on the locked GLM feature matrix, blocks held-out
families and downstream tasks, tests structural metadata leakage, decomposes the
hard-negative failures, and measures direction stability.

It cannot reconstruct exact shared-task-suffix features because v9 did not save
token IDs or retained-position indices. It also cannot run the historical-paper
four-cell regression because no historical activation feature matrix is present
in the local frozen cache. Both limitations are recorded before analysis.

