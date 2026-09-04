# V1.13 bootstrap resynchronization failure

Pod `6f8skqxvd5n9xd` started successfully with two B300 GPUs and working SSH.
Bootstrap installed the pinned Transformers Git commit into the prepared
virtualenv. Its next command used `uv run python` for the identity check. Uv
then synchronized the environment back to the project lock and restored
Transformers 4.57.6, so the pinned-commit check failed closed.

No checkpoint weight was downloaded or loaded. No model forward, activation,
proxy score, or scientific row was produced. V1.14 keeps the same running Pod
and changes only the post-overlay invocation to the virtualenv's Python binary.
