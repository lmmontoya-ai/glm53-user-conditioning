# Mechanism archive

This directory is an unchanged copy of the study's original code tree at tag
`glm53-user-eval-v23-final` (commit `92fb062`) of the source repository: `src/glm53_user_eval`,
`pipelines/glm53_user_eval`, the `tests/glm53_user_eval*` directories, `results/`, the source
lock file, RunPod bootstrap scripts, and the two container build workflows. Versions v8 through
v21 hold the mechanism work that is out of scope for the current pipeline: activation probes,
Hua-style steering, local FP8 serving on rented GPUs, and their infrastructure, status files, and
preregistrations. The same package also contains the shared runner and analysis scripts (`scripts/`,
`configs/`, `reference/`, `status/`) that produced the two behavioral runs, the power analysis (v22),
and the offline decomposition (v23); they are kept here in place because the package is
import-coupled and the new `src/glm53` modules load the verified runner and extractor from this
directory by path. Nothing here is maintained or executed by the new stage scripts except those two
files. The heavy dependencies this code imported are listed under the optional `mechanism-archive`
group in `pyproject.toml` and are not installed by default. The original code imported
`src.probe.sequence_linear` from the paper repository in `src/glm53_user_eval/v10/analysis.py`;
that module was not copied, so the v10 offline diagnostics will not run from this archive.
