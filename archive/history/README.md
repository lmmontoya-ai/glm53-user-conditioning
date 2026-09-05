# Source history bundle

`non-verbal-eval-awareness-glm53-history.bundle` is a self-contained git bundle of the study's
full history in the source repository `github.com/lmmontoya-ai/Non-verbal-Eval-Awareness`: the
branch `glm53-user-eval-v23-behavioral-decomposition` (tip `92fb06220d124c6165852355b3b26b781ed9fe86`)
and all 163 `glm53-*` tags, including every preregistration tag. It contains the paper repository's
ancestor commits as well, which is why it is large.

- size: 470680688 bytes
- sha256: `4f6eecf02f8b8cb7d7cb2f7b7174e22454543f46ddada9cfdcd6c8f7646607be`
- heads: 164 (1 branch, 163 tags)
- created 2026-09-04 with `git bundle create`; verified with `git bundle verify`

The file is not tracked in git (GitHub's 100 MB file limit). It is attached to the GitHub release
`source-history` of this repository and kept locally at this path. To restore the history:

```
git clone non-verbal-eval-awareness-glm53-history.bundle glm53-source-history
cd glm53-source-history && git tag | head
```

The study's tracked files at the final tag are also present as plain files under
`archive/mechanism/` (code, configs, status, results, infra) and
`archive/mechanism/artifacts/glm53_user_eval/` (the remaining tracked artifacts, with
`COPY_MANIFEST.json` listing every file's sha256).
