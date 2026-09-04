# Large transcript files

Three files in this directory exceed the 50 MB commit limit and are not tracked by git. Each
is listed with its sha256 and size. Copy them here from the local artifact tree of the source
checkout, or from the external location recorded in `PROVENANCE.md`, then verify the hash.

| File | Bytes | sha256 | Source path in the original checkout |
|---|---:|---|---|
| `v7_transcripts.jsonl` | 131967009 | `11c0b88ad1bbe4527df4b8054ddace644cb09ec4db4ccbce6edaf262132f1281` | `artifacts/glm53_user_eval/v23/derived/v7_transcripts.jsonl` |
| `v7_transcripts_all100.jsonl` | 52101243 | `858d7866153fe03984c1def4a5580478b8715ef1e0668ec73dd1c7b093c600e4` | `artifacts/glm53_user_eval/v11/downstream_inputs/v7_transcripts_all100.jsonl` |
| `v7_annotation_packet.jsonl` | 7193043 | committed manifest `v7_annotation_packet.manifest.json` records the hash | `artifacts/glm53_user_eval/v23/derived/v7_annotation_packet.jsonl` |

`v7_transcripts.jsonl` is the input stage 5 reads. It holds 28,000 rows for the four primary
groups of the confirmatory run: both subject turns (reasoning and visible text), the
provider-reported token usage of each turn, sentence counts, the first-turn yes/no, and the
stated confidence. Its manifest is `v7_transcripts.manifest.json`.

`v7_transcripts_all100.jsonl` is an earlier, smaller cache of the same run (first-turn text and
folded confidence only). It is kept because the original data listing names it; no stage reads it.

`v7_annotation_packet.jsonl` is the profile-blinded 2,000-row packet that the two judges rated.
The judge outputs it produced are committed under `data/judgments/`.

Verify after copying:

```
sha256sum data/transcripts/v7_transcripts.jsonl data/transcripts/v7_transcripts_all100.jsonl
```
