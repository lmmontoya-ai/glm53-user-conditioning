# GLM-5.3 v8 mechanism study: v1.6 active-rate probe correction

## Final stop after v1.6 attempt 3

At the user's request, endpoint `w1o540a69narzo` and its local launcher were
terminated. The endpoint never became ready: it moved from cached-model
initialization to a throttled worker state. No rate-probe request, scientific
job, handler model load, model forward, activation extraction, proxy score, or
scientific row was produced. The attempt is recorded in
`SERVERLESS_ATTEMPT_20260830_3.json`.

Post-stop verification found zero Serverless endpoints and zero Pods. Network
volume `a9diryunoj` remains in `US-KS-2` at 500 GB. The account reports
USD 110.3821185572 remaining and USD 0.049/hour current spend, attributable to
the retained volume. The scientific state remains **M0 and M1 passed; M2 not
run; M3--M8 not run**.

Initial shutdown captured: 2026-08-30 09:38:11 UTC
Offline v1.4 implementation completed: 2026-08-30
Serverless live-rate stop recorded: 2026-08-30 14:21 UTC

The v1.5 preflight endpoint `vmcevzx6lhypon` subsequently reached an idle ready
worker after 38:48, but RunPod reports zero GPU rate for an idle Serverless
worker. The v1.5 launcher failed closed and deleted the endpoint before any job
or model forward. The compact record is `SERVERLESS_ATTEMPT_20260830_2.json`,
SHA-256
`e33bd10c8380229967c87bbd574ba4c67d80cbb6434fc0c78778d056b30dc9b8`.

Preregistration v1.6 moves the rate observation into a 90-second
non-scientific `rate_probe` handler request. That request may check the exact
checkout, GPU count, and cached snapshot path, but it cannot load the model,
run a forward, or emit a scientific row. The scientific job remains forbidden
until the active rate matches USD 19.16/hour, the probe completes, the budget
check passes, and the endpoint is updated to zero minimum workers.

## Live-rate correction

The first v1.4 resume attempt remained infrastructure-only. Endpoint
`nsiraki23t7l8u`, job
`e55b92c9-1ae7-4042-8233-2123516ea8f3-e1`, staged the exact cached model and
began creating the container. RunPod's live account meter then reported
USD 19.16/hour for the four H100 NVLs, not the USD 13.392/hour rate in the
submitted supervisor payload. The endpoint was deleted before the handler,
M2, or any model forward. The balance moved from USD 111.2289789498 to
USD 110.5061341794. No endpoint remains and network volume `a9diryunoj` is
preserved.

The compact attempt record is
`SERVERLESS_ATTEMPT_20260830_1.json`, SHA-256
`cf4612b388a369933f9c88f6236f29f2ac3779fc6de105da8f31aa2db306d5a7`.
Preregistration v1.6 preserves every v1.4 scientific and audit rule, locks the
observed USD 19.16/hour active-worker rate, shortens maximum job runtime to
4.7 hours, and requires a completed non-scientific active-rate probe before the only
scientific job can be submitted. Its SHA-256 is
`85887f466c1c4aaca551bc611e75e7cbfdac0925d859e881c588c27fb39eb140`.

The user explicitly stopped execution before M2. No v8 model forward, target
activation, proxy score, probe result, or intervention result was produced.
The correct scientific state is therefore **M0 and M1 passed; M2 not run**.

## Resume update

This section supersedes the partial-implementation warning later in the
historical shutdown record.

- Audit-enforcement implementation commit:
  `ee1e852cf026282b58c5d63c8b6c31bf1b16fadc`.
- Preregistration amendment: `audit_enforcement_v1_4`.
- Preregistration SHA-256:
  `e5c2209e9cccb6faf9cc04ac0a1775ad6117b2df348b2c8f4db3ed236a3dd805`.
- Required execution tag: `glm53-user-eval-v8-preregistered-v1.4`.
- M0 was regenerated and passes, including byte-level validation of all three
  governed eval/deploy dataset artifacts.
- The registered dataset summary is locked at
  `7f9c138d845b05d87d2feb6b61dfe594f6e78aa4bb73dcf2838f87ec129b31cb`.
- The v8 suite has 115 passing tests. Ruff passes for the v8 pipeline, source,
  tests, and Serverless handler. The governed contrastive dataset contract has
  9 passing tests.
- A repository-wide compatibility run produced 466 passes and 33 failures.
  The failures are pre-existing missing paper artifacts, legacy benchmark/HF
  fixture resolution, obsolete Prime Intellect tests, and Windows path
  assumptions. None imports or executes the v8 package. The v8 and governed
  dataset suites are the paid-run blockers and both pass.
- No v8 model forward or scientific row was produced while implementing or
  testing v1.4. M2 remains the first paid gate.

v1.4 now enforces hashed M2-M8 decision inputs, measured direction-specific
positive controls, fail-closed pilot-alpha selection, frozen shared-versus-
distinct pathway selection, expanded independent recomputation, seeded manual
review, and final evidence hashing. The fame-control readout also has disjoint
training and held-out identities.

## Scientific state

- Project: `glm53_user_eval_mechanism_v8`
- Parent result: `glm53-user-eval-v7-final` at
  `2b21609c67f9921fb36426cd95c2d0c2faec9c60`
- Parent interaction: -0.65024890176303 percentage points, 95% interval
  [-1.2099960877594156, -0.09734531995242601]
- v1.4 implementation commit: `ee1e852cf026282b58c5d63c8b6c31bf1b16fadc`
- Scientific preregistration tag: `glm53-user-eval-v8-preregistered-v1.6`
- Preregistration SHA-256:
  `85887f466c1c4aaca551bc611e75e7cbfdac0925d859e881c588c27fb39eb140`
- M0 decision: passed.
- M1 decision: passed.
- Deterministic v8 test count at v1.4: 115 passed.
- Transcript cache: 7,000 expected rows, 6,989 eligible rows, and 11 immutable
  missing first turns retained as missing rather than fabricated.
- Transcript-cache SHA-256:
  `18510279e618577e0366b0158d6cfe22aa4540ccc40687f1d05dbc8eda4c502c`
- Proxy-codebook SHA-256:
  `d4631439589ad64ef7aa0ddc56dff8a3879885eb98833df93ba0c6b22f8a7efd`
- A--K token IDs: A 362, B 425, C 356, D 422, E 468, F 434, G 479,
  H 472, I 358, J 619, K 730.

M2 remains unopened. In particular, there is no basis yet for a claim about
local-checkpoint parity, eval/deploy representation validity, recruitment, or
causal dependence.

## Repository state at the initial shutdown

- Branch: `glm53-user-eval-v8-mechanism`
- Last clean base commit before this shutdown checkpoint:
  `466908d5329a11d63f71898c5d93579371e48952`
- Commit `466908d5` records the offline completion audit. The shutdown checkpoint
  preserved a partial implementation that has since been completed by
  `ee1e852c`.
- Partial implementation checkpoint: `ddb5e6f881bbc178672d8ba1abb375de49cfe775`.
- The partial implementation changes exactly these files:
  - `src/glm53_user_eval/v8/artifacts.py`
  - `src/glm53_user_eval/v8/decisions.py`
  - `src/glm53_user_eval/v8/on_pod.py`
  - `src/glm53_user_eval/v8/science.py`
  - `src/glm53_user_eval/v8/supervisor.py`
- That partial checkpoint had only fail-closed gate-input hashing, a pilot-alpha
  selector, a held-out eval/deploy positive-control routine, and hashed M2--M5
  decision inputs. All listed gaps were completed in `ee1e852c`.
- **Do not run the old partial checkpoint.** Run only the v1.6 tag after it has
  been pushed and independently resolved to a clean commit.
- Technical image tag: `glm53-user-eval-v8-serverless-image-v3`
- The technical commit records and sanitizes the disposable embedded checkout
  before reapplying the preregistered clean-tree gate. It does not change the
  scientific code or preregistration tag.
- GitHub repository:
  <https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness>

## RunPod state at shutdown

- Pods: none.
- Serverless endpoints: none.
- Temporary GHCR registry credential: deleted.
- Current RunPod balance: USD 111.4234233942.
- Current spend rate: USD 0.049/hour, attributable to the retained network
  volume only.
- Balance before the Serverless attempts: USD 112.7900341422.
- Observed balance change: USD 1.366610748. This includes persistent-volume
  charges and infrastructure preparation; it must not be reported as a clean
  scientific-compute cost.
- Preserved network volume: `a9diryunoj`, `US-KS-2`, 500 GB, mounted as
  `/workspace` on ordinary Pods and `/runpod-volume` on Serverless workers.
- Do not delete the volume until its v7/v8 artifacts and exact-checkpoint lineage
  have been independently restored and verified.

## Infrastructure attempt ledger

1. Ordinary Pod allocation was attempted first. Twenty-six exact 4 x H100 NVL
   allocations and twenty-one exact 5 x H100 PCIe allocations failed before Pod
   creation. They produced no GPU spend and no scientific rows.
2. Endpoint `7om5as1uxr21bp`, job
   `79582e98-643f-4f65-b2cd-fa7a83b84de6-e2`, pulled image v2 and prepared the
   exact cached model. The initial job expired during cold-start registration.
3. A retry on the now-ready worker, job
   `2c5ef92c-e514-4907-a02f-a8c332356649-e2`, reached the handler and failed
   before M2 because the embedded checkout appeared dirty. No model forward ran.
4. Technical commit `685d5e0b` and image v3 added a fail-closed checkout audit
   and deterministic restoration of the disposable embedded checkout. Image v3
   was built successfully by GitHub Actions run `33303280246` with digest
   `sha256:cca1a50c0cca1a18c0aa732f8a90ea5f4b479e3c96a57ba6137869c18b784b94`.
5. Endpoint `8al5d833ano8p0`, job
   `35eeef6c-2c30-4a9b-b777-9d443e4def1a-e2`, was deleted while queued after an
   anonymous GHCR pull-rate failure. No handler ran.
6. Endpoint `j632alqfz97xp0`, job
   `05581b66-70e0-4231-9ca1-49292a15f1c5-e1`, was deleted while queued after the
   same GHCR pull-rate failure despite authenticated registry transport. The
   temporary registry credential was later deleted.
7. The final transport path used official Docker Hub image
   `runpod/pytorch@sha256:f40e33a190d6823439541d1dde52003fbed66539a7af998f38e29f499ca5bdd6`
   with a pinned startup bootstrap. Template ID: `s452pdfo5f`.
8. Endpoint `i9820a4e9li07x`, job
   `2984992b-2098-4232-8b4f-cd4bd0ac6fed-e1`, successfully verified and unpacked
   the official image and reached `image ready, initializing model files`. It was
   deleted immediately when the user requested shutdown. No container handler or
   scientific forward had started.
9. Endpoint `nsiraki23t7l8u`, job
   `e55b92c9-1ae7-4042-8233-2123516ea8f3-e1`, reached `model ready` at
   14:18:20 UTC and began creating the container. The live meter showed
   USD 19.16/hour for its four H100 NVLs, while the frozen job payload used
   USD 13.392/hour. It was deleted before the handler or M2. The exact attempt
   is recorded in `SERVERLESS_ATTEMPT_20260830_1.json`.
10. Endpoint `vmcevzx6lhypon` reached an idle ready state after 38:48 without a
    job. RunPod exposed no GPU rate in that state, so the v1.5 preflight deleted
    it before model use. The attempt is recorded in
    `SERVERLESS_ATTEMPT_20260830_2.json`.
11. Endpoint `w1o540a69narzo` used the v1.6 non-scientific active-rate-probe
    preflight. It cycled through cached-model initialization and then remained
    throttled without a ready worker. No probe request or scientific job was
    submitted. The endpoint and local launcher were terminated on user request;
    the attempt is recorded in `SERVERLESS_ATTEMPT_20260830_3.json`.

The earlier GHCR template `a9fwi6am3u` and Docker Hub bootstrap template
`s452pdfo5f` remain as non-billable configuration objects. Prefer
`s452pdfo5f` on resume because it avoided the observed GHCR layer-throttling
failure.

## Safe resume sequence

1. Fetch the branch and tags, then verify that
   `glm53-user-eval-v8-preregistered-v1.6` resolves to a commit containing
   implementation commit `ee1e852c`.
2. Verify M0/M1 decision files and hashes on the preserved volume. Do not rebuild
   the transcript cache from mutable inputs.
3. Query the live RunPod balance, live four-H100-NVL price, volume identity, and
   capacity. Preserve at least the preregistered USD 15 reserve.
4. Inspect or recreate the Docker Hub bootstrap template so it checks out the
   v1.6 tag, then create one endpoint with:
   - four NVIDIA H100 NVLs;
   - model reference
     `zai-org/GLM-5.3-Flash:04c4e9e95c5da8862dced7e5056455116f83a7e0`;
   - network volume `a9diryunoj` in `US-KS-2`;
   - one active worker minimum and maximum during the live-rate preflight,
     followed by a server-verified update to zero minimum workers before job
     submission;
   - 16,920-second execution timeout and 60-second idle timeout.
5. Submit the frozen 90-second non-scientific `rate_probe`. Require the active
   account-meter delta to equal USD 19.16/hour within USD 0.01 and require the
   probe to finish. Delete the endpoint on any mismatch. Submit exactly one
   `supervise_v8` job with the same rate and a 4.7-hour maximum only after this
   check passes, then monitor it independently of the
   workstation. Do not submit a duplicate while it is queued or running.
6. Require the checkout-sanitization report and M2 exact-checkpoint/runtime
   decision before allowing M3. Continue only through machine-passed gates.
7. After the terminal result, back up compact artifacts, verify hashes, delete the
   endpoint, and confirm the volume still exists.

## Shutdown verification

At 2026-08-30 09:27 UTC, the last verified remote state was:

```text
runpodctl pod list        -> []
runpodctl serverless list -> []
currentSpendPerHr         -> 0.049
network volume            -> a9diryunoj, US-KS-2, 500 GB
```

This pause is an infrastructure stop, not a scientific negative result.

At 2026-08-30 09:38 UTC, one stale local read-only process left over from the
interrupted transcript-cache inspection was found and terminated. Its exact
PIDs were 42880 (`pwsh.exe`) and 51484 (`python.exe`). A subsequent process
scan found no process whose command line referenced this repository or the v8
pipeline. The RunPod CLI was not available in the resumed shell, so the remote
state was not re-queried; the 09:27 UTC empty-Pod and empty-endpoint result above
remains the latest verified remote observation.

The offline requirement audit in `COMPLETION_AUDIT_20260830.md` found several
M2--M8 enforcement gaps, including a hard-coded M7 positive-control flag and a
missing manual-audit gate. They are addressed by v1.4. Paid execution remains
forbidden until the v1.4 tag is pushed and the live RunPod budget and endpoint
state are rechecked.
