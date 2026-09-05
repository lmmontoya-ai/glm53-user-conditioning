# V17 final report

## Outcome

V17 stopped before the Hua positive control. It produced no GLM forward, confidence score, activation direction, local-parity estimate, or causal result. The correct terminal classification is `infrastructure_unavailable_before_hua_positive_control`.

This is not evidence for or against the scientific hypothesis. V7's confirmed behavioral interaction remains unchanged, and V16's source-readout failure remains unchanged.

## What was completed

The repository contains a preregistered and tested Hua-style experiment with:

* the exact 16 published Hua prompt pairs;
* GLM-safe colon-token localization rather than a template-specific `-2` index;
* literal and depth-mapped layer schedules;
* a positive-control coefficient gate independent of Famous-AI outcomes;
* local behavioral parity using 6,387 reconstructable V7 rows and two reversed codebooks;
* a paired pilot and confirmation schedule;
* status-only, actor-only, formality, prompting, sign-flip, and Gaussian controls;
* atomic result checkpoints, signed S3 transport, two deletion watchdogs, and fail-closed decisions.

The final regression run passed 401 tests. The V17-only suite passed 18 tests. Preregistration validation, Ruff, PowerShell parsing, and Bash syntax checking passed.

## What ran on paid infrastructure

Early launches stopped while hardening private-repository transport, Windows/Linux byte preservation, and signed immutable inputs. These stopped before model staging.

One two-B300 run downloaded and hash-verified the exact 328,337,455,672-byte checkpoint in about 108 seconds and loaded all 1,872 tensors. It then found a prompt-scope implementation error before calling the model: the GLM chat template trims the trailing space from `Final answer: `, while the old span finder required an exact match for that assistant prefix. The bug was fixed and covered by a regression test. No scientific row or model forward was produced by that run.

A three-H200 retry stopped while installing the pinned runtime because its host demanded credentials for a public GitHub checkout of the exact Transformers commit. V17 then packaged the exact commit as a 20,949,119-byte source archive with SHA-256 `17890f68cae495a88b51db8105fd9bca43d5357f671fce925e3fe1f63c3cac0a`, verified a local installation, and transported it through the signed S3 input channel.

Two four-GPU RTX PRO 6000 Server Edition allocations were accepted at $8.36/hour but disappeared before bootstrap produced a heartbeat. A final four-GPU RTX PRO 6000 Workstation Edition request, restricted to the preregistered European datacenters, was rejected during Pod creation. No live Pod or Serverless endpoint remains.

After enumerating all 23 result objects and recovering every unique artifact locally, V17 permanently deleted the temporary 20 GB `glm53-v17-evidence` S3 volume. RunPod then reported no remaining network volumes.

The RunPod billing API currently records $5.153383 in V17 Pod charges from six completed billing records. More recent failed-host billing had not fully appeared in that endpoint at close; the balance declined by about $6.75 between the recorded H200 launch balance and final cleanup. The final account balance was $116.330247.

## Scientific interpretation

There is no V17 scientific result to interpret. In particular, this run does not establish whether Hua-style steering transfers to GLM-5.3, whether the exact local checkpoint reproduces the V7 interaction, or whether deployment steering attenuates that interaction.

The strongest supported result remains the V7 behavioral interaction. Any future attempt at the V17 question must be a separately authorized project with a newly frozen infrastructure plan; the current V17 decision does not authorize more compute.
