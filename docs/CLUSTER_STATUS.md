# Cluster execution status and transfer handoff

**Updated:** 2026-10-06 (researcher's America/New_York date).  
**Current instruction:** The runtime implementation thread now owns the upcoming
human-agent transfer, cluster measurement and full-experiment workflow. Finish
local verification and packaging before giving transfer instructions; the researcher
performs every remote operation in their authenticated session. No lock recovery,
resubmission, upload or cluster modification was performed by the agent. This note records user-supplied terminal evidence,
not a fresh direct inspection of the cluster or certification of a transfer bundle.

## Workspace and operating agreement

- Login: `zzhan330@login.arch.jhu.edu`; the researcher maintains the MFA SSH session.
- Project workspace: `/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics`.
- The researcher runs cluster commands. Agents must not access or modify the
  cluster directly without an explicit instruction.
- Installations, environments, caches and processing belong on project storage
  and allocated compute nodes, never the login node. Scheduler/accounting and
  lightweight file inspection have been performed from the login session.
- Local user-run commands use Windows CMD, never PowerShell. Cluster commands
  are supplied inside the existing SSH session without an SSH prefix.

## Billing diagnosis and verified recovery

Job `996713_4` originally used account `tshu2` and QoS `scavenger`. It remained
pending with `AssocGrpBillingMinutes` after colleagues reported a billing fix.
The October 6 association output explained the mismatch:

| Account | Parent | Allowed/default QoS | Billing-minute cap | Scheduler-reported usage |
|---|---|---|---:|---:|
| `tshu2` | `pi-tshu2` | `scavenger` | 3,000,000 | 10,070,943 |
| `tshu2_scai01` | `pi-tshu2` | `jhu` | 60,000,000 | 15,526,565 |

These are snapshot values for weighted Slurm billing minutes, not dollars or
unweighted GPU minutes. Earlier reports showed a 3,000,000 billing-minute cap
on `tshu2_scai01`; the increased 60,000,000 cap was present in both accounting
output and the scheduler's association cache. The default `tshu2` account remained
over its cap. Storage directory names do not select a job's compute account.

The researcher updated the existing pending job to account `tshu2_scai01` and
QoS `jhu`. The first call printed `Access/permission denied`, although the ensuing
job record already showed both new values. A repeated update returned without
an error. The reason temporarily became `None`, and the job subsequently ran.
The mixed initial response is recorded without attributing an unverified cause.

**Observed outcome:** this job passed scheduling under `tshu2_scai01`/`jhu` and
started. This confirms billing admission for that execution, not permanent quota
availability or a fix to the separate `tshu2` account.

## What actually ran

- Job: `996713_4`, still named `neurosym-qwen-default` after its account change.
- Request: one GPU, 96 GiB host memory, 30-minute time limit, eligible partitions
  `rtx6000,a100,h100,h200`.
- Allocated node: `gr103`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, reported 97,887 MiB memory;
  driver `595.71.05`.
- Slurm start/end: `2026-10-06T17:22:43` / `2026-10-06T17:22:49`, copied verbatim
  from cluster output; their timezone has not been independently established.
- Result: `FAILED`, exit `1:0`, elapsed `00:00:06`; the batch step also failed.
- Log: `/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/logs/qwen-default-996713_4.log`.

The log printed `CACHED qwen38-27b story_01` through `story_11`. In the existing
extraction implementation, this means the files passed the stored run-hash,
tokenizer-plan-hash and completion-attribute checks. Extraction then reached
the separate temporal-alignment command. No new Qwen model forward pass was
performed in this run; allocation on this GPU is not a new numerical validation
of Qwen inference on Blackwell. The cache checks are not a full tensor-content
audit or a scientific prediction result.

## Remaining blocker: an interrupted alignment lock

The failure occurred while acquiring this file:

`/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics/data/features/frozen/qwen38-27b/aligned/RUNNING.lock`

The researcher read its contents:

```json
{"pid": 3931069, "job": "995154", "started_utc": "2026-10-04T09:57:13.176008+00:00"}
```

The subsequent `squeue -u zzhan330` contained no jobs. Accounting showed:

| Job | State | Exit code | Elapsed | Significance |
|---|---|---|---|---|
| `995116_0` | COMPLETED | `0:0` | `00:03:18` | Earlier model task completed |
| `995116_1` | COMPLETED | `0:0` | `00:03:17` | Earlier model task completed |
| `995116_2` | COMPLETED | `0:0` | `00:05:45` | Earlier model task completed |
| `995116_3` | COMPLETED | `0:0` | `00:03:14` | Earlier model task completed |
| `995116_4` | TIMEOUT | `0:0` | `00:02:39` | Original Qwen task interrupted |
| `995154_4` | TIMEOUT | `0:0` | `00:05:02` | Retry whose array base ID matches the lock |
| `996713_4` | FAILED | `1:0` | `00:00:06` | New allocation stopped at that existing lock |

Both timed-out tasks' batch steps were cancelled with `0:15`. A `0:0` field on
the parent timeout record is not successful completion. The lock's recorded
array base `995154` corresponds to the finished `995154_4` task. Together with
the empty user queue, this identifies an interrupted, stale alignment lock in
the supplied snapshot. It is not evidence of an active current alignment process.

**The lock remains in place.** No deletion or archival move was carried out.
The number and completeness of files already inside `aligned/` have not been
re-inspected. Qwen extraction is cached for all stories; Qwen alignment is not
yet certified complete. The other four tasks' earlier completion records are
retained evidence, not a new all-model artifact audit.

## Deferred work for the integrated transfer

1. Preserve the downloaded dataset, model weights, all extracted story HDF files,
   existing aligned files and run provenance during transfer. Use the current
   integrated implementation's own verification/package receipts; earlier ZIP
   hashes do not certify changes made in other implementation workstreams.
2. When cluster execution resumes, recheck that the lock's job remains inactive,
   archive the stale lock with its provenance, and resume **Qwen alignment only**
   on an allocated CPU node. Alignment aggregates existing states onto the scan
   clock; it does not need a GPU or repeat the completed model extraction.
3. Use the existing compatible extraction/alignment code and configuration.
   Alignment identities include hashes of `model_features.py` and `temporal.py`;
   blindly replacing these and writing into the old run directory can cause an
   identity conflict. Analysis code can be transferred separately from the
   extraction code. Resolve any actual version mismatch explicitly before reuse.
4. Verify the final aligned completion receipt and all required story files,
   then check actual input availability for the integrated analysis plan before
   launching its large fits. Do not treat cached source states as an aligned
   completion receipt or use a missing model condition silently.

These steps remain pending within the upcoming integrated transfer. Recheck live
job/storage state before executing them; the October 6 snapshot is not a current
queue or quota report. The previous pause belonged to the earlier extraction
thread and does not cancel the researcher's newly authorized local implementation.
Accepted annotation build `5cd83e0779bb5440da47` remains closed and unchanged by
this operational work. No encoding, decoding or human-model result is established
by the scheduler recovery or cached-file messages.

## Relevance after the final runtime pass

- **Retain:** account/QoS diagnosis, project paths, cached extraction evidence,
  alignment lock provenance, and extraction/alignment identity constraints.
- **Recheck:** live quota, free space, running jobs, account admission and whether
  another thread has since completed Qwen alignment. Do not assume the old lock
  still exists, and do not delete a lock based solely on this note.
- **Superseded:** earlier ZIP hashes and analysis runtime estimates do not validate
  the revised code. [compute.md](compute.md) describes the current implementation
  and its still-pending full verification.
- **Initial cluster allocation:** plan a short 30-minute request over multiple
  compatible GPU partitions for numerical compatibility and actual workload timing.
  Determine production walltimes from those measurements. Current one-hour
  preparation/four-hour fit windows are configurable execution policies, not
  measured runtime requirements, and can be overridden without redefining fits.
- **Continuation:** new analysis jobs use process-owned locks and explicit saved
  progress. This does not change or automatically recover the legacy Qwen
  alignment lock. The old compatible extraction code remains intact.
- **Scientific status:** no fMRI encoding/decoding experiment or human-model
  comparison result is established by this operational record.
