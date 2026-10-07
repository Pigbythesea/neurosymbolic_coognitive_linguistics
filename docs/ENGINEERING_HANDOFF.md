# Engineering handoff

Updated 2026-10-07. This is the engineering entry point for a colleague taking
over the current working tree. Scientific framing and protocol decisions belong
to the separate scientific thread; see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).
Remote facts below come from researcher-supplied logs, not direct cluster access.

## Current position

**The trace/storage optimization passed integrated local checks, was installed
by job 1026424, and passed real fitted-reference CUDA qualification in job 1026427.
Selected production fits completed; the full experiment has not been launched.
The selected complete exports and geometry panels finished in controller job
1026940 (43:34); detailed GPU time, phase timing and final bytes await collection.** Scientific definitions, configurations,
folds, optimizer steps and seeds remain unchanged in this optimization pass.
Scientific decisions are recorded in [SCIENTIFIC_STATUS_HANDOFF.md](SCIENTIFIC_STATUS_HANDOFF.md).

### Latest evidence and optimization boundary

- Installation `1021945` and CUDA verification `1021947` completed. Controller
  `1021961` completed 20 selected workers / 54 logical jobs in 42:33. Ordinary
  refit epochs were about 4–7 seconds. These supersede the older timing table below.
- Context trace workers 102 and 1220 each consumed about 108 allocated minutes
  over four resumptions. Each finished its first seed and began exporting its
  second. Completed trace files were about 5.0 and 5.6 GiB. Old `runtime.json`
  reported only the last process, so its 546/663-second export phases were not
  total export times. The subsequent bottleneck was export/derived work.
- Allocated CPU job `1024054` collected the real fitted audit in about 62 seconds.
  The researcher transferred `artifacts/trace-audit-1024054.zip`; all 286 embedded
  file hashes were verified during local import. It contains actual weights,
  windows, full predictions, two complete-source trace samples per system,
  finished second-seed weights/checkpoints, installed source, and storage inventory.
- A prototype batched capture passed per-array tolerance checks on all 570 sample
  query executions, but failed exact geometry aggregation: one predicate changed
  from three unique numerical signatures to two, shifting a mean coordinate by
  about 1.65. This is **not** qualification of that prototype. The production
  replacement preserves scalar trace arithmetic and HDF5 lexical field order;
  no numerical rounding or new deduplication rule was adopted. `decoders.py` and
  the training `decoder_batch.py` remain unchanged.
- The first integrated local trace check stopped at the shared-reducer assertion.
  Its reference accumulated signatures in annotation order, unlike the legacy and
  optimized reducers' lexical query/repeat order. A targeted real-source diagnosis
  (147 queries, 149 available source/items) found identical unique-signature counts
  and bit-identical means when the reference used legacy order; the wrong order
  caused differences up to 9.54e-7. The verifier now preserves legacy aggregation
  order and explicitly checks signature counts without relaxing exact equality.
  Evidence: `artifacts/trace-reducer-order-diagnostic.json`. The subsequent complete
  local check and six-stage packaging passed; CUDA qualification also passed below.
- Installation `1026424` completed in 11 seconds. CUDA qualification `1026427`
  completed on RTX PRO 6000 in 20:37, testing all 570 real sample query executions
  with zero array/probability differences from the original GPU traces. Signature
  counts, legacy-ordered geometry means, cross-seed constant-program reuse and
  fitted/prediction/source-block recovery passed. Peak allocated/reserved CUDA
  memory was 196/232 MiB; process MaxRSS was about 6.30 GiB. These are sample
  qualification peaks, not full worker budgets. The total time includes repeated
  scalar/reference comparisons; compact exports took about 14.1/15.3 seconds per
  two-source sample and occupied 2,492,704/7,839,453 bytes (model/brain).
  Copied legacy subsets expand hard links, so their compression ratio must not
  be extrapolated to full traces. Evidence: `artifacts/trace-verification-cuda-1026427.json`.
- Before fit reuse, inspection of the helper found `execution.py` missing from its
  allowed list of already qualified optimization changes. The corrected helper
  supports `--installed-code`, imports the unchanged qualified installation, and
  records its own SHA256 separately in the reuse registry. Transfer only the small
  maintenance packet and run `maintain_qualified_fits.sbatch`; do not overwrite
  installed code or alter numerical qualification receipts. The original bundled
  `qualify_trace.sbatch inspect/maintain` invokes the older helper and will reject
  this execution-module change. Read-only maintenance inspection `1026930`
  completed in 10 seconds: 27 decoder fits, nine selection receipts and 27 encoding
  fits are reusable; four finished decoder refits can be restored for optimized
  export. Cleanup identified exactly nine retired test directories, 349,588,394
  bytes. Maintenance `1026932` then completed in eight seconds: reuse was applied,
  all four finished refits were restored, and all nine retired test directories
  were removed with their metadata archived. Original current fits, reference
  traces, encoding objects, data and caches were retained.
- Selected production controller `1026940` completed in 43:34 with exit 0 and
  `SELECTED WORKERS COMPLETE`; the researcher-reported queue was empty. Arrays:
  preparation `1026943`, selection reuse `1026948`, decoder export/refit `1026953`
  and continuation `1027029`, geometry `1027030`, `1027048`, `1027052`.
  All six context decoder outputs and their six geometry panels are complete.
  The first decoder array needed continuation, which the controller handled.
  Accounting and cumulative fit runtimes/storage still need inspection; 43:34
  includes reused work, scheduling and overlapping stages, and is not a cold-run
  or full-study runtime estimate.

Implemented and numerically qualified; fit reuse/cleanup completed, full throughput remains pending:

1. `trace_store.py` stores shared source/repeat tables with explicit query-local
   descriptor/operator membership, routing, masks, latents and composition steps.
   GPU-to-host transfers are consolidated; original arithmetic and precision are
   retained. Source/repeat blocks resume independently. Legacy files stay readable.
2. `trace_geometry.py` reduces all requested grounding kinds/scopes in one source
   pass. Exact numerical-signature deduplication and original ordering remain.
   Pair-matrix reuse is keyed by actual factors/mask, with a 128 MiB resident cap.
   Temporary source-vector tables survive yields and are removed after all panel
   outputs finish; small reduction/support counts remain.
3. `decoder_stages.py` commits weights/projector and prediction stages separately.
   Export resumes skip training-window preparation and query packing. A finished
   training checkpoint also returns before repacking. Parameter-free programs
   can be reused across compatible actual fits; model tensors/gradients/targets
   are excluded and resident program storage is bounded.
4. Decoder runtime records now retain every attempt and cumulative phase/wall
   seconds, including yields. Encoding/PCA and permutation/bootstrap definitions
   were not redesigned in this pass; no additional speedup is claimed for them.
5. `adopt_protocol2_fits.py` explicitly qualifies reuse against the pinned installed
   protocol-2 manifest. Completed ordinary fits retain their original identities.
   Measured context weights (including two finished refits whose export was
   interrupted) can be imported with original weight hashes into new export runs.
   Optimizer-in-progress checkpoints are never relabelled or imported.
   Qualified existing PCA/window caches retain their original producer receipts;
   new consumers can read them explicitly without copying or recomputing them.
   Missing cache entries are generated under the new identity, never stamped
   with the old producer code. Imported projectors must exactly match prepared
   projectors before reusing their windows.
6. `cleanup_retired_decoder_tests.py` targets only the nine inventoried unfinished
   pre-protocol-2 decoder directories (349,588,394 bytes, about 333 MiB), after
   fresh qualification, dependency and idle-job checks. It archives their JSON
   provenance before deletion. Current fitted parents/traces, encoding objects,
   source data and caches remain protected. These nine directories were removed
   by researcher-run maintenance job `1026932`; no other deletion was authorized
   or reported in this pass.

The selected complete export/geometry path finished successfully on the cluster;
its detailed accounting and full-study runtime/storage projection remain pending. The prior
rough 200 GPU-hour estimate covered ordinary preparation/fitting/encoding only;
it excluded expensive context export/geometry and queueing. Do not quote it as a
full-study forecast. The eight-GPU several-day target remains unverified until
complete new export/geometry and concurrent throughput are measured.

## Workspace, ownership, and operating rules

- Local: `C:\Users\pigby\neurosymbolic_coognitive_linguistics`.
- Cluster: `/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics`.
- User: `zzhan330`; login: `login.arch.jhu.edu`; account: `tshu2_scai01`; QoS: `jhu`.
- The researcher performs every remote command, upload, submission, cancellation,
  and long local run. Give local commands in **Windows CMD, never PowerShell**;
  give cluster Bash commands for their existing authenticated SSH session.
- All project environments, caches, temporary files, logs, and outputs belong
  inside the cluster workspace. No installation, environment construction,
  downloading, or analysis on login nodes. Submit these to allocated compute.
- Aim for **4–8 concurrent GPUs total across stages**, with eight the current
  project ceiling. Prefer organized arrays and measured, resumable allocations.
  Arrays still expose individual tasks and do not reduce resource consumption.
- Eligible decoder GPU partitions: `l40s,rtx6000,a100,h100,h200`.
  FP64-heavy preparation/encoding profiles use `a100,h100,h200`.
  **Do not use B200/B300.** Shared-account availability must be inspected again;
  a previously observed 16-GPU account limit is not eight reserved project GPUs.

## Exact version to preserve

This optimization pass started from HEAD `742f1af2a544dfa18d2d4ab22a37211238c966a5`.
**Optimization edits are uncommitted. Cloning that HEAD alone is insufficient.**
Preserve the working tree and retain the older bundle as historical evidence.

Current implementation centers on the three analysis/experiment/compute configs,
decoder batching/selection, encoding selection, geometry, execution and packaging.
New required modules are:

```text
neurosym/protocol.py
neurosym/decoder_minibatch.py
neurosym/study_jobs.py
neurosym/study_reports.py
neurosym/grounding_checks.py
```

Current qualified installation: `analysis_code/17181841f6fd955de63477a406143d2b31e739b4b7507d81846cc4860abe71ff`;
development manifest `01b086932fdf03228e899b2e5c131e7514b46ba76d67b080245548837a2df61e`;
ZIP SHA256 `224d371b89c38c3bd73580e9fbe702cef34fe20a8fee5ecc27be4b1e067489a7`.
The local maintenance helper correction is newer than this immutable bundle;
its separate transfer receipt pins the helper and the qualified installation.

The following identities describe the **earlier verified cluster installation**,
not the current working tree. The accepted semantic build itself is unchanged.

| Identity | Previously installed value |
|---|---|
| Accepted semantic build | `9bce4af5463a57683ed375a8b0fdcbcb83620bce923624d7075bd0c4f4ac847f` |
| `artifacts/analysis-source.zip` SHA256 | `ca7f8569ec31d301194c54e0c926e2401d03c4b54703cef63915eb96471bde88` |
| Isolated analysis code directory | `analysis_code/cccd608df834c15bfa18b28005116be86465ae1da610973b2e33a2699f3e0b18` |
| Development manifest hash | `0fdc388fef401ad513c9e2d853fc2601d4915711aaacf3b95534bfffcddde2c6` |
| Analysis configuration hash | `f763fcb4ecd1bc35f3db61c8e1b0c59ac4fb867a5a51e3524bc8821ccf6f0b7c` |
| Compute configuration hash | `99ef22a04ab211da887649c8ba18d2f2ab51c6458d911429ad36841145a6a124` |
| Experiment definition hash | `10b298a1ce530f541eabb9aee376d042aff161139eaeb3bbe7a406294845940d` |

The isolated code directory is packaged in the ZIP and installed
on the cluster; it is not an unpacked local directory. The manifest is
`artifacts/execution/<manifest-hash>/manifest.json`, also identified by
`artifacts/execution/latest.json`.

Cluster `.analysis-source.json` selects installed code. Editing local source does
not update that installation. Code/configuration changes require fresh packaging,
identities, and transfer. Never relabel old checkpoints or receipts to force
compatibility with new training code. Preserve them under their original identity.
These optimization changes do not require reannotation or frozen-model extraction.

## Protocol-2 implementation record (qualification subsequently completed)

- Source programs now support multiple observations in one tensor execution.
  Training packs 16 sources per AdamW step, precompiles train/validation query
  constants once per fit, and bounds their combined storage to 1 GiB. Public
  inputs and targets remain separate. The shorter last batch and observation
  repeats retain the declared equal-story/source objective.
- Three deterministic whole-story inner folds select nonlinear readout settings
  once at seed 11 per observation/outer partition; refit seeds share that receipt.
  Linear probes use fixed settings. Structured nulls reuse matched settings.
  These are protocol changes, not identical old optimizer trajectories.
- Encoding uses one union of three candidate weight grids and one selected fit;
  validation rows and fold scores preserve equal-story weighting. Existing
  FP64 primal/dual algebra, response operators and exact-fit sharing remain.
- Intermediate-layer probes prepare only outer-fold projectors. PCA still uses
  GPU moments/projection and the configured CPU partial eigensolver. Decoder
  arithmetic stays FP32; PCA/ridge stay FP64; TF32 is disabled.
- Full traces are exported only for structured context parents. Trace export
  resumes at completed source/repeat boundaries. Faithfulness replacement
  checks also save progress at allocation boundaries. Geometry aggregates
  streaming sums, retains anatomical signatures and RDMs, and avoids storing
  redundant per-story high-dimensional arrays in the default manifest.
- The manifest includes geometry, semantic coverage, comparison panels, paired
  effects, crossed intervals and reporting. Missing semantic support is explicitly
  unavailable. Query/target/weight support must match for decoder contrasts.
- The dispatcher caps its outstanding GPU tasks at eight across its project
  manifests, under a shared controller lock; CPU workers have a separate cap.
  Manual jobs are outside that ledger. Arrays organize submissions; a task still
  requests one GPU. Independent-fit vectorization/multiple processes sharing one
  GPU are not implemented or assumed in a throughput forecast.
- Packaging generates artifacts/analysis-transfer.json and copies the current
  update helper. The helper verifies those uploaded hashes and requires an idle
  project plus an allocated node. It no longer pins a superseded ZIP hash.

Targeted checks on 160 actual accepted queries across seven real text-feature
observations passed scalar/single-source/multi-source logits, losses and gradients
for all four families (multi-source maxima approximately 2.39e-6 / 1.87e-7 /
2.40e-6 respectively). CPU epoch and mid-minibatch recovery produced identical
parameters and selection. Python 3.11 syntax and both phase dependency graphs
passed. These checks do not validate convergence, full-array memory use or GPU
throughput. The complete six-stage preparation and updated CUDA verification
were subsequently completed as recorded above; they do not certify the new trace optimization.
Additional targeted checks passed trace/faithfulness export recovery on two
complete real sources, streaming/compact geometry on 16 supported concepts,
equal-story row scaling, and development semantic coverage (ten stories,
eleven query families). The latter is a support audit, not annotation accuracy.

Cluster `.venv-analysis` has PyTorch `2.10.0+cu128` (CUDA runtime 12.8).
Existing `.venv` and `.venv-extraction` are separate environments. Setup/preflight
confirmed all 18 response files at expected sizes and five aligned model receipts,
22 metadata-verified files each: Qwen 3.5 9B base/post, OLMo 3 7B base/instruct,
and Qwen 3.8 27B. This checks metadata/provenance, not a fresh full tensor rehash.

## Earlier cluster evidence (superseded by the current-position section)

| Job | Work | Last confirmed result |
|---|---|---|
| `1019947` | Install current source bundle; reuse environment | Complete, 10 s, exit 0; preflight issues empty |
| `1019967` | Source-batched CUDA equivalence and recovery | Complete, 1:47, exit 0, about 5.26 GiB host RAM |
| `1019982` | Logical 30: brain subject01/fold0 preparation | Complete, 25 s, about 1.31 GiB host RAM |
| `1019983` | Logical 3894: Qwen27B/layer64/fold0 preparation | Complete, 24 s, about 2.31 GiB host RAM |
| `1019984` | Logical 0: CPU query prior | Yielded at 26:49, exit 75; inner 4/9, first LR, epoch 9; about 8.75 GiB host RAM |
| `1020081_0` | Logical 31: brain linear decoder | Yielded at 26:47; reached inner fold 2/9 |
| `1020081_1` | Logical 33: brain structured decoder | Yielded at 26:46; still inner fold 1/9 |
| `1020081_2` | Logical 3895: model linear decoder | Yielded at 26:47; reached inner fold 2/9 |
| `1020081_3` | Logical 3897: model structured decoder | Yielded at 26:45; still inner fold 1/9 |

All four array tasks used separate RTX PRO 6000 Blackwell GPUs on `gr101`, one
GPU plus six CPUs and 32 GiB RAM each. Warm epoch times:

| Work | Current measured epoch | Previous executor comparison |
|---|---:|---|
| CPU prior, four threads | 17–18 s | About 54–55 s |
| Brain linear, GPU | 43–46 s | About 335–354 s |
| Model linear, GPU | 42–44 s | No equivalent completed comparison |
| Brain structured, GPU | 105–110 s | Old first epoch 987.6 s; new first epoch 109.3 s |
| Model structured, GPU | 105–109 s | No equivalent completed comparison |

Short live samples showed linear GPU SM activity 5–7%, structured 0–17%, and
roughly 1–1.5 GiB framebuffer use. Task CPU accounting was consistent with one
busy core despite six allocated. This suggests small-kernel/host overhead; it
does not isolate the cause or establish a speedup proportional to unused GPU
capacity. Logs are `logs/batch-decoder-1020081_*.log` and
`logs/batch-prior-1019984.log`. All these known timing allocations have ended;
the entire current user queue has not been independently inspected.

## Workload and runtime target

Protocol 2 development resolves to 10,285 logical items in 4,033 workers:

| Work | Development | Final |
|---|---:|---:|
| Selection receipts (including fixed linear settings) | 608 | 58 |
| Decoder refits, including shared priors | 1,944 | 186 |
| Decoder preparation | 318 | 29 |
| Logical encoding fits | 5,688 | 567 |
| Geometry panels | 220 | 0 |
| Geometry comparison panels | 1,505 | 0 |
| Semantic coverage / study report | 2 | 2 |

Final evaluation has 224 workers and remains a separate explicit release.
There are 347 nonlinear/prior tuning conditions across both phases, each with
three inner folds and two learning rates, followed by shared-settings refits.
The 319 other selection receipts contain fixed linear settings and train no
inner models. Encoding logical fits can still share identical numerical objects.
These counts include all scheduled downstream work, not just fitted parents;
they are not GPU allocations or simultaneous job IDs.

The previous 4,002 decoder / 23,220 encoding counts and 100-160-day extrapolation
describe the superseded grid. The several-days-on-eight-GPUs objective remains
a throughput target, not an achieved result or defensible new runtime estimate.
Measure protocol 2 selection, refit, export, geometry, host/VRAM use and concurrent
throughput on the cluster. Do not multiply the old per-source epoch timings by
the new fit counts, or infer a 16-fold speedup from batch size.

## Storage and cleanup

Earlier allocated-node disk measurements: dataset 25 GiB, frozen features 47 GiB,
Hugging Face model cache 115 GiB, analysis cache 2.7 GiB, and analysis outputs
363 MiB: about 190 GiB across those paths, excluding environments and other
directories. The new audit measures **15,379,811,885 bytes (14.32 GiB)** under
analysis outputs; the old 363 MiB figure is obsolete. Other directories were not
remeasured by this trace audit. The shared lab filesystem previously had 8.9 TiB free; that is
not a personal quota or reserved capacity.

Protocol 2 development/final encoding operator-and-metric array arithmetic is
approximately 252 decimal GB (235 GiB) before masking, compression and identical-fit
sharing. This is an array ceiling for those outputs, not total workspace use;
decoder weights/predictions/context traces, geometry, metadata and caches are
additional. The previous 1.036 TB encoding figure and 2-3 TB planning allowance
belonged to the old grid and must not be quoted as the new measured requirement.
Geometry saves RDMs and anatomical signatures by default; wide native/implied
signatures and all per-story vectors are reconstructible rather than duplicated
in every panel. Actual complete output sizes remain to be measured.

The disk analysis cache has a 128 GiB admission limit, 1 GiB entry reservations,
and a 16 GiB minimum-free-space guard; it is not automatic disk LRU cleanup.
Host/device/raw resident reuse budgets are separately 4/4/16 GiB. Process exit
releases RAM/VRAM, **not downloaded files or saved outputs**. Some intermediate
checkpoints are removed after completed fitting stages; no broad cleanup is
automatic. Do not delete caches, reservations, or old outputs without checking
active ownership and the reproducibility/continuation requirements.

## Next engineering work, in order

1. Completed: run the integrated local command in Windows CMD:
   **call scripts\validate_trace_optimization.cmd**. It compares the actual fitted
   audit against the compact codec, same-device scalar execution, exact grounding
   aggregation, cross-seed program reuse and stage recovery, then invokes the
   existing six-stage preparation/package command. No synthetic observations are used.
   A pre-existing ZIP is not evidence that this command passed.
2. Completed by job `1026424`: transfer the new ZIP, installer, update helper and generated
   transfer receipt. Inspect active project jobs first, then install only through
   the allocated-node update helper. Reuse the existing analysis environment and
   frozen model features. All paths remain in the project workspace.
3. Real fitted-reference CUDA qualification passed in `1026427`; maintenance
   inspection/application passed in `1026930`/`1026932`. The next selected run
   uses workers `100,101,102,1218,1219,1220,2308,2309,2310,2382,2383,2384` in
   manifest `01b086932fdf03228e899b2e5c131e7514b46ba76d67b080245548837a2df61e`:
   a checked dependency-closed set of 12 workers / 16 logical items, comprising
   two preparation/selection chains, six context decoder outputs (three seeds
   each for subject01 and Qwen27B layer64), and their six full geometry panels.
   Four decoder weights are restored; two third-seed refits remain. A lightweight
   allocated CPU controller dispatches 30-minute resumable worker arrays, with
   up to two decoder GPUs and four geometry GPUs overlapping (at most six for
   this selected DAG, still subject to the project-wide eight-GPU ceiling).
   This selected run completed under controller `1026940` in 43:34; collect the
   cumulative decoder runtimes, per-worker accounting, geometry sizes and cache
   occupancy next. No new annotation or frozen-state extraction is required.
   Inspect packing size, optimizer steps, epoch/validation/checkpoint/export time,
   convergence, peak host/VRAM use and final bytes. Completed eligible outputs
   remain production results; these are not substitute-data experiments.
4. Set allocation walltimes from complete measured work and run organized arrays
   under the shared eight-GPU admission cap. Several-day completion still needs
   measured concurrent throughput. Independent-fit vectorization is an optional
   subsequent optimization if batching leaves a demonstrated bottleneck; it is
   not currently implemented and has no assumed speedup.
5. Complete development reports and resolve scientific choices before explicitly
   releasing the separate final manifest. Preserve old outputs/checkpoints under
   their original identities; no broad deletion or automatic remote action occurred.

## Where the next engineer should look

- [CLUSTER_STATUS.md](CLUSTER_STATUS.md): detailed chronological evidence and old
  commands. Earlier pending/running statements are historical; this handoff
  consolidates their latest outcomes.
- [compute.md](compute.md): implementation rationale. Its opening pending-check
  statements predate the successful packaging/CUDA/timing jobs recorded here.
- [decoder_batch.py](../neurosym/decoder_batch.py),
  [decoder_fit.py](../neurosym/decoder_fit.py),
  [decoders.py](../neurosym/decoders.py): immediate runtime work.
- [execution.py](../neurosym/execution.py),
  [submit_analysis.py](../scripts/submit_analysis.py),
  [compute.json](../configs/compute.json): inventory, dispatch, resource budgets.
  `job --index` refers to a logical item; `worker --index` refers to a grouped
  worker. Those index spaces are different.
- [run_analysis.sbatch](../scripts/run_analysis.sbatch): project-local environment,
  installed-code selection, GPU verification, signal/checkpoint handling.
  Manually submitted jobs do not resume automatically; the existing watched
  dispatcher supports continuations but is not the proposed global controller.
- [prepare_analysis.cmd](../scripts/prepare_analysis.cmd),
  [verify_compute.py](../scripts/verify_compute.py),
  [package_analysis.py](../scripts/package_analysis.py): verification and packaging.
- [update_analysis.sbatch](../scripts/update_analysis.sbatch): current update-only
  helper. It pins bundle/helper hashes; refresh pins after a new bundle. Do not
  reuse the historical one-time `finish_cluster_setup.sbatch` recovery procedure.

No cluster command, scientific configuration change, or new experiment launch
was performed while writing this handoff.
