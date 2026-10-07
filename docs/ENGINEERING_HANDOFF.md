# Engineering handoff

Updated 2026-10-07. This is the engineering entry point for a colleague taking
over the current working tree. Scientific framing and protocol decisions belong
to the separate scientific thread; see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).
Remote facts below come from researcher-supplied logs, not direct cluster access.

## Current position

**Data, model features and the existing analysis environment are on the cluster.
Protocol 2 is implemented locally; it has not been packaged, transferred or
qualified on CUDA. The full experiment has not been launched.** The researcher
approved the claim-directed redesign: primary final-layer controls, descriptive
intermediate-layer profiles, shared three-fold selection and cross-source
minibatches. Scientific decisions are recorded in
[SCIENTIFIC_STATUS_HANDOFF.md](SCIENTIFIC_STATUS_HANDOFF.md).

The latest four GPU timing tasks and CPU-prior task ended through cooperative
checkpointed yields. They are partial production fits, not completed result
panels. Slurm reports `FAILED` for exit 75; their explicit `ANALYSIS YIELDED`
messages establish intentional continuation. Do not treat every exit 75 as safe
without its log/checkpoint evidence. No full nested decoder fit has yet supplied
an end-to-end runtime or final-output storage measurement.

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

Local implementation started from HEAD `39668387b0510004d5e4f4dfe6c40a23565d5ab1`.
**Protocol 2 includes uncommitted changes. Cloning that HEAD alone is insufficient.**
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

The following identities describe the **previous verified cluster installation**,
not the current working tree. The accepted semantic build itself is unchanged.

| Identity | Previously installed value |
|---|---|
| Accepted semantic build | `9bce4af5463a57683ed375a8b0fdcbcb83620bce923624d7075bd0c4f4ac847f` |
| `artifacts/analysis-source.zip` SHA256 | `887975589fdb38a2fd626b5556551319436504e1cc8cebdc7e58a358929cb0e7` |
| Isolated analysis code directory | `analysis_code/87dbb953cf22ecba8dab1004d4290760f65ab21c3d38e4033a9a6fcdd73d21ea` |
| Development manifest hash | `0febeb5b61465fd9e0311e2e7deca3f95bbdcb5da2dc485079270ccef4e2ed9d` |
| Analysis configuration hash | `77dafd1fdc55ab4502b390a385671304a956c1335cc1bec3da5b45b4681bcedd` |
| Compute configuration hash | `8805b3651a715e568305151fe3a9e46c50e77b5e8389cbc7bf5526bd2636eea2` |
| Experiment definition hash | `b90d89f9131e821e133ad3f43965a789008af1ee79eb1f1b27bb56f2ba068596` |

That ZIP had 258 payload files and 89,708,425 bytes, as recorded in the previous
handoff. The isolated code directory is packaged in the ZIP and installed
on the cluster; it is not an unpacked local directory. The manifest is
`artifacts/execution/<manifest-hash>/manifest.json`, also identified by
`artifacts/execution/latest.json`.

Cluster `.analysis-source.json` selects installed code. Editing local source does
not update that installation. Code/configuration changes require fresh packaging,
identities, and transfer. Never relabel old checkpoints or receipts to force
compatibility with new training code. Preserve them under their original identity.
These optimization changes do not require reannotation or frozen-model extraction.

## Implemented locally; verification boundary

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
remain researcher-run requirements; old receipts do not certify protocol 2.
Additional targeted checks passed trace/faithfulness export recovery on two
complete real sources, streaming/compact geometry on 16 supported concepts,
equal-story row scaling, and development semantic coverage (ten stories,
eleven query families). The latter is a support audit, not annotation accuracy.

Cluster `.venv-analysis` has PyTorch `2.10.0+cu128` (CUDA runtime 12.8).
Existing `.venv` and `.venv-extraction` are separate environments. Setup/preflight
confirmed all 18 response files at expected sizes and five aligned model receipts,
22 metadata-verified files each: Qwen 3.5 9B base/post, OLMo 3 7B base/instruct,
and Qwen 3.8 27B. This checks metadata/provenance, not a fresh full tensor rehash.

## Latest cluster evidence

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

Latest allocated-node disk measurements: dataset 25 GiB, frozen features 47 GiB,
Hugging Face model cache 115 GiB, analysis cache 2.7 GiB, and analysis outputs
363 MiB: about 190 GiB across those paths, excluding environments and other
directories. The shared lab filesystem previously had 8.9 TiB free; that is
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

1. Run the integrated local command in Windows CMD:
   **call scripts\prepare_analysis.cmd**. It verifies the complete accepted
   corpus, current protocol/numerics/recovery/inventory and packages the bundle.
   A pre-existing ZIP is not evidence that this command passed.
2. After success, transfer the new ZIP, installer, update helper and generated
   transfer receipt. Inspect active project jobs first, then install only through
   the allocated-node update helper. Reuse the existing analysis environment and
   frozen model features. All paths remain in the project workspace.
3. Run updated CUDA numerical qualification, then complete representative
   production selection/refit and derived panels with the new manifest.
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
