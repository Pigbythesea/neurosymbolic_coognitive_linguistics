# Engineering handoff

Updated 2026-10-07. This is the engineering entry point for a colleague taking
over the current working tree. Scientific framing and protocol decisions belong
to the separate scientific thread; see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).
Remote facts below come from researcher-supplied logs, not direct cluster access.

## Current position

**Data, model features, the analysis environment, and the latest verified code
are on the cluster. The full experiment has not been launched.** The latest
optimization batches queries within each source passage and substantially
improves runtime. It still takes hours to tune an ordinary decoder condition.
The immediate task is a coordinated execution redesign before expanding the grid.

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

Local branch is `main`, HEAD `c9543802ec05b3cd307259bdca0f73256bfa7d3b`.
**The verified implementation includes uncommitted changes. Cloning HEAD alone
is insufficient.** Preserve the current working tree and verified bundle.

Modified tracked files at handoff creation:

```text
configs/compute.json
docs/CLUSTER_STATUS.md
docs/compute.md
neurosym/analysis_runs.py
neurosym/decoder_fit.py
neurosym/decoders.py
neurosym/execution.py
scripts/package_analysis.py
scripts/verify_analysis.py
scripts/verify_compute.py
```

New implementation/helper files not yet tracked:

```text
neurosym/decoder_batch.py
scripts/benchmark_prior.py
scripts/benchmark_prior.sbatch
scripts/update_analysis.sbatch
```

This handoff is also newly created. In particular, `decoder_batch.py` is required
by the updated executor, not an optional benchmark.

| Identity | Current value |
|---|---|
| Accepted semantic build | `9bce4af5463a57683ed375a8b0fdcbcb83620bce923624d7075bd0c4f4ac847f` |
| `artifacts/analysis-source.zip` SHA256 | `887975589fdb38a2fd626b5556551319436504e1cc8cebdc7e58a358929cb0e7` |
| Isolated analysis code directory | `analysis_code/87dbb953cf22ecba8dab1004d4290760f65ab21c3d38e4033a9a6fcdd73d21ea` |
| Development manifest hash | `0febeb5b61465fd9e0311e2e7deca3f95bbdcb5da2dc485079270ccef4e2ed9d` |
| Analysis configuration hash | `77dafd1fdc55ab4502b390a385671304a956c1335cc1bec3da5b45b4681bcedd` |
| Compute configuration hash | `8805b3651a715e568305151fe3a9e46c50e77b5e8389cbc7bf5526bd2636eea2` |
| Experiment definition hash | `b90d89f9131e821e133ad3f43965a789008af1ee79eb1f1b27bb56f2ba068596` |

The ZIP has 258 payload files and 89,708,425 bytes. Its SHA was rechecked for
this handoff. The isolated code directory is packaged in the ZIP and installed
on the cluster; it is not an unpacked local directory. The manifest is
`artifacts/execution/<manifest-hash>/manifest.json`, also identified by
`artifacts/execution/latest.json`.

Cluster `.analysis-source.json` selects installed code. Editing local source does
not update that installation. Code/configuration changes require fresh packaging,
identities, and transfer. Never relabel old checkpoints or receipts to force
compatibility with new training code. Preserve them under their original identity.
These optimization changes do not require reannotation or frozen-model extraction.

## Implemented and verified

- [decoder_batch.py](../neurosym/decoder_batch.py) compiles public query inputs
  into packed constants and batches descriptor, candidate, primitive, binding,
  and composition calculations **within one source**. Supervision remains separate.
- [decoder_fit.py](../neurosym/decoder_fit.py) still performs **one AdamW update
  per source**, roughly 800 updates per epoch. There is no cross-source minibatch
  or vectorized independent-fit training yet. Source order, weighting, candidate
  order, repeat averaging, and model parameters are preserved.
- Decoder arithmetic remains FP32; PCA/ridge use FP64; TF32 is disabled.
  Existing training settings remain hidden size 64, maximum 80 epochs, patience
  10, learning rates 0.0003/0.001, and seeds 11/29/47.
- Packed query constants have a 128 MiB / 2,048-program per-model memory limit.
  Learned values are recomputed after updates. Actual fits reach this cap;
  eviction/recompilation is a candidate bottleneck, not yet profiled attribution.
- Untraced validation uses batching. Detailed held-out trace export retains the
  scalar reference path. Epoch logs separate training and validation timing,
  but **exclude the checkpoint write that follows the epoch**.
- PCA merges training-only per-story centered moments; GPU statistics/projection
  and a CPU partial eigensolver are the current hybrid choice. Ridge chooses
  primal/dual solves and reuses compatible designs, spectra, and response moments.
- Encoding saves response operators instead of enormous dense voxel prediction
  arrays. Identical compatible fits can share outputs. Geometry has GPU product
  and permutation paths, but its complete downstream schedule still needs to be
  included in end-to-end execution planning.

The researcher completed all six stages of `scripts\prepare_analysis.cmd`:
accepted-corpus/compiler checks, temporal support, real-query/experiment checks,
compute equivalence/recovery, inventory, and checked packaging. CPU maximum
batched logit/gradient/weighted-loss differences were approximately
`2.623e-6 / 2.146e-6 / 9.537e-7`; resumed checkpoint parameter difference was zero.
CUDA verification also passed. Verification uses actual accepted queries and
released features; tiny verifier epochs are not production runtime estimates.

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

The current development manifest contains 25,146 logical items grouped into
3,362 workers: 30 priors, 348 preparation items, 3,654 decoder fits, and 21,114
encoding fits. Expected final-stage counts add 3 priors, 29 preparation items,
348 decoders, and 2,106 encodings. Combined decoder count is 4,002 and encoding
count 23,220. These are work items, not simultaneous GPUs or required job IDs.
Complete geometry, comparisons, statistics, and reporting are additional scheduling
work; fit-manifest completion alone is not project completion.

An ordinary current decoder fit performs nine inner folds times two learning
rates, then a refit. Assuming 12–20 epochs per candidate, measured rates imply
roughly 2.6–4.4 hours for linear selection and 6.3–10.6 hours for structured
selection, before remaining fit/export costs. Earlier 15–60 minute fit estimates
are superseded. A prior conditional full-grid extrapolation was about 100–160
days on eight continuously occupied GPUs; it is not a measured end-to-end run
and assumes epoch counts and unmeasured family costs.

**The goal is several days on eight GPUs; this is not yet achieved or forecast
with confidence.** The proposed five-day engineering budget is 960 GPU-hours:
600 decoder, 230 encoding, 70 preparation/geometry/statistics, 60 headroom.
It requires about nine GPU-minutes per complete decoder fit, amortized across
concurrent fits. The encoding allocation extrapolates only one nine-participant
panel (5:16, about 40 GiB host RAM), so it needs broader measurement. Queue,
engineering, and human review time are outside this compute budget.

## Storage and cleanup

Latest allocated-node disk measurements: dataset 25 GiB, frozen features 47 GiB,
Hugging Face model cache 115 GiB, analysis cache 2.7 GiB, and analysis outputs
363 MiB: about 190 GiB across those paths, excluding environments and other
directories. The shared lab filesystem previously had 8.9 TiB free; that is
not a personal quota or reserved capacity.

Full development/final encoding operator-array arithmetic is about 1.036 decimal
TB before masking, compression, and identical-fit sharing. It excludes decoder
weights/traces, feature vocabulary metadata, geometry, and caches. The previously
discussed 2–3 TB workspace allowance is capacity planning, **not a measured size
or validated upper bound**. Complete representative outputs are still needed.

The disk analysis cache has a 128 GiB admission limit, 1 GiB entry reservations,
and a 16 GiB minimum-free-space guard; it is not automatic disk LRU cleanup.
Host/device/raw resident reuse budgets are separately 4/4/16 GiB. Process exit
releases RAM/VRAM, **not downloaded files or saved outputs**. Some intermediate
checkpoints are removed after completed fitting stages; no broad cleanup is
automatic. Do not delete caches, reservations, or old outputs without checking
active ownership and the reproducibility/continuation requirements.

## Next engineering work, in order

1. Preserve this working tree and bundle. Use the current measurements as the
   baseline; do not launch or repeatedly resume the full old decoder grid.
2. Implement one coordinated performance pass: precompile/reuse packed constants
   against the actual working set, remove avoidable host synchronization, improve
   batched tensor execution and trace export, and batch independent fits where
   useful. Independent fits must retain separate parameters, optimizer states,
   random streams, clipping, and stopping decisions. Profile to establish where
   runtime is spent rather than assuming all unused VRAM buys speed.
3. Obtain protocol decisions from the scientific thread before enabling either
   **cross-source minibatches** or **three whole-story inner folds**. Both were
   proposed, neither is implemented/adopted. Minibatching changes optimizer
   trajectories and convergence; three folds changes tuning support. Do not
   report either as a numerically identical execution optimization. An alternative
   preserving current splits is sharing identical training trajectories across
   swapped outer/inner story pairs, with separate validation/stopping histories;
   this also remains unimplemented and has no guaranteed twofold gain.
4. Finish the execution/storage plan: compatible encoding reuse, compact indexed
   trace outputs, all downstream geometry/statistics/reporting tasks, global
   eight-GPU admission, organized arrays, safe resumption, and per-fit time/byte
   accounting. Existing per-stage concurrency settings do not enforce that global
   cap. Multiple array tasks each requesting a GPU do not share one allocation;
   concurrent fits within one GPU require an explicit execution design.
5. Verify exact optimizations against the reference on real inputs. For any
   approved training change, verify its objective/weights and convergence
   separately. Package once the integrated implementation is ready; the researcher
   runs long verification, transfers it, and installs on allocated compute.
6. Qualify complete representative fits and concurrent throughput on the cluster,
   retaining eligible production work. Include tuning, refit, checkpoint/export
   time, geometry, and final output sizes. Then set walltime/concurrency and launch
   the full schedule using the measured budget. Several-day completion remains
   an open engineering requirement until those measurements support it.

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
