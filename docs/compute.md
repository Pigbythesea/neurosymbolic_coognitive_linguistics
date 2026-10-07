# Analysis computation and cluster handoff

Updated 2026-10-06. Execution implementation within
[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) and
[experiment_definitions.md](experiment_definitions.md). The runtime pass preserves
the accepted annotations, candidates, temporal alignment, train/test partitions,
full voxel panel, model/layer inventory, losses, source/story weights, nulls,
hyperparameter grids, seeds, epochs and early-stopping rules.

## Current verification and next command

The compiled build is readable after the researcher's Windows ACL repair.
Python 3.11 syntax and configuration parsing passed during implementation.
A short calculation on released Deniz letter features compared adaptive primal
ridge against the reference dual solve: maximum operator difference
5.134781488891349e-16. This is a numerical identity check, not a brain result.
The corresponding direct-versus-sufficient-statistic selection loss check differed
by at most 5.273559366969494e-16; a wider actual feature matrix selected the dual path.
The final-pass short PCA check used actual released English1000 columns from
stories 01/02: merged centered moments versus concatenated fitting gave a maximum
subspace difference of 2.886579864025407e-15 and zero FP32 projection difference.
This check covers CPU arithmetic on those inputs, not CUDA or neural recordings.

**The full updated verification, packaging and CUDA checks remain pending.**
The researcher runs long commands; the agent has not started cluster jobs,
connected remotely, recovered a lock, or transferred this implementation.

Run in local Windows CMD:

```cmd
cd /d C:\Users\pigby\neurosymbolic_coognitive_linguistics
call scripts\prepare_analysis.cmd
```

This performs complete annotation/compiler/support checks, analysis verification,
compute equivalence and packaging. It logs to
`artifacts\prepare-analysis.log`; success is `ANALYSIS PACKAGE READY`.
An existing ZIP or earlier receipt does not certify these changes. Packaging
rejects mismatched source, configuration, build and verification identities.

The extended compute verifier uses actual released textual arrays and accepted
queries. It checks primal/dual operators, each group contribution, compact
serialization, target metrics, both ridge selection formulations, dense versus
factorized decoder logits/gradients/routing, within-observation reuse, validation
transfers, prior sharing, cache budgets/invalidation, checkpoint recovery,
CPU/CUDA RDM products and permutation tails, and complete scheduling coverage.
Ridge tolerances are 1e-8; FP32 decoder logits/routing 2e-5 and gradients 3e-5.
CUDA checkpoint comparisons allow 3e-5 and must retain the selected epoch.
These checks do not substitute text arrays for missing scientific fMRI/model inputs.

## Where computation belongs

The current official [Skipjack partition guide](https://docs.arch.jhu.edu/en/latest/1_Clusters/Skipjack/2_Slurm/Partitions.html)
was checked on 2026-10-06. It lists CPU `med`, `l40s`, `a100`, `h100`,
`h200` and `rtx6000`. H100 SXM and NVL share `h100`; RTX PRO 6000 is
`rtx6000`. **B200/B300 are excluded by the researcher's instruction.**
The documentation's inventory/capacity figures are not fully consistent across
pages, so use live partition/node reports before scheduling. Published billing
ratios are marked as placeholders; no dollar-cost ranking is inferred here.

| Work | Execution | Reason |
|---|---|---|
| Extraction of frozen model states | Existing GPU pipeline | Large transformer forward passes; preserved separately. |
| Fold-local PCA and observation windows | GPU moments/projection, CPU top-component eigensolver by default | Cache centered moments per story; merge only training stories. Batched ragged parcel products/projections use FP64 CUDA. Full device eigendecomposition is selectable. |
| Ridge encoding | FP64 CUDA on A100/H100/H200, or explicit CPU path | Large sufficient-statistic products, eigendecompositions, response operators and voxel metrics. |
| Learned decoders | FP32 CUDA; L40S/RTX PRO 6000 also eligible | Differentiable fitting with actual observations resident on device. No AMP or TF32. |
| Query-only priors | CPU, shared fits | No observation/PCA dependence; substantial query execution overhead. |
| Geometry | Explicit CPU or CUDA | GPU products/reconstruction and batched RDM permutations; CPU labeling, aggregation and rank construction. Small cases can stay on CPU. |

Moving every operation to a GPU is not the optimization target. Transfers,
Python overhead, and small irregular work can dominate. Dense FP64 work and
resident differentiable operations have GPU paths; preparation overlaps fitting.
No distributed training or multi-GPU requirement has been introduced. GPU profiles
request multiple compatible partitions, never a mandatory individual GPU model.
The CPU eigensolver is a selectable hybrid implementation, not a measured claim
that it beats device eigendecomposition. Compare preparation timings on cluster.

## Implemented calculation and reuse changes

- **Local PCA:** `pca.py` caches per-story counts, means and centered scatter
  matrices. Stable merging reconstructs a fold's training-only covariance;
  heldout statistics are never included. Constant-coordinate thresholds, ranks,
  signs, parcel assignments and eight-component limits are retained. GPU batches
  pad only internal matrices, then discard padded coordinates. Statistics and
  projections use FP64; stored windows retain the existing FP32 format.
  Actual assigned parcel sizes: median 157, range 21–479. Dense FP64 scatter
  arrays for ten development stories/all nine participants total about 4.7 GiB
  before metadata. CUDA moments/projection and the eigensolver are configured
  independently through `preparation`; CPU preparation remains available.
- **Adaptive ridge:** solve the feature-space system when active feature width
  is smaller than training-row count, otherwise solve the sample-space system.
  The estimator and penalty remain identical. Primal selection projects the
  response moments into the smaller basis. Final prediction uses the cheaper
  matrix-product association according to actual dimensions, retaining FP64.
- **Shared encoding work:** normalized designs, kernels and mixture spectra use
  bounded resident caches. Related participant fits run in one worker and visit
  an inner fold across the panel before the next fold. Compact fold/grid score
  files avoid repeating that selection work during final per-condition execution.
- **All-voxel moments:** disk-cache training-normalized response Gram/cross
  products by participant, exact ordered stories, actual retained rows,
  target batch, backend and input/code identities. These expensive response
  products are independent of semantic feature groups. FP64 CUDA moments can be
  reused across verified CUDA hardware under the same library/input identity;
  each entry saves the producing device in `producer.json`. CPU moments remain
  separate. This permits floating-point equivalence, not a cross-GPU bitwise claim.
- **Decoder algebra:** scalar binding contracts low-rank factors directly,
  including the original invalid-site penalty. Composition still constructs the
  dense row-softmax relation matrix because the nonlinearity requires it.
  Site projections, primitive scores and factors are reused only within the
  same observed source/repeat and optimizer step; learned values are cleared
  before another observation/update.
- **Query overhead:** batch descriptor encodings, reuse immutable descriptor
  serialization and token packs, reuse training-only vocabularies and projected
  windows. Query-only priors share identical fits while keeping each logical
  condition's scoring/provenance.
  Public query syntax is frozen and compiled once: validation, binding operations
  and relation descriptors are reused. Learned vectors remain source/update-local.
- **Transfers and I/O:** stage actual decoder windows once per fit; transfer
  validation scalars per source/repeat. Encoding matrices stay on the selected
  device through products and metrics. Trace public inputs reference the pinned
  compiled query/catalog rather than repeating its candidate JSON in every trace.
  Matched/mismatched prediction JSONL is losslessly gzip-compressed at level 1;
  `complete.json` lists the filenames and all probabilities remain present.
  Identical arrays within each heldout source/repeat share HDF storage through
  hard links; every query's original trace paths and membership remain intact.
  CUDA target normalization stays on device together with response products.
- **Geometry:** optional CUDA FP64 cosine products, compact encoding
  reconstruction, and batched ranked-RDM permutations. CPU generates the same
  permutation sequence; near-tail ties use the reference statistic. Native
  source preprocessing can be reused within a shared AnalysisData session.
  No item-label shuffle is replaced with independent RDM-cell shuffling.
  `geometry-panel` accepts a JSON list of complete geometry option dictionaries;
  it shares native source vectors, encoding-implied vectors and latent indexes.
  Grounding extraction reads primitives/factors once per trace and reuses relation
  products across items. RDM normalization also stays on CUDA when requested.
- **Recovery:** decoder checkpoints include optimizer, RNG, best validation state,
  patience and partial-epoch source order/cursor/losses. Cooperative yield occurs
  after a completed source update. Encoding flushes a completed-voxel cursor and
  resumes metric blocks; completed selection/cache entries are reused. Preparation
  yields between published entries. An individual covariance/eigensolve/cache
  build is not preemptible; whole heldout trace passes are not checkpointed.
  The advance signal/reserve must exceed these uninterrupted operations, as checked
  by real cluster timings. Hard kills can still require artifact inspection.
  New analysis locks are OS-owned and released on process death; legacy presence
  locks still stop for explicit inspection. No extraction lock behavior changed.

## Compact encoding output contract

Every logical condition retains its own identity, support declaration, nested
selection, completion receipt and interpretation. Its `artifact.json` points to
`data/analysis/encoding-objects/<numerical-identity>/encoding.h5`.
Identical selected fits can share this object even if logical seed/contrast names
differ. Sharing requires equal participant, train/test row identities, ordered
feature groups, fitted-model source, selected regularization and numerical backend.

The HDF stores FP64 heldout response operators H_g for each feature group,
FP64 training voxel means, training/test row indices and all voxel-level metrics.
For centered pinned training responses Y, the group contribution is H_g Y;
the total prediction is sum_g(H_g) Y plus the stored mean. Readers reconstruct
voxel batches without a new fit or hyperparameter search. Geometry and paired
encoding comparisons resolve and validate shared artifacts; legacy dense files
remain readable. Feature vocabularies/scalers and selected settings are retained.
Analysis on new stimuli or fitted coefficients would require rebuilding that
selected ridge estimator from pinned inputs, not reading a saved weight matrix.

For the current development definitions, metadata-only full-row arithmetic gives:

| Encoding arrays only | Uncompressed ceiling before masks/compression/sharing |
|---|---:|
| Previous duals + predictions + all group contributions | 36,533,910,825,360 bytes (36.53 TB) |
| Current operators + means + metric arrays | 952,894,714,032 bytes (0.953 TB) |

This is about **97.4% less in that array bound**, not measured disk usage or a
97.4% reduction of the entire project. Decoder weights/traces, cached moments,
geometry, metadata, source datasets and frozen states are additional. The
manifest recomputes these bounds from the actual configured inventory.

## Storage lifecycle

`configs/compute.json` separates storage/execution policy from scientific settings.

- Persistent intermediates: `data/processed/analysis-cache-v2`, **128 GiB cap**,
  **1 GiB maximum per entry/staging reservation**. Concurrent reservations are
  accounted for with an OS-released lock and an atomic ledger.
- Retained numerical host cache: 4 GiB; retained CUDA cache: 4 GiB per device;
  raw observation cache: 16 GiB per worker, all with LRU eviction.
  These bound retained cache payloads, not total RSS/VRAM: active working arrays,
  Python metadata, allocator reservations and library workspaces are additional.
- Keep a 16 GiB filesystem free-space reserve for cache publication/output
  checks. This cannot certify the PI/project quota, and concurrent output writes
  can consume space between checks. The output artifacts do not have a claimed
  total-size cap. Long trace writes periodically check free space.
- A killed cache writer leaves its reservation and staging path for inspection.
  No source data, fit, prepared dependency or old cache is automatically removed.
  Cache-full is an explicit stop, not an unreported eviction/recomputation.
- Cache receipts hash every payload. Completed caches publish atomically.
  Per-entry OS locks prevent duplicate first-writer computation. Lock files are
  never unlinked; validate cross-node advisory locking on the actual filesystem.
  Persistent moments/scores/projectors/windows are regenerable; accepted
  annotations, raw recordings, frozen-state provenance and fit evidence are
  retained research inputs/results. Compact predictions require the pinned
  training responses to remain accessible.
- Retire an obsolete cache generation only after every job consuming it is
  inactive and after retaining the relevant provenance. A queued worker with
  `require_prepared` still needs its prepared cache. Reclaiming space is
  deliberately not an automatic side effect of fitting.

Local read-only status in CMD:

```cmd
.venv-analysis\Scripts\python.exe -B scripts\run_analysis.py cache-status
```

Append `--scan` to total cache/build/fit/artifact files; this may take time on
shared storage. The command does not modify the ledger or delete anything.
Configured caps are software budgets, not claims about available PI allocation.
The documented [Skipjack storage guide](https://docs.arch.jhu.edu/en/latest/1_Clusters/Skipjack/Quickstart.html)
does not establish a separate personal quota for this repository. No assumed
node-local scratch path is used: all environments, caches and temporary output
remain under the inspected project workspace.

## Workers, resources and submission

Current planning requests (not measured minima or certified admission limits):

| Profile | Request | Maximum simultaneous workers |
|---|---|---:|
| Optional CPU preparation | med, 8 cores, 64 GiB, 1 h | 8 |
| GPU preparation (default) | a100/h100/h200, 1 GPU, 11 CPU cores, 64 GiB, 1 h | 2 |
| CPU priors/optional decoders | med, 4 cores, 32 GiB, 4 h | 8 |
| GPU encoding | a100/h100/h200, 1 GPU, 11 CPU cores, 64 GiB, 4 h | 4 |
| GPU observed decoding | l40s/rtx6000/a100/h100/h200, 1 GPU, 6 CPU cores, 32 GiB, 4 h | 4 |
| Explicit CPU encoding alternative | med, 8 cores, 128 GiB, 4 h | 8 |

Profiles are independent: the three GPU profiles together permit ten GPU workers.
Encoding retains large raw response matrices; prepared decoding only loads its
projectors/windows and public queries, so its host-memory request is smaller.
The worker limit is up to nine related logical fits per process, preserving
per-fit receipts. These are resumable allocation windows, not fit duration
estimates. Initial compatibility/timing work should request 30 minutes, with
multiple eligible partitions. Use `--time HH:MM:SS` on the submission helper to
override a dispatch's walltime without changing fit identities. Set production
requests from measured timings, including initialization and uninterrupted I/O.
Panel size, partition eligibility and
concurrency may be adjusted as execution settings without removing scientific fits.
CPU/memory requests and account/queue limits still require live inspection.

The version-2 manifest under `artifacts/execution/<hash>/manifest.json` retains
all logical jobs and separately records physical workers. It includes development
folds, seeds, models/layers, decoder nulls and multistory-geometry fitted parents.
Final evaluation requires a separate final manifest and explicit release.
Geometry/comparison/report commands consume completed parents; the manifest does
not invent new all-pairs contrasts or choose compositional holdout keys.

The submission helper defaults to a read-only dry run. On explicit researcher
submission it dispatches only workers whose actual preparation receipts are complete.
Pending/running workers count against each resource's concurrency cap for this
manifest. Inspect older active manifests before starting another. Ready
work has no artificial serial-lane predecessor. Each call submits a ready wave;
`--watch` keeps dispatching on the researcher's allocated controller node.
`--submit --resume` reopens its durable ledger after the controller exits, with
the same `--worker` selection if one was used. Selected worker IDs permit real
cluster timing without changing their scientific fit definitions.

The worker receives `B:USR1` three minutes before its walltime and also checks a
local deadline. Clean yields exit 75 and publish an attempt token; `--watch` can
continue these in another allocation. Actual failures/timeouts stop dispatch
for diagnosis; they are not silently resubmitted. Already submitted jobs are not
cancelled by a controller error/exit. Two yields without recorded progress stop
for inspection. Never reduce the reserve below measured uninterrupted operations.
An uncertain `sbatch` outcome remains marked `submitting`, requiring scheduler
inspection instead of blind replay. No automatic legacy-lock recovery occurs.

Workers validate the immutable code/manifest once per process and retain loaded
inputs across their fits. Every CUDA type used must have a current matching
numerical-verification receipt. By default a newly assigned GPU runs the verifier
on that allocation and publishes a receipt before fitting; a per-device lock avoids
duplicate verification. GPU verification is part of initial allocation time.
Do not restrict eligibility to one model just to match an old receipt. An RTX
extraction log alone is not a numerical receipt. Access remains account-dependent.

Runtime files and decoder epoch histories record elapsed phases, peak RSS and
CUDA allocated/reserved memory. Worker records under
`data/analysis/execution/<manifest>/workers/` include shared selection time;
per-fit timings alone omit that earlier panel work. Use completed full-scope jobs to estimate
remaining device hours and calendar time separately. No measured GPU speedup
or defensible whole-program hour count is established yet.

## Remote handoff

The researcher owns all remote operations:

1. Finish local verification/package creation.
2. Recheck [CLUSTER_STATUS.md](CLUSTER_STATUS.md), project free space/quota,
   submission limits and partition availability before transfer. Its historical
   account/QoS and stale Qwen alignment evidence remain useful; timestamps and
   old extraction completion records do not certify current analysis readiness.
3. Install the verified isolated analysis bundle on an allocated node. Preserve
   raw data and extraction caches; this pass did not change `model_features.py`
   or `temporal.py`.
4. Resolve the documented Qwen alignment-only interruption with the researcher's
   explicit instruction, using the existing compatible extraction implementation;
   verify actual fMRI files and all aligned states.
5. Request a short multi-partition GPU allocation, run packaged numerical
   verification, check OS locking on project storage, and time selected actual
   preparation/encoding/decoder work. Verify interruption/continuation and inspect
   GPU utilization, peak RSS/VRAM, cache growth and uninterrupted phase durations.
   Full neural arrays and hidden states are available there; local numeric checks
   cannot certify those I/O paths or predict their runtime.
6. Set resource requests and walltimes from the measurements, then dispatch the
   complete development manifest. Geometry panels consume the fitted parents;
   final-story evaluation uses the separate explicitly released final manifest.
   The researcher runs all commands. No setup or computation belongs on login nodes.

The implementation is ready for those validations; it is not a claim that the
full cluster experiment has already run or is scientifically validated.
