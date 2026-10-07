# Cluster execution status and transfer handoff

For a consolidated current engineering state and next-work checklist, start with
[ENGINEERING_HANDOFF.md](ENGINEERING_HANDOFF.md). This file retains the detailed
history, including superseded pending actions and runtime estimates.

**Updated:** 2026-10-07 (researcher's America/New_York date).
**Current instruction:** The runtime implementation thread owns the human-agent
cluster measurement and full-experiment workflow. Local preparation, transfer,
cluster setup, CUDA numerical verification and the first three real GPU workers
have completed; see the latest evidence below. The researcher performs every
remote operation in their persistent authenticated session. This note records
user-supplied terminal evidence, not a direct agent inspection of the cluster.
Earlier sections retain historical failures and pending actions as provenance;
the latest evidence supersedes their status statements.

## Latest update: source batching passed local preparation (October 7)

### Latest actual decoder timing array (researcher snapshot)

Final researcher accounting for array 1020081: tasks 0/1/2/3 all cooperatively
yielded with explicit ANALYSIS YIELDED logs and exit 75:0. Elapsed durations were
26:47 / 26:46 / 26:47 / 26:45; TotalCPU about 26:20 each. MaxRSS was
3,686,352 / 3,033,164 / 6,161,416 / 3,214,948 K. Linear tasks reached inner fold
2/9; structured tasks remained within inner fold 1. These are retained partial
fits, not completed scientific panels. The four timing allocations have ended;
no further old-executor continuation is recommended before the coordinated
redesign. This does not claim the entire user queue was independently checked.

The researcher now explicitly targets the complete experiment within several
days at eight-GPU scale, retaining scientific scope/rigor. Proposed roadmap
budget (not a runtime forecast): five continuously supplied compute-days =
960 GPU-hours, allocating 600 to all roughly 4,002 development/final decoder
fits, 230 to encoding (previous single-panel extrapolation), 70 to preparation,
geometry/statistics and 60 to operational headroom. The decoder throughput
target is about nine GPU-minutes per complete logical fit, including tuning,
refit and export, amortized across concurrently executed independent fits.
Queue delay, human review and engineering time are separate; three days is a
stretch target with a substantially smaller available decoder budget.
Three whole-story inner folds and source minibatching remain proposed protocol
changes; immutable annotations, all outer evaluations, participants/models/layers,
seeds and controls are retained. No analysis configuration or implementation
was changed by this roadmap discussion.

Subsequent live inspection at about 11:31 confirmed distinct assigned GPU UUIDs
for linear task 0 (numeric job 1020083) and structured task 1 (1020084).
Linear samples: SM activity 5-7%, 1,483 MiB framebuffer, about 75 W.
Structured samples: SM activity 0-17%, 1,043 MiB framebuffer, about 72 W.
These are five-second samples, not full-run utilization or an attributed profile.
sstat CPU time was 11:02/11:08; accounting is consistent with roughly one busy
CPU core near this snapshot, not six busy allocated cores. Warm linear epochs
remain about 42-47 s; structured epochs about 105-109 s. No numerical error,
completed nested fit or completed array allocation is established by this update.

The researcher asks for compute/design changes because hundreds of days is
unacceptable within the 4-8 GPU ceiling. Recommendations under discussion, not
implemented or adopted: source minibatching with unchanged story/source/query
weighting but explicitly revised optimizer trajectories; three whole-story inner
folds while retaining every outer heldout story and all scientific conditions;
precompiled resident query programs, compiled tensor execution, and optionally
vectorized independent fits. Learning rates/stopping must be appropriate for
the revised optimizer; fewer steps alone cannot be called equal-convergence
speedup. Three inner folds changes tuning/training support and must be versioned.
An alternative retaining current leave-one-story inner selection can share
identical training trajectories across swapped outer-test/inner-validation story
pairs: actual split metadata has 90 inner training lists but only 45 distinct
ordered lists per condition/seed/rate. Separate validation selection and stopping
would still be required; this is not an extra guaranteed 2x on top of three-fold
selection. No scientific split or training configuration changed in this review.

Array **1020081**, tasks 0-3, was RUNNING at 6:02, all on gr101/rtx6000.
Each task requested one GPU and identified an RTX PRO 6000 Blackwell Server
Edition with 97,887 MiB. Task mapping is 0->logical31 (brain linear),
1->33 (brain structured), 2->3895 (model linear), 3->3897 (model structured).
No completed nested fit or new utilization sample has been supplied yet.

- Brain linear: cold epoch 57.54 s; epochs 2-5 43.81-45.95 s.
- Brain structured: cold epoch 109.26 s; epoch 2 106.37 s.
- Model linear: cold epoch 51.84 s; epochs 2-5 42.89-44.11 s.
- Model structured: cold epoch 111.76 s; epoch 2 104.75 s.
- These actual-data speedups are substantial versus the old executor, but
  ordinary fits retain 18 inner training runs plus a refit. At 12-20 epochs per
  inner run, selection alone would take about 2.6-4.4 h for linear and 6.3-10.6 h
  for structured, excluding checkpoint I/O, setup and final outputs. The earlier
  15-60 minute per-fit scenarios are not supported by these rates. Full-grid
  expansion remains deferred pending attribution of remaining runtime; the
  nearly identical brain/model epoch durations do not identify a bottleneck.
- Prior **1019984** ended at 26:49, exit 75:0, MaxRSS 9,171,892 K (~8.75 GiB).
  It reached inner 4/9, lr 0.0003, completed epoch 9, then explicitly logged
  ANALYSIS YIELDED. This is a checkpointed continuation, not a completed panel
  or a numerical exception. No continuation has been submitted by the agent.

Actual allocated-node du output: dataset 25G, frozen features 47G, model cache
115G, analysis cache 2.7G, analysis outputs 363M. Rounded sum is about 190 GiB
for these directories, excluding environments and other workspace directories.
The measured frozen-feature occupancy supersedes the uncompressed 117 decimal
GB input-array arithmetic as the current on-disk size. Future trace/output
storage remains unmeasured; 2-3 TB remains an allowance, not an observed total.

Latest researcher evidence after the CUDA check:

- Preparation job **1019982**, logical index 30: completed on ga131 in 25 s,
  exit 0:0, MaxRSS 1,372,812 K. Job **1019983**, logical index 3894: completed
  on ga131 in 24 s, exit 0:0, MaxRSS 2,418,756 K. Both published dependencies
  under the new 0febeb5b... manifest; the second log includes device verification.
- CPU prior **1019984**: running on csr056 at the latest 12:34 snapshot,
  inner fold 2/9, second learning rate just started. Warm epochs remain about
  17.3-17.5 s. The zero TotalCPU/MaxRSS fields in running-job sacct output do
  not establish zero actual resource usage. Completion has not been supplied.
- Disk-cache ledger: 2,819,703,476 published bytes (2.63 GiB), 392 entries,
  one 1 GiB reservation explicitly owned by active job 1019984. This reservation
  is not a stale lock or another GiB of confirmed payload. df reports 8.9T free
  on shared lab storage, not a separate personal quota.
- Researcher preference: 4-8 concurrent GPUs maximum, grouped arrays to keep
  queue organization compact. Running array tasks remain individually visible.
  Next proposed timing is a single four-task array over logical indices
  31/33/3895/3897, one GPU each, at most four concurrently. No launch by agent.

Storage arithmetic from pinned local metadata, not remote measured disk usage:
model download files 123,456,349,577 bytes; unaligned FP32 feature arrays
86,899,982,336 bytes; aligned FP32 feature arrays 29,684,072,448 bytes; dataset
26,525,553,107 bytes. Sum is about 0.267 decimal TB before feature compression
and filesystem/environment overhead. The development encoding-array ceiling is
952,894,714,032 bytes; applying the same full-row formula to the 2,106 final
encoding conditions gives 82,740,207,096 bytes, for 1.036 decimal TB combined.
These are operator/mean/metric arrays only, before masks/compression/sharing;
they exclude feature vocabularies, decoder outputs, traces, geometry and metadata.
A provisional 2-3 TB total workspace allowance is capacity planning, not a
validated upper bound; complete representative decoder outputs are still needed
to extrapolate trace and metadata storage. No disk policy or fit code was changed.

The researcher ran all six stages of `prepare_analysis.cmd`, ending in
ANALYSIS PACKAGE READY. The agent verified the resulting ZIP inventory, every
payload hash/CRC and all current isolated-code input hashes. The unchanged
accepted semantic build is 9bce4af5463a57683ed375a8b0fdcbcb83620bce923624d7075bd0c4f4ac847f.

- Bundle SHA256: `887975589fdb38a2fd626b5556551319436504e1cc8cebdc7e58a358929cb0e7`.
- ZIP size: 89,708,425 bytes; 258 payload files.
- Code root: `analysis_code/87dbb953cf22ecba8dab1004d4290760f65ab21c3d38e4033a9a6fcdd73d21ea`.
- Development manifest: `0febeb5b61465fd9e0311e2e7deca3f95bbdcb5da2dc485079270ccef4e2ed9d`.
- CPU verification status: verified. Batched maximum logit/gradient/weighted-loss
  errors are 2.623e-6 / 2.146e-6 / 9.537e-7. Checkpoint parameter error is zero.
- Work inventory is unchanged: 25,146 development logical items / 3,362 workers;
  prior workers now use the dedicated cpu-prior resource profile.

`scripts/update_analysis.sbatch` is a new, separate update-only helper, pinned
to the above bundle and installer/environment-helper hashes. It runs on med
with four CPUs, 12 GiB, 15 minutes; checks for other active/queued researcher jobs
in this workspace; confirms existing package/CUDA-wheel versions and cache paths;
backs up the installation receipt; invokes the checked installer and preflight.
It performs no pip installs, extraction/alignment, or scientific fits. Bash and
embedded Python 3.11 syntax were checked locally. The prior one-time recovery
helper remains historical and is not used for this update.

The researcher must upload the bundle, installer and update helper, submit the
helper in the persistent SSH session, and return its log/accounting outcome.
The researcher subsequently confirmed update job **1019947**, completed on
**csr126** in **10 seconds**, exit **0:0**. Its log ends with ANALYSIS SOURCE
UPDATE COMPLETE, environment reused, no scientific fits submitted, dated
2026-10-07T04:19:36Z (October 7 in America/New_York). Preflight issues are empty:
all 18 response files have the expected sizes, all five model alignments have
22 metadata-verified files each, and complete annotation coverage remains valid
with 200 parcels and the unchanged experiment-definition hash. These checks
do not rehash every full array or run fits. The supplied user queue was empty.

The researcher confirmed updated CUDA verification job **1019967**, completed
on **gr101** in **1:47**, exit **0:0**, batch MaxRSS **5,514,944 K** (about
5.26 GiB host RAM). Its log explicitly passes source-batched logits, weighted
losses, gradients and candidate order, as well as PCA/ridge, cache identity and
interrupted-epoch recovery, ending in COMPUTE NUMERICS VERIFIED. The pasted log
does not include numerical error maxima or the current GPU product name.
Its six-source, thirteen-query epochs are numerical checks, not full training
timings; they do not establish the production speedup.

Next proposed submissions refresh preparation receipts under the new manifest
for logical jobs 30 and 3894 (subject01 and Qwen27B layer64, fold0), while CPU
prior job 0 measures the revised implementation independently. Existing
compatible observation caches may be reused, but old-manifest execution receipts
cannot satisfy new-manifest dependencies. Follow with real linear/structured
decoder jobs 31/33 and 3895/3897 after preparation succeeds. Optimized full-fit
timing and the revised best backend for the prior remain unmeasured. These
retain production training definitions and checkpoint progress. No new
scientific jobs have been submitted by the agent.
No further full local preparation is needed for
the separate transfer helper or this post-packaging status-note update. The ZIP
retains its verified packaged documents; its code and scientific inputs match
the current implementation. Old source/checkpoints remain preserved under their
original identities; no unchecked checkpoint migration is authorized by a CPU
equivalence receipt alone.

## Latest evidence: transfer and first production workers

The researcher ran the complete local `prepare_analysis.cmd` successfully and
transferred the verified bundle. ZIP SHA256:
`d8d203a58b79af3c856a26e96247be20b39004af187eb4acba5fdb93ca0a9a58`.
Semantic build:
`9bce4af5463a57683ed375a8b0fdcbcb83620bce923624d7075bd0c4f4ac847f`.
Development execution manifest:
`cdc369d14df5a32eb730083323fa537227a0cd2f357f2b379a42be51b8721879`.

- Fresh inspection reported 8.9 TB available on the shared project filesystem,
  an empty user queue, and account `tshu2_scai01` with allowed/default QoS `jhu`.
  This is not a separate personal storage quota or proof of unlimited billing.
- Setup job **1018625** completed in **4:20**, exit `0:0`. The recovery script
  verified the uploaded bundle and existing alignment code, archived the known
  inactive Qwen lock, resumed alignment, and installed the analysis environment
  on an allocated CPU node. Its completion marker is dated
  `2026-10-07T01:19:07Z` (October 6 in the researcher's local timezone).
- Preflight reported no issues: all 18 response files were present at expected
  sizes; all five models had 22 source/aligned files each with verified metadata,
  dimensions, provenance and timing. This does not rehash every tensor payload.
- CUDA verification job **1018775** completed on `gh122` in **21 seconds**, exit
  `0:0`, and wrote `artifacts/compute-verification-cuda.json`. Ridge, PCA,
  decoder gradients, cache identity and checkpoint recovery passed. The GPU
  product name is not established by the pasted log alone.

| Job | Manifest worker | Work | Node | Elapsed | Batch MaxRSS | Outcome |
|---|---:|---|---|---|---:|---|
| 1018878 | 30 | Subject 01, fold 0 brain preparation | gh103 | 0:47 | 8,175,740 K (~7.8 GiB) | Complete, 0:0 |
| 1018879 | 982 | Qwen 27B layer 64, fold 0 preparation | gh110 | 0:23 | 2,647,916 K (~2.5 GiB) | Complete, 0:0 |
| 1018880 | 1093 | Nine-participant, fold 0 Qwen-plus-symbolic matched-binding encoding | gh110 | 5:16 | 42,079,132 K (~40.1 GiB) | All nine logical fits complete, 0:0 |
| 1018881 | 0 | Shared query-prior panel, fold 0, seed 11 | csr057 | 26:44 | 5,983,788 K (~5.7 GiB) | Cooperative yield, exit 75:0 |

All GPU requests were one GPU, 11 CPU cores, 64 GiB RAM, 30 minutes,
eligible for `a100,h100,h200`. The CPU prior requested 9 cores and 32 GiB for
30 minutes, respecting the inspected `med` limit of 4000 MiB per CPU.
Model preparation also performed an automatic numerical check for its assigned
GPU type, so its elapsed time includes that verification. These are production
artifacts, retained for the complete experiment rather than disposable timing data.

The prior completed both learning rates in inner fold 1 (13 and 12 epochs),
then reported seven completed epochs in inner fold 2 at the first learning rate.
Epochs took roughly 47-48 seconds. Its log ends with `ANALYSIS YIELDED:
Allocation boundary reached; resume saved work.` Slurm labels its nonzero
application yield code as FAILED, although this is the intended continuation
path, not an exception or out-of-memory report. The worker was manually submitted
and does not resume automatically. Subsequent accounting reported TotalCPU
`03:52:02`, UserCPU `03:51:24`, SystemCPU `00:37.152` for the 26:44 allocation:
an average of 8.68 busy cores, or 96.4% of nine allocated CPUs. This demonstrates
high CPU consumption, not efficient thread scaling or superiority to CUDA.
Real-job checkpoint reloading remains to be inspected; no continuation was
submitted by the agent. At the researcher's request, further decoder submissions are on hold
while this CPU job's runtime and resource allocation are assessed.
Full nested decoder selection is not yet timed. Do not infer
whole-program duration from the 21-second numerical check or one encoding panel.
Learned decoder timing, device-memory/cache measurements, multi-node lock
validation and measured production submission settings remain pending.

Commands for the cluster now run directly in the researcher's persistent SSH
shell; Windows CMD is used only for local commands/transfers. All project writes,
including environments, temporary files, logs and caches, stay under the project
workspace. No setup or scientific computation runs on a login node. Source changes
to this local status document do not modify the already-installed analysis code.

### Prior backend comparison submitted by the researcher

`scripts/benchmark_prior.py` and `scripts/benchmark_prior.sbatch` use the installed,
hash-checked analysis code and the complete actual first inner fold of development
fold 0, with the same query targets, seed, source updates and optimizer. Each case
runs a three-epoch timing prefix, including validation, through `train_decoder`.
The first epoch records cold timing; epochs 2/3 give warm timing. This is a
hardware benchmark, not a shortened final scientific fit; the configured 80-epoch
cap, patience and all production definitions remain unchanged.

Two 15-minute allocations are planned: CPU `med` with 9 cores/16 GiB compares
1/4/9 threads; one GPU with 3 CPU cores/16 GiB, eligible for all five allowed GPU
partitions, compares CPU/1-thread and CUDA/1-thread on that same node. This
separates thread scaling on the CPU partition from the same-host CPU/GPU comparison.
Different node CPU types are recorded explicitly. Cases start in fresh processes
and do not read or publish the production prior-fit cache or checkpoints. Outputs
are confined to `artifacts/prior-benchmarks/job-<job-id>/`; the same-workload hash,
initial parameter hash, timings, CPU usage, memory, loss histories and last-epoch
weights allow subsequent performance and numerical comparison. CUDA timings are
synchronized and TF32 stays disabled. No full bundle rebuild is needed.

The researcher supplied completion evidence for both allocations:

| Job/case | Node | Warm epoch seconds | Average busy CPU cores | Outcome |
|---|---|---:|---:|---|
| 1019122 / CPU 1 thread | csr054 | 80.35 | 0.99 | Complete |
| 1019122 / CPU 4 threads | csr054 | 54.42 | 3.91 | Complete |
| 1019122 / CPU 9 threads | csr054 | 49.43 | 8.54 | Complete |
| 1019123 / CPU 1 thread preceding CUDA | gr101 | Not measured; cold epoch 399.99 s | Not reported | Yield 75 before CUDA started |

CPU job 1019122 completed in 10:32, exit 0:0, MaxRSS 5,093,232 K. Its CPU was
Intel Xeon Platinum 8480+. All three cases shared workload and initial-weight
hashes. Maximum validation NLL differences versus one thread were 3.61e-8
(four threads) and 8.81e-9 (nine); all selected epoch 3 within this prefix.
Last-parameter relative L2 differences were 6.31e-6 and 6.47e-6, respectively,
with maximum coordinate differences about 0.0083/0.0086. These are close loss
trajectories, not bitwise equality or certification of full-fit equivalence.
Four threads take about 10% longer than nine, with approximately half the
consumed CPU-seconds per warm epoch. One thread uses still fewer CPU-seconds,
at greater elapsed time; core consumption alone does not include allocation
memory constraints or queueing.

GPU allocation 1019123 ended after 13:08, exit 75:0, MaxRSS 4,759,252 K. It had an
RTX PRO 6000 Blackwell Server Edition (97,887 MiB; driver 595.71.05), but the
original suite ran CPU first. Only that CPU case began, consumed the budget and
yielded; **there is no CUDA prior timing from this job**. The cause of the slower
CPU epoch on this host is not established by the pasted output.

The local benchmark helper now runs CUDA directly for the `gpu` suite, accepting
a completed CPU reference directory as its second argument. Comparison checks
the same workload/initial weights and labels different-host timing explicitly.
It prints CPU/affinity/GPU metadata before training, including when a later yield
prevents a complete result. The researcher transferred and ran this correction
as job **1019191**, with the outcome below.

CUDA job 1019191 on gr101 yielded after 13:12, exit 75:0, TotalCPU 12:56.845,
MaxRSS 2,909,736 K. CUDA actually ran on the RTX PRO 6000, with one intra-op CPU
thread; the reported host CPU was Intel Xeon 6767P and CPU affinity [0, 1, 3].
Input/model setup took 80.85 s; the first full epoch took 345.27 s. There is no
complete three-epoch result or warm CUDA median. Validation NLL after epoch 1
was 3.0156716816506868, within 2.5e-8 of the completed CPU/1-thread result.
Full-prefix numerical comparison and last-weight comparison were not reached.
The reported inter-op setting of 128 is a configured pool size, not evidence of
128 active threads; accounting averaged approximately one busy CPU core.

For like-for-like cold epochs, med CPU/4-thread took 58.39 s and CPU/9-thread
52.66 s. The preceding GPU-host CPU/1-thread case took 399.99 s in a separate
allocation. The host CPU and allocation differ from the med benchmark, so this
does not establish a pure CUDA-versus-CPU hardware speed ratio. It does establish
that the measured CUDA allocation has no practical speed advantage over med for
this implementation/workload. No further prior timing run is needed now.

Code inspection identifies per-query answer/loss calls, small descriptor tensor
creation, and one optimizer update per source. These are plausible sources of
GPU launch/host overhead; no profiler has attributed their individual costs.
The decision is **CPU med, four threads for the existing query-only prior**.
This recommendation does not determine the backend of observation-conditioned
decoders, nor rule out later vectorization. Any such optimization must retain
source update boundaries, query/story weighting, splits and model definitions.

The researcher resumed manifest worker 0 as **1019247** on csr056 with four CPUs,
12 GiB and a one-hour allocation (3600 s budget, 180 s yield reserve). At the
provided 5:32 snapshot it was RUNNING. The log reports inner 2/9, learning rate
0.0003, `DECODER RESUME completed_epochs=7`, then epochs 8 through 13. Complete
post-resume epochs 9-13 take 52.66-54.72 s. This confirms actual production
checkpoint reload and continuation across allocations; the panel is not yet
complete. Rising validation loss during these epochs does not indicate an
execution failure; the existing best-epoch selection and patience remain active.

Next proposed measurements run existing logical decoder jobs **31, 33, 3895,
3897** (not worker indices): subject01 linear/structured and Qwen27B layer64
linear/structured, all development fold0, seed11. Their preparation dependencies
are the completed logical jobs 30 and 3894 (manifest workers 30 and 982).
These are retained production fits with 30-minute GPU allocations and checkpoint
continuation, not shortened training definitions. No observation-conditioned
decoder runtime has yet been provided, and CUDA superiority for those fits is
not assumed from the encoding timings. Proposed requests use one GPU over all
five allowed partitions, six CPUs and 32 GiB per job, while prior 1019247 remains
independently active.

The researcher subsequently supplied these job IDs and a 9:55 snapshot:

| Job | Logical index | Status/node | Evidence |
|---|---:|---|---|
| 1019295 | 31 | RUNNING, gr101 / rtx6000 | First full linear brain epoch: 371.38 s, validation NLL 3.44778 |
| 1019296 | 33 | RUNNING, gr101 / rtx6000 | Device numerical checks passed; entered structured inner 1/9 at lr 0.0003; no full training epoch yet reported |
| 1019297 | 3895 | PENDING | MaxGRESPerAccount |
| 1019298 | 3897 | PENDING | MaxGRESPerAccount |

Prior 1019247 was still RUNNING at 19:03 in that same snapshot. The subsecond
epoch timings in the structured log belong to the automatic numerical checks,
not its full training fold. Absence of a first full epoch is not itself proof
of a hang or an attributable training duration. The linear cold-epoch timing is
a performance concern; warm timing, CPU/GPU utilization and structured timing
are still needed before broad decoder submission. No CPU/CUDA superiority for
the observation-conditioned decoders has been established.

Official Slurm reason-code documentation identifies MaxGRESPerAccount as a
per-account GRES limit on the applicable QoS. This is not a disk quota or a
request-walltime problem. Two personal GPU jobs running does not establish a
two-GPU account cap: other account usage and typed GPU/partition limits must be
inspected. Earlier estimates assuming four continuously available decoder GPUs
remain hypothetical.

The next researcher snapshot resolves the scheduling question: `jhu` has
MaxTRESPA=`gres/gpu=16` and MaxTRESPU=`gres/gpu=16`. The account queue shows 14
other running single-GPU jobs plus this project's two, exactly 16 GPUs in use.
The two pending jobs therefore await room under the shared account cap; there
is no evidence of a two-GPU personal cap. The current rtx6000 partition reports
QoS=N/A (no separate partition QoS), 20 physical GPUs across gr101-103 and
OverSubscribe=YES:4. That setting alone does not establish active CPU sharing
or explain the observed training performance. Other visible QoS names do not
establish authorization to use them; no QoS/account change is proposed.

Live sstat records for 1019295/1019296 report CPU time 12:16/13:05 and MaxRSS
3,556,648 K / 5,072,844 K respectively. TRESUsageInAve contains no gpuutil or
gpumem counters. No simultaneous elapsed duration was included in this latest
snapshot, so these CPU times cannot independently establish an average core
utilization ratio. Missing GPU counters are not zero GPU activity. The latest
decoder log still contains one full linear epoch and no full structured epoch.
Prior 1019247 progressed into inner fold 3/9, with warm epochs around 55.4 s.

The next useful read-only measurement is live nvidia-smi activity in an existing
researcher-owned running allocation, alongside final scheduler/log evidence
after the four submitted decoder windows finish or yield. No extra training job,
new allocation, walltime extension, or broad submission is proposed. The user
performs these cluster inspections; the agent has not accessed the cluster.

### GPU telemetry and next implementation blocker

The researcher ran five-second inspection steps in each existing allocation.
They completed as 1019295.0 / 1019296.0; these are telemetry steps, not completed
training fits. The visible device UUIDs differ, confirming the two jobs were
inspecting different assigned RTX PRO 6000 GPUs (each locally indexed zero).
Linear telemetry: SM activity 7-17%, memory activity 0% at reported resolution,
953 MB framebuffer usage, approximately 76 W. Structured: SM activity 8-14%,
memory activity 0%, 1063 MB framebuffer usage, approximately 73-74 W. These short
samples establish low sampled GPU activity and small memory usage, not a full
run utilization average or exact attribution of CPU/kernel/synchronization time.

At 18:02 elapsed, linear job 1019295 had logged epoch 2 at **353.77 s**, versus
371.38 s for epoch 1. The slow execution persists after the first epoch.
Structured 1019296 still had no full epoch reported. Model jobs 1019297/1019298
were pending; prior 1019247 remained running at 27:10. The current decoder path
has not demonstrated efficient GPU execution and should not be broadly submitted
unchanged. Increasing GPU size or allocation duration is not the demonstrated
remedy. Prior CPU/4-thread placement remains the measured choice for existing code.

Local code inspection confirms per-query answer/loss calls, descriptor tensor
creation, and per-candidate structured binding calls. Candidate optimization:
batch these within each existing source optimizer update, prepack immutable
public descriptors, and retain exact query/story weighting, repeats, model
architectures, splits and precision. Validate logits, losses, gradients, traces
and checkpoint behavior against the existing executor on accepted real inputs.
GPU utilization alone does not prove which individual operation dominates.

Before editing, the agent attempted to read the current build's complete.json:
`data/processed/semantics-reviewed/builds/9bce4af5463a57683ed375a8b0fdcbcb83620bce923624d7075bd0c4f4ac847f/complete.json`.
This raised PermissionError (Errno 13). The user's earlier permission repair
named a different build, b8e9ffca..., so it does not establish readability here.
Per the user's workflow, decoder implementation is stopped at this access blocker;
no synthetic inputs, substitute annotation build, or unverified decoder patch was
introduced. The user must restore access to the current accepted build before
real-input equivalence validation and the implementation pass can proceed.

Proposed user actions: cancel only still-pending GPU measurement jobs, request
cooperative USR1 checkpoint yields from any still-running decoder measurement
jobs, leave CPU prior 1019247 alone, and repair/inspect local build inheritance.
No remote cancellation/signal or local permission repair has been performed by
the agent. Existing code/manifest remain installed; future code changes will
require their own pinned bundle/manifest. Existing checkpoints must not have
their code identities edited to force reuse after a decoder implementation change.

### Access restored and local source-batching implementation

The researcher enabled inheritance on the active 9bce4af... build: all 170 files
processed successfully, with sandbox-user access visible in the resulting ACL.
The agent then read its 16,573-byte complete.json successfully. The access blocker
above is resolved; annotation bytes were not changed.

Shutdown evidence: brain jobs 1019295/1019296 exited at 23:00 with code 75 and
explicit ANALYSIS YIELDED messages. Linear epoch 3 took 335.06 s. The first full
structured epoch took **987.60 s (16:28)**, validation NLL 2.98806319. These are
checkpointed exits, not numerical exceptions. Job 1019298 was cancelled before
starting. Job 1019297 was observed RUNNING at 1:00, having entered model linear
inner fold 1; its cooperative USR1 exit was requested in instructions again, but
no confirming output has yet been supplied. The invalid-job message for 1019298
does not negate its confirmed cancelled state.

Local implementation now includes `neurosym/decoder_batch.py`, integrated into
training, untraced validation, packaging, code identities and compute verification.
See `compute.md` for the exact algebra and cache policy. Full reviewed query
programs execute in source batches; no optimizer steps, examples, query weights,
repeats, seeds, folds or model families are removed. Pointwise trace generation
and legacy structured syntax remain supported. Numerical equality is assessed
within FP32 tolerances, not asserted bitwise.

Focused CPU equivalence checks passed on 14 real queries across all four
families (maximum logit/gradient/loss differences 1.20e-6 / 4.18e-7 / 2.39e-7).
A subsequent full verifier passed its original decoder checks and the new source
checks including a complete source with a large candidate catalog, then reached
checkpoint/cache checks. The agent interrupted that long run under the user-run
workflow; no completed updated verification receipt or updated package is claimed.
The full-source case is now confined to the new source-batch comparison to avoid
repeating it throughout unrelated existing checks. Python 3.11 syntax checks pass.

Resource definitions now separate CPU prior (4 CPUs/12 GiB/1-hour windows) from
CPU observation decoders (9 CPUs/32 GiB). Unused CPU preparation/encoding fallbacks
were corrected to satisfy the previously inspected med memory-per-core limits.
GPU decoder defaults are 30-minute resumable windows with concurrency two;
measured optimized runtime remains pending and may justify later changes.

Next user action: `call scripts\prepare_analysis.cmd` locally for complete checks
and a new pinned bundle/manifest. There has been no cluster transfer or installation
of this pass. The currently installed source is unchanged. Old checkpoints remain
associated with that old source, and cannot simply be resumed under the revised
code hash. Request a cooperative exit from the old CPU prior as well before the
upgrade, if still running; do not delete its checkpoint. Frozen extraction and
annotation contents remain valid. After local preparation, transfer/install on a
compute node, then run actual CUDA equivalence and measure the revised decoder
path before full submission. No speedup factor is yet claimed.

Full production resource profiles still need the final measured settings recorded
before broad submission. Production source/configurations and existing checkpoints
have not been modified by the agent. No new remote job was submitted by the agent.

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
