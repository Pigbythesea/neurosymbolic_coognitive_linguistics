# Engineering handoff

Updated 2026-10-08. This is the engineering entry point for a colleague taking
over the current working tree. Scientific framing and protocol decisions belong
to the separate scientific thread; see [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).
Remote facts below come from researcher-supplied logs, not direct cluster access.

## Current position

**First-day milestone COMPLETE (2026-10-08), confirmed by researcher-supplied
accounting, controller log and final interim summary.** All 1,602/1,602 selected
logical items have current execution receipts; the 1,050-item early panel and
28/28 primary decoder system/fold panels are complete. The last task 1028558_0
completed on ga135 in 00:03:58 with exit 0:0 after its hold was released. The queue
is empty and the controller printed `FIRST-DAY MILESTONE COMPLETE; stopping for
scientific review.` The automatic stop worked. The remaining development inventory
and final story were not submitted. No selected-result gap remains in this report.

Next action is evidence export and scientific review, not another fit submission.
Run the existing start script in `inspect` mode on allocated CPU compute to refresh
`artifacts/production-evidence-latest.zip`, then download it. The locally downloaded
older archive contains only 82 items until replaced; do not interpret it as this
completed release. Review task-level controls/readouts, matched encoding and geometry
support/stability before any expansion. Runtime projections remain projections;
controller elapsed time includes the environment hold and is not GPU computation.
The held-job monitoring limitation below should be addressed before future runs.

### Pre-completion operational history

**Portal failure reconciliation (2026-10-08):** The researcher supplied explicit
October 7-through-now accounting with array expansion and duplicate records. It
lists 23 FAILED allocations with exit 75:0 and one historical TIMEOUT, 1022403
(`ns-p2-derived`, 02:00:28, reported exit 0:0). Seven exit-75 allocations are from
the current first-day run: 1027522_0, 1027768_0, 1028563_0, 1029353_0,
1029872_0, 1030142_0 and 1030317_0. Their 26–28-minute durations match the
30-minute allocation's safe-point yield window. The latest supplied inventory
already confirms all 372 selected decoder results, all 28 primary decoder panels,
and all selected geometry/comparison panels complete; no missing current decoder
result is indicated by these allocation labels. Individual checkpoint logs were
not newly downloaded in this audit. Earlier 1026953_0/1 are the documented trace
measurement yields followed by continuation 1027029. The 1022403 timeout predates
first-day controller 1027471; its detailed log is not present in the local audit,
so no additional cause is inferred beyond the reported time limit. No OOM,
node-failure or other non-75 FAILED entry appears in the supplied accounting.
At that audit, the nine-item encoding worker still awaited completion; it has since
completed as recorded above. Do not rerun the old tests.

**Latest recovery (2026-10-08):** The researcher released task 1028558_0 once with
`scontrol release`; the returned queue shows RUNNING on ga135 at 00:00:10.
That cleared the environment-retrieval hold. The same task subsequently completed
without duplicate submission or run restart, as recorded above.

**Status immediately before recovery: 1,593/1,602 selected items had execution
receipts; one nine-item encoding worker is blocked at Slurm startup.** Array task
1028558_0 is PENDING with reason `user env retrieval failed requeued held`.
Controller 1032744 remains RUNNING and repeatedly reports one outstanding encoding
worker; no GPU task is currently running in the supplied queue. All 28 ordinary
decoder system/fold panels are complete. Selected counts: 372 decoder, 603/612
encoding, 106 geometry panels, 377 comparison panels, 38 preparations, 96 selections
and one semantic-coverage item. The final study report is intentionally outside
this milestone. This is not completion of the whole development inventory.

Controllers 1027471, 1029464, 1030647 and 1031577 completed their approximately
four-hour windows with exit 0; those are normal controller handoffs. The latest
excerpt does not enumerate every worker's historical exit status. Confirmed blocker
is a scheduler environment-retrieval hold, not demonstrated scientific-fit failure.
The controller currently treats held tasks as occupied pending work, so it keeps
waiting and handing off rather than surfacing a blocking state. Record this for
the next administrative revision; do not modify the running immutable packet.
The researcher captured Slurm details/accounting, released the already requeued
task once, and checked its new state. No full run restart or result deletion was
needed. No remote action was performed by the agent.

**Latest researcher direction: implement the first-day evidence release; unrestricted
full launch remains on hold.** The new administrative packet now selects exactly
810 workers / 1,602 logical items and stops for review when they finish. The researcher
transferred and launched it as controller **1027471**, observed running at 00:00:42.
The older installed packet would continue through the full inventory and must not
be launched. Numerical qualification remains valid; no scientific source rebuild.

**The trace/storage optimization passed integrated local checks, was installed
by job 1026424, and passed real fitted-reference CUDA qualification in job 1026427.
Selected production fits completed; the full experiment has not been launched.
The selected complete exports and geometry panels finished in controller job
1026940 (43:34); accounting/storage and comparison job 1027113 have been inspected.** Scientific definitions, configurations,
folds, optimizer steps and seeds remain unchanged in this optimization pass.
Scientific decisions are recorded in [SCIENTIFIC_STATUS_HANDOFF.md](SCIENTIFIC_STATUS_HANDOFF.md).

### First-day evidence run (completed; launch record and fixed selection)

Researcher-supplied startup evidence for 1027471 confirms selection hash
`5d674e4428baf6debefc0a1ee970129ff82b188465a23df7e83ddb0feb17e54e`
and `gpu_ceiling=6`. The queue shows exactly six GPU tasks: encoding
1027477_0/1 (H100), geometry 1027478_0/1 (H200), and decoding 1027486_0/1
(RTX6000), plus CPU controller/prior/report work. The latest status has 25/1,602
selected execution receipts, up from 21 at startup. Earlier 82 available results
included qualified archives whose current-manifest adoption receipts are recorded
as their workers run; these counts are different, not evidence of lost results.
The 294-worker / 1,050-item PRODUCTION PLAN line describes the prediction subpanel;
the FIRST-DAY MILESTONE line defines the complete authorized selection. No error
appears in this startup excerpt. Sustained throughput/completion is not yet measured.

The earlier report conflated the roughly 211 GPU-hour full-development projection
with time to first useful evidence. The supervisor feedback requested first-day
results, but also automatic continuation. Its priority/reporting requirements were
implemented; an explicitly costed first-day milestone was missing. The researcher's
latest instruction now supersedes automatic continuation.

The authorized first evidence release is **810 existing workers / 1,602 logical
items**, including 82 available results. It retains ordinary folds 0/5 across all
nine participants and five final-layer models, all primary readouts/controls/seeds,
matched-content and model-conditional encoding, both contexts' primary native and
encoding-implied geometry, and learned-grounding comparisons for participants
01/02/03 plus Qwen27B and OLMo3-7b-base in both contexts/all three seeds. These
grounding systems are selected by fixed IDs, audit reuse and cross-model-family
coverage, not favorable effects. Full context coverage remains later work.

The measured-class projection is **38–42 remaining GPU-hours**, approximately
6–7 hours at six continuously occupied GPUs or 13–14 hours at three GPUs occupied
on average. The range varies unmeasured model-augmented encoding from 1x to 5x a
semantic-fit calibration; it is not a confidence interval. Queueing, dependency
gaps, device/convergence variation and overhead can exceed these values. Plan for
first-day evidence, not guaranteed first-day completion. A lighter release retaining
all primary prediction and native geometry needs about 29–33 GPU-hours. The full
primary context panel needs about 60–65. Exact dependency-complete selections and
assumptions: `artifacts/first-day-runtime-plan.json`, generated by
`artifacts/first-day-planning.py`; detailed choices: [PRODUCTION_RUN_PLAN.md](PRODUCTION_RUN_PLAN.md).

Implemented in the administrative controller: only the dependency-complete selection
can be dispatched; out-of-selection submissions are rejected; completing the
selection stops even at a simultaneous controller time boundary. Continuations pin
the same immutable packet and selection. Reports show milestone and full-inventory
completeness separately, distinguishing qualified reused evidence from executor
receipts. A completion record is published only after the final milestone report
confirms all selected receipts. Six-GPU cap, worker groups, resume/reuse, manifest
and estimators are unchanged. No arbitrary 24-hour kill is introduced: first-day
completion is a runtime target, not an enforced cancellation deadline.

New packet: `4c7c1d85c8d981b91713b1ddd08bc5b67fb4a995e1c50e0b0e2c850468d9e066`.
ZIP SHA256: `a3a1e62aba5aa724b6660ff7193875c253b08ff2de3945d8d575fd3b8086f971`
(107,522 bytes). Selection SHA256:
`5d674e4428baf6debefc0a1ee970129ff82b188465a23df7e83ddb0feb17e54e`.
Validation covered the complete selected dependency graph at caps 1/4/6, cap
lowering, other recorded project allocations, no expansion after completion,
boundary/continuation decisions, actual archived yields, downloaded evidence counts,
and archived task/story scores. Dry-run, packet hashes/CRC and Bash syntax passed.
Transfer and launch are now reported complete; monitor controller 1027471 and its
same-packet successors from the persistent SSH session. No analysis rebuild, environment
installation or another numerical benchmark is required. No remote action was
performed by the agent. Report readiness before the submission command.

### Older installed packet (historical; replace before running)

The researcher set the project GPU ceiling to **six**, superseding the earlier
eight-GPU operating target. See [PRODUCTION_RUN_PLAN.md](PRODUCTION_RUN_PLAN.md).
Separate administrative helpers preserve the qualified installation and manifest.
They prioritize the 294-worker / 1,050-item ordinary-fold 0/5 primary comparison
panel, with concurrent context fitting, encoding and geometry, then continue the
entire remaining development inventory automatically. The existing controller
lock/dispatch ledger and one-GPU-per-worker allocation remain; no competing
controller or numerical retraining protocol is introduced.

The production packet adds descriptive interim evidence reports, including
qualified archived fits, exact support-checked paired encoding effects and
task/story decoder comparisons. It keeps the final complete-report barrier and
story-11 release intact. Its inspect mode reads existing cluster evidence without
submitting fits. Local validation used the actual manifest and two archived fitted
prediction summaries; the first cluster report subsequently verified remote
ordinary-fit access and paired encoding comparisons, as recorded below.
No cluster operation was performed by the agent while preparing this packet.

Production packet `2e0ac8b23d2a005ce9b88598c05d8d19e61d38a12a36a968d31a22e751b6683c`
is prepared; ZIP SHA256
`9b82b6a65bc384d3cfc44e4bbf107545c470750bccf3aba74f00f0e211cb594a`
(37,942 bytes). Local checks completed the entire real dependency graph at GPU
ceilings 1/4/6, verified the two archived fitted task/story summaries and two real
yield records, rejected token mismatches, and passed packet integrity, dry-run,
Python 3.11 syntax and Bash syntax checks. Evidence:
`artifacts/production-verification.json`. The analysis ZIP and original
PROJECT_HANDOFF.md retain their previous SHA256 values. Transfer the production
ZIP, transfer receipt, installer and start script only; do not repackage analysis.

Researcher-run inspection `1027340` subsequently completed in six seconds, exit 0.
The downloaded `artifacts/production-evidence-latest.zip` passed CRC and snapshot
generation agreement across all report files. The current reporter read 82 available
logical results (21 current execution receipts plus qualified reuse), including
52/1,050 early-panel items and two complete ordinary decoder system/fold fit panels.
Nine matched-binding encoding contrasts across all participants on story 01 passed
the installed identical-support comparison. All six context summaries and five
comparison panels were readable; remaining unavailable geometry support was explicit.
This closes the remote reporter/access check; **no full launch has been reported**.
The earlier full-launch recommendation is withdrawn under the latest first-day
instruction above. The new small packet implements the stop/selection change;
no analysis rebuild or numerical benchmark is required. Scientific warning:
on the available ordinary story, structured accuracy trails the shared query prior
for subject01 and Qwen27B. Prioritize the planned control-complete panels and review
interim evidence; do not infer biological grounding from map stability alone.
Inspection provenance: `artifacts/production-inspection-1027340.json`.

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
  Detailed researcher-provided accounting now confirms 5,937 GPU-seconds
  (1.649 GPU-hours), including continuations, across this selection. Six traces
  occupy 2.989 GiB total: brain traces about 833 MiB each and model traces about
  187 MiB each. Exports take 462-523 seconds per fit; complete geometry panels
  take 274-295 seconds of process time. Against corresponding original whole
  files, trace storage shrank 6.87x (brain) and 26.80x (model). Evidence is saved
  in `artifacts/runtime-measurement-1026940.json`. The 43:34 includes reused
  work, scheduling and overlapping stages; it is not a cold full-study estimate.

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

The selected complete export/geometry path finished successfully on the cluster.
Extrapolating measured conditions by workload class gives about 206 GPU-hours
for development preparation, selection, refitting/export, encoding and geometry;
see `artifacts/runtime-planning-extrapolation.json`. This is a conditional subtotal:
comparison panels/reports, queueing and final evaluation are excluded; encoding
calibration covers semantic groups rather than all model-augmented widths, and
completed/reused work is not subtracted. It is not a complete forecast or a
guaranteed lower bound. Eight continuously occupied GPUs would process that
subtotal in about 26 hours. Several-day completion remains plausible but unverified.
Production comparison workers `2548,2549,2550,2581,2582` subsequently completed
under controller `1027113` (two minutes, exit 0), arrays `1027114` and `1027121`.
Three human-model derived-RDM panels took 12.14, 13.02 and 13.62 seconds; two
brain seed-stability map panels took 25.09 and 22.95 seconds (A100 process time).
Together with earlier native/implied comparison timings, a conditional projection
adds roughly 4.5 GPU-hours for development comparisons, giving about 211 GPU-hours
before scheduling/launch overhead and the other extrapolation uncertainties above.
This does not establish every comparison category or wider encoding cost.
Engineering evidence supports the qualified executor without another optimization
pass. It does not authorize unrestricted full launch: the latest first-day plan
above takes precedence. The researcher has set the ceiling to six GPUs; report
the concrete milestone and readiness before providing launch commands.
No full-development launch or further numerical code change has been performed.

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
- The current production ceiling is **six concurrent GPUs total across stages**,
  with lower limits supported by the production controller. Earlier eight-GPU
  statements describe the historical target/immutable manifest resource maximum.
  Prefer organized arrays and measured, resumable allocations.
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
directories. The latest post-1026940 report shows decoder outputs 17 GiB,
geometry 2.0 GiB, encoding objects 631 MiB and analysis cache 3.5 GiB, plus
smaller metadata/report directories. These include retained reference outputs.
Dataset, model cache and frozen-feature directories were not remeasured.
The shared lab filesystem has 8.8 TiB free; that is not a personal quota or
reserved capacity. Applying the measured trace sizes to 54 brain and 30 model
development parents gives about 49.4 GiB of full traces, conditional on comparable
source/query support. This replaces old multi-hundred-GiB trace extrapolations.

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
   This selected run completed under controller `1026940` in 43:34; cumulative
   decoder runtimes, accounting, geometry sizes and cache occupancy were then
   inspected. The five ready derived-RDM/map comparison panels subsequently
   completed under controller `1027113`, as recorded above. No new annotation
   or frozen-state extraction is required.
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
