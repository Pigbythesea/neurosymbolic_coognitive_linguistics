# First production run plan

Prepared 2026-10-07 from FIRST_RUN_SCIENTIFIC_FEEDBACK.md and the researcher's
subsequent decision: **six concurrent GPUs maximum by default**. The full
development run has not been launched by the agent.

**Current status (2026-10-08): selected first-day release COMPLETE; scientific review
before expansion.** The final interim summary confirms 1,602/1,602 current execution
receipts, the complete 1,050-item early panel, and 28/28 primary decoder panels.
The controller stopped at the selected boundary and the queue is empty. The final
encoding task 1028558_0 completed in 00:03:58, exit 0:0, after release of its Slurm
environment-retrieval hold. Regenerate and download the evidence ZIP using CPU
`inspect` mode; no additional fits are authorized by this completion.

The researcher authorized this plan. The new packet
implements the exact selection and completion stop, superseding the earlier
automatic-continuation instruction. The researcher launched it as controller
**1027471**; initial logs confirm the selected 810 workers / 1,602 items, six-GPU
cap and 25 completed execution receipts. Six GPU tasks were observed running.
Do not restart production or expand the selection before scientific review.
Both inspect and run modes in the new packet use the same pinned selection.

## Preserved scientific installation

- Manifest: `01b086932fdf03228e899b2e5c131e7514b46ba76d67b080245548837a2df61e`.
- Installed code: `analysis_code/17181841f6fd955de63477a406143d2b31e739b4b7507d81846cc4860abe71ff`.
- All 4,033 workers / 10,285 logical items remain in the development inventory.
- No change to labels, estimators, settings, seeds, folds, precision, output
  identities, final-report completeness or the separately reserved final story.
- Existing qualified outputs and continuations remain reusable. Do not rebuild
  the analysis manifest or rerun `prepare_analysis.cmd` for this administrative packet.

## First-day choices and runtime

The roughly 211 GPU-hour estimate described the entire development inventory, not
the time needed to answer the first core questions. The earlier readiness report
did not separately cost that milestone. The following selections use the actual
manifest, retain complete existing worker groups and dependencies, and subtract
the qualified completed results in inspection 1027340.

| Cumulative evidence release | Workers / logical items | Remaining GPU-hours | Hours at 6 continuously occupied GPUs | Hours at 3 occupied GPUs on average |
|---|---:|---:|---:|---:|
| Primary prediction | 294 / 1,050 | 27–31 | 4.5–5.2 | 9.0–10.5 |
| Primary prediction + native/encoding-implied geometry | 645 / 1,417 | 29–33 | 4.8–5.5 | 9.5–11.0 |
| Recommended: above + selected learned-grounding contexts | 810 / 1,602 | 38–42 | 6.3–7.0 | 12.6–14.1 |
| Above with every primary system's learned-grounding contexts | 1,446 / 2,274 | 60–65 | 10.1–10.8 | 20.1–21.6 |

These are measured-class extrapolations, not measured concurrent panel runtimes.
Ranges are sensitivity scenarios assigning unmeasured model-augmented encoding
1x–5x the semantic-encoding calibration, not confidence intervals. Selection/refit
costs extrapolate from few systems; current context costs use a newly refitted
seed's cumulative whole-fit duration rather than a restored-weight export alone.
Queue gaps, critical paths, controller turnaround, reporting, GPU type and training
variation add uncertainty; division by the cap does not measure those costs.
CPU priors/reporting are additional and can overlap GPU work. The recommended
release targets useful first-day evidence; roughly 13–14 hours at half cap occupancy
leaves some headroom within a day, but does not guarantee completion in 24 hours.
The broadest context option leaves much less headroom. No hundred-day estimate
applies to these selections.

Reproducible planning: `artifacts/first-day-planning.py` and
`artifacts/first-day-runtime-plan.json`, including exact worker indices, timing
samples, completion subtraction and cost by workload. These are planning artifacts,
not a submit script or a newly installed experiment definition. The administrative
implementation verifies that its selected worker set matches this proposal exactly.

## What the recommended release answers

The priority prediction panel covers ordinary folds 0 and 5 (heldout stories 01
and 06), all nine participants and all five final-layer models. It contains:

| Work | Logical items |
|---|---:|
| Decoder preparation | 28 |
| Selection receipts | 86 |
| Decoder refits including shared priors | 342 |
| Encoding fits | 594 |
| Total | 1,050 in 294 existing workers |

All three refit seeds, primary readouts, structured retraining nulls, mismatch
evaluation and existing faithfulness checks remain. Encoding includes all nine
semantic contrasts and the three-condition final-layer model-conditional chain.
The selected worker groups are dependency complete and unchanged.

Add native geometry for all nine participants/five final-layer models, both
complementary five-story contexts, encoding-implied geometry, co-occurrence and
the corresponding comparisons/coverage. This adds only about 1.5 GPU-hours to
the primary prediction release and gives an independent organization analysis
if structured decoding needs revision.

For learned-grounding geometry, the recommended first-day release includes
participants **subject01/02/03**, **qwen38-27b** and **olmo3-7b-base**, both
complementary contexts and all three seeds. These fixed IDs use existing audit
work and include two model families; they were not ranked by effect size. The
30 context fits include six available fits and 24 new fits. Their parent selection,
geometry and supported comparisons are included. Other primary systems' grounding
contexts, the remaining ordinary folds and intermediate layers remain in the full
study for later continuation. This is a production evidence release, not a change
to the data, scientific estimators, annotation families or final study scope.

Complete ordinary controls answer whether the observation contributes and whether
the structured readout differs from simpler ones. Encoding tests graph information
beyond matched content and beyond model features. Native geometry examines
organization independently of a learned decoder. The selected context fits expose
grounding behavior and initial context/seed/participant stability; three participants
and two model families do not establish full-panel anatomical generalization.
Ordinary controls never substitute for context-fit controls: their partitions differ.

Two heldout development stories support interim story-specific findings, not the
complete ten-story conclusion. Keep every task family and report null/negative
results alongside favorable ones. Seeds are not participants; missing is not zero;
no partial-family BH or unplanned population claim. Development-guided method
changes must be recorded and invalidate their dependent outputs appropriately.
Story 11 stays reserved. A review decision concerns interpretation and diagnosed
method issues; it is not permission to suppress unfavorable completed results.

## Useful inspection before any new fit

Inspection 1027340 already provides 82 available logical results and two complete
ordinary decoder system/fold panels. `decoder.json` contains task/story support,
all readouts, priors, mismatch/training-null effects and selection histories;
`encoding.json` contains nine supported participant comparisons on story 01;
`geometry.json` exposes current support and existing comparison results. These can
be examined locally without GPU allocation or training.

On ordinary story 01, three-seed mean structured accuracy trails the query-only
prior by about 5.65 percentage points for subject01 and 4.00 for Qwen27B. Qwen's
linear/MLP matched-minus-mismatch effects are about +2.05/+2.19 points, whereas its
structured aggregate is about -0.09. This motivates a readout/task/control review;
it does not diagnose an architecture failure across the study. For example, Qwen's
structured composed-task accuracy exceeds the prior by about 4.89 points but the
retrained null by only 0.08: beating a prior alone is insufficient evidence of
observation-specific recovery. Inspect support before interpreting rare tasks.
The current mean matched-binding encoding delta-r is about +0.00013 across nine
participants on one story (six positive, three negative), also inconclusive.

Review all task-level contrasts, validation histories, story support and map/control
reliability together. Do not automatically increase epochs, regenerate annotations
or search for favorable tasks from these development scores. If simple readouts
work and structured ones do not, investigate the structured readout; if encoding
works but decoding does not, interpret their different access demands; if all
observation controls are similar, inspect observation construction and task priors.
Unaffected encoding/native results can be retained if a decoder revision is justified.

## Implemented administrative restriction

Dispatch is restricted to the chosen dependency-complete selection and stops when
it is complete. It cannot automatically enter the remaining full inventory.
Controller continuations pin the same packet, selection and global six-GPU cap. Prioritize
complete primary comparison panels and ready derived results within the selection;
independent encoding/native work can overlap context parents. Keep the existing
worker groups and safe-point resumes. No new numerical benchmark is needed.

Reports expose both milestone completeness and the unchanged full inventory.
If the researcher chooses a calendar budget, stop new dispatch at that boundary,
allow active workers to finish or save their existing safe-point continuation,
and report exact unfinished conditions. A clock boundary is not evidence that a
panel is complete. Any later expansion is a reviewed continuation of the retained
study, not an automatic second full recomputation. The selection/stop policy is
implemented locally. No calendar cutoff was requested or installed: first-day
completion remains a target, while the implemented stop is based on completion
of the selected work. Worker time boundaries preserve existing safe-point resumes.

## Existing scheduling and resumption (retain inside the selected release)

`submit_production.py` runs outside the immutable scientific bundle. It verifies
that bundle and uses its original array worker script. It retains the existing
manifest dispatch ledger and project-wide controller lock; do not run the old
full-manifest dispatcher alongside it.

At a six-GPU ceiling, soft shares are three prediction/preparation workers, one
context preparation/fitting worker, one encoding worker and one geometry worker.
These are preferences, not reservations. Ready branches borrow unused slots;
each original resource profile still limits concurrency. Both running and pending
GPU requests count toward six, including recorded tasks in other project manifests.
Untracked manual jobs remain outside this ledger; do not add manual study workers
while production is active.

The new packet orders early primary work first and excludes workers outside the
selected release. Ready refits/comparisons precede further setup. Seeds remain grouped as in
the qualified manifest. Worker allocations request 30 minutes and one GPU when
applicable, with existing safe-point exit-75 continuation. No GPU sharing between
independent worker processes is introduced.

The lightweight CPU controller uses two CPUs / 12 GiB / four hours on `med`.
It schedules a successor only at a clean allocation boundary, under an
`afterany` dependency on itself. Scientific workers continue during handoff.
Failures and uncertain submissions stop new dispatch and require inspection;
they do not cancel active workers or blindly retry failures. Controller queueing
can leave a gap after running workers finish. No uninterrupted occupancy is promised.

`artifacts/submissions/<manifest>/production-control.json` holds the current cap,
initially six. The controller reads changes each cycle and accepts 1-6. Lowering
the cap lets existing jobs finish and withholds new GPU requests until below it.
A fresh explicit controller invocation sets the requested cap again. Changes
above six require a later researcher decision and revised policy.

## Interim evidence

The initial report reads already completed scientific outputs while production
dispatch proceeds. A single background reporting process refreshes at most every
15 minutes. The reporter never trains a model, reads full trace tensors, performs
new permutation tests or evaluates story 11.

Location: `artifacts/production-reports/<manifest>/latest/`.

- `complete.json`: snapshot generation, expected/available counts, missing early
  jobs and completed observation/fold fit panels.
- `inventory.json`: every expected condition, with exact completion/provenance.
- `decoder.json`: aggregate and task/story scores, query/source/repeat counts,
  paired effects, mismatch/faithfulness, selection settings and saved histories.
- `encoding.json`: declared adjacent-condition comparisons on identical fitted
  support, using the installed support-checking routine. A missing middle model
  condition does not permit a baseline-to-full shortcut.
- `geometry.json`: semantic/item support, geometry availability and raw existing
  comparison results. Paths retain full original result details.
- `summary.md`: readable progress and matched/mismatch aggregate table.

Qualified archived fits are labelled separately from current execution receipts.
Missing/unavailable support is never zero. Context and ordinary results remain
separate. No partial-population confidence intervals, significance labels or BH
redefinition are introduced. A complete fit panel does not automatically mean
all paired comparisons have matching support; unavailable contrasts are explicit.
The original complete study report retains its declared inference and dependencies.

Latest files are replaced atomically per file, with a generation identifier and
`complete.json` published last; readers should match generations across files.
Small support/comparison caches and a progress history avoid repeated raw reads.
Full past snapshot copies are not accumulated. Scientific outputs are not deleted.

## Transfer and operation

`package_production.py` verifies scheduling against the real full dependency graph,
real archived summaries and yield records, then creates `artifacts/production-run.zip`,
`production-transfer.json`, `install_production_packet.py` and `start_production.sbatch`.
The archive expands only inside `artifacts/production/<packet-hash>` on allocated
compute. `start_production.sbatch` defaults to **inspect**: verify installation,
collect current evidence and create `artifacts/production-evidence-latest.zip`,
with no fit submissions. The new packet's explicit `run` mode starts only the
first-day selection and stops after its final report confirms all selected receipts.
It does not launch the rest of development or the reserved final story.

The researcher handles all uploads, inspection and submissions in the persistent
SSH session. Local commands use Windows CMD. Every file/cache/environment remains
inside the project workspace. Full launch readiness is reported before its command
is supplied, as requested.

The older, superseded packet was
`2e0ac8b23d2a005ce9b88598c05d8d19e61d38a12a36a968d31a22e751b6683c`.
Its 37,942-byte ZIP has SHA256
`9b82b6a65bc384d3cfc44e4bbf107545c470750bccf3aba74f00f0e211cb594a`.
Local verification passed the entire real dependency inventory at limits 1/4/6,
actual archived decoder summaries, actual yield/token handling, packet integrity,
dry-run and Python/Bash syntax. Researcher-run inspection `1027340` subsequently
completed in six seconds (exit 0), reading the current cluster scientific summaries
and computing nine support-checked paired encoding comparisons. The downloaded
evidence ZIP passed CRC and report-generation consistency checks. No full development
launch has been reported. See the engineering
handoff and `artifacts/production-inspection-1027340.json` for current evidence.

The new first-day packet is
`4c7c1d85c8d981b91713b1ddd08bc5b67fb4a995e1c50e0b0e2c850468d9e066`,
107,522 bytes, ZIP SHA256
`a3a1e62aba5aa724b6660ff7193875c253b08ff2de3945d8d575fd3b8086f971`.
Selection SHA256:
`5d674e4428baf6debefc0a1ee970129ff82b188465a23df7e83ddb0feb17e54e`.
The actual 810-worker dependency graph passed scheduling checks at caps 1/4/6;
out-of-selection dispatch, post-completion expansion and unwanted completion-boundary
continuation were rejected. Same-packet continuation, cap lowering, archive progress,
real-yield handling and task/story summaries passed; packet dry-run, CRC/hashes and
Bash syntax passed. Evidence: `artifacts/production-verification.json`.

Transfer only `production-run.zip`, `production-transfer.json`,
`install_production_packet.py` and `start_production.sbatch` from local `artifacts`
to the cluster workspace's `artifacts`. The researcher submits the start script
in `run 6` mode from their persistent SSH session. Installation, verification,
reporting and dispatch occur on allocated CPU compute; scientific workers request
their existing CPU/GPU profiles. No direct agent cluster access, analysis rebuild,
environment setup or repeated numerical benchmark is required.

## Interpretation and timing

The two locally inspected context fits have very small aggregate matched-minus-
mismatched effects. This is a reason to bring controls and task-level summaries
forward, not evidence of a particular failure mechanism or a reason to drop branches.
Three versus nine selected epochs are validation outcomes, not permission to add
epochs based on heldout scores. Anatomy requires observation dependence, specificity
and stability; native/encoding analyses retain independent scientific value.

The prior roughly 211 GPU-hour estimate is a conditional development baseline,
approximately 35 hours at six continuously occupied GPUs. Queueing, wider encoding
costs, convergence variation, launch overhead and final evaluation remain uncertain.
Reordering improves time to interpretable evidence; stopping at an explicit release
also limits the initially authorized work. Neither reduces the preserved full
study's total numerical work. Use the first-day table above for the next decision,
not the full-study duration. Do not promise a guaranteed first-day completion or
two automatic full recomputations.
