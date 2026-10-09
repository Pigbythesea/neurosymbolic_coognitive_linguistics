# First production run: scientific feedback and execution priorities

**Date:** 2026-10-07

**Audience:** Researcher and cluster-handling / implementation thread.

**Recommendation:** Start primary production work now, order it to finish complete scientific comparisons early, and expose those results before the full manifest finishes. Reserve the next run window for justified revisions and remaining work. Preserve the full scientific scope.

This report records the scientific thread's inspection and launch advice. The researcher requested it for the cluster thread to read and use when adjusting the plan. It requests scheduling and interim-reporting changes; it does not authorize changes to scientific estimators, annotations, training settings, or direct cluster operation. Continue the established researcher-operated cluster workflow.

Read alongside [SCIENTIFIC_STATUS_HANDOFF.md](SCIENTIFIC_STATUS_HANDOFF.md) and [ENGINEERING_HANDOFF.md](ENGINEERING_HANDOFF.md). The historical rationale remains in [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md). This report concerns the first run's scientific usefulness and timing, rather than another optimization campaign.

## 1. What was inspected and what is established

The review examined the scientific, engineering and historical handoffs; current analysis, experiment and compute configurations; the actual development manifest; dispatch and reporting code; runtime evidence; and saved fitted decoder outputs.

The inspected development manifest was:

`01b086932fdf03228e899b2e5c131e7514b46ba76d67b080245548837a2df61e`

It contains **10,285 logical items in 4,033 workers**. Final evaluation is a separate release with 224 workers. Worker counts are not GPU-hours and cannot be scaled directly into elapsed time.

The engineering record reports successful numerical qualification, selected production fits, complete selected exports/geometry, and selected comparison workers. That supports beginning production without another general optimization round. It does not establish that the decoder uses the matching observations, that its maps have biological meaning, or that all planned fits will meet the estimated runtime.

No cluster connection, submission, cancellation, configuration change or numerical code change was made during this review. Remote status is based on researcher-supplied evidence and the repository records.

## 2. Existing scientific evidence deserves immediate attention

The locally available [fitted audit](../artifacts/trace-audit-1024054.zip) already contains full heldout prediction summaries for two structured context fits. These summaries cover the five heldout stories; they are not scores calculated from only the small trace samples used for numerical qualification.

Both fits use **seed 11**, train on **stories 06–10**, and test on **stories 01–05**. The model is repository ID `qwen38-27b`, layer 64; the brain fit is participant `subject01`.

| Observation | Matched accuracy | Mismatched-observation accuracy | Matched minus mismatched | Matched NLL | Mismatched NLL |
|---|---:|---:|---:|---:|---:|
| Qwen 27B, final layer | 32.88% | 33.00% | -0.12 percentage points | 3.0308 | 3.0243 |
| Brain, participant 01 | 27.15% | 26.89% | +0.26 percentage points | 5.4807 | 5.4650 |

These are the saved `story_macro` aggregates. Uniform-candidate chance is approximately **13.84%** for this aggregate; lower NLL is better.

For exact audit provenance, the summaries are in these ZIP members:

- Model: `fits/2334141570afef71cdd3b03c3b36033f322234e8258b6c4ab295984667550a51/complete.json`.
- Brain: `fits/b132d86dcec08aa750c2b935120f551192f6742e11bbb78d7b22f676b9ad87f6/complete.json`.

These are earlier fitted outputs used in the optimization audit and retained through qualified reuse. This review did not inspect all scientific scores from the latest six context outputs or comparison panels.

### Interpretation

**Above-chance performance in these two fits does not yet demonstrate convincing observation-specific recovery in aggregate.** Replacing the observation with a mismatched one produces almost the same accuracy and does not worsen aggregate NLL.

This is a warning about the current measurement's interpretation, not a rejection of the research idea, annotation set or every decoder task. Some individual families show differences. Also, a within-story mismatch can retain meaningful story context; frozen-model prefixes share earlier text. These results alone cannot distinguish query/candidate priors, persistent context, weak observation use, difficult targets or an inadequate readout.

The immediate comparison is therefore against the **query-only, linear, MLP and disrupted-training controls**, using matched evaluation support and per-task results. Anatomical maps and aggregate accuracy cannot substitute for that comparison.

The brain context fit selected **three refit epochs**, compared with nine for the model fit. Its story-01 NLL is approximately **13.75**, while the other four stories are approximately **3.02–4.08**. Inspect the existing selection history and story-level behavior. Three epochs is not intrinsically an error: it resulted from validation. Do not simply increase epochs or change preprocessing because a heldout score looks poor.

The ordinary development folds contain the comparator families needed for this examination. The context fits were primarily created to support geometry; their controls and training support must not be assumed identical to ordinary-fold baselines.

## 3. Deadline and runtime interpretation

The official COLING 2027 ARR submission deadline is **October 12, 2026, 11:59 p.m. AoE**, equivalent to **October 13, 2026, 7:59 a.m. New York time**. The December commitment date is a later stage for reviewed papers, not an extension for an initial submission. Sources: [official dates](https://2027.coling-iccl.org/) and [main-conference call](https://2027.coling-iccl.org/calls/main_conference_papers/).

The engineering projection is approximately **211 GPU-hours for development**, including the subsequently timed comparison classes.

| Continuously occupied GPUs | One development pass | Two development passes |
|---|---:|---:|
| 8 | 26.4 hours | 52.8 hours |
| 6 | 35.2 hours | 70.3 hours |

These calculations exclude queueing, launch overhead, interruptions and final evaluation. At an illustrative 70% average occupancy, two passes with an eight-GPU ceiling take roughly 75 hours. This is arithmetic for planning, not measured concurrent throughput or a completion promise.

The [stored extrapolation](../artifacts/runtime-planning-extrapolation.json), together with the later comparison estimate in the engineering handoff, assigns roughly:

| Work | Estimated GPU-hours |
|---|---:|
| Decoder selection and fitting/export | 156 |
| Encoding | 34 |
| Preparation, geometry and comparisons | 21 |

Decoder work dominates. Dropping a few small geometry comparisons will not save a full day. Calibration covers few conditions; wider model-augmented encoding is less thoroughly timed; completed/reused work is not subtracted. The final manifest has no independently established whole-run elapsed-time estimate in this review.

The goal is useful evidence during the first day and enough remaining time to respond to it. Two complete recomputations are a contingency, not a scientific requirement.

## 4. Why launch order and interim reporting need adjustment

The [dispatcher](../scripts/submit_analysis.py) scans eligible workers in manifest order and groups them by resource. It respects dependencies and global/per-resource caps, but has no explicit scientific-comparison priority. Several seeds are bundled in one primary worker.

The [study job builder](../neurosym/study_jobs.py) makes the final study report depend on all fitted parents, geometry panels, comparisons and coverage. The [report reader](../neurosym/study_reports.py) requires completed parent receipts. Individual `complete.json` records become useful earlier, but the existing final report is a completion barrier.

Therefore, a high completed-worker count after a day does not guarantee a balanced, interpretable set of human–model comparisons. Improve the order in which comparisons become complete and provide a separate interim summary. Preserve the existing final report's completeness requirements.

## 5. Requested production priorities

These are priorities within the full production study. Completed results remain part of the study; the remaining conditions continue afterward without a new implementation cycle.

### A. Collect the existing scientific evidence immediately

Read already completed ordinary-fold results and assemble the corresponding query-only, linear, MLP, structured and retrained-null scores. Include matched/mismatched observations, per-task accuracy and NLL, candidate chance, evaluation support and selection histories. Check the latest context outputs as well, rather than treating the two archived fits above as the latest complete evidence.

This is a read-only evidence review and can proceed while independent production work begins. It is not a request for another synthetic check, annotation review or fresh benchmark run.

### B. Finish complete primary comparisons early

Prioritize all nine participants and all five final-layer models, interleaved by heldout story. Complete the corresponding baselines and controls together so a result can answer a scientific question immediately.

One concrete ordering is to finish **ordinary folds 0 and 5** across that panel before completing every fold for the earliest participant. These fold numbers are a scheduling suggestion based on their positions, not their effects. Retain all ten development folds in the study. Include query priors, all primary readout families, matched/mismatched evaluation, structured training nulls and the existing faithfulness checks.

Prioritize the matched-content encoding comparisons and model-conditional encoding on corresponding participant/story support alongside decoding. Retain the other declared semantic comparisons; do not restrict which annotation families are scored.

### C. Schedule the other scientific branches alongside prediction

Bring forward the existing two multistory context partitions and their relevant native geometry, structured geometry parents and encoding-implied geometry. Complete their necessary coverage and comparison outputs together. Encoding and native geometry remain informative if the structured decoder needs revision; one branch does not require a positive result from another.

The engineer should resolve these priorities into dependency-complete worker selections under the existing shared six-to-eight-GPU operating limit. Use the qualified executor and existing selection mechanism where practical. Do not create competing controllers that bypass the project-wide cap. Preserve useful overlap among preparation, decoding, encoding and derived work.

Do not promise that simply selecting seed 11 runs one seed everywhere first: current primary worker groups contain three seeds. Prioritizing complete existing worker groups avoids an unnecessary executor change. Any later change to worker packing must preserve individual fit definitions and provenance.

### D. Expose interim scientific summaries

Provide a read-only summary of completed matched panels before the final study report is eligible. At minimum include:

| Output | Scientific purpose |
|---|---|
| Coverage by task, story and observation | Distinguish missing/unseen/uncertain support from a measured failure. |
| Decoder effects against priors, mismatch and training nulls | Determine whether matching observations contribute to semantic answers. |
| Linear/MLP/structured comparisons and selection histories | Locate a possible readout-specific problem or convergence issue. |
| Paired encoding effects on identical rows | Test prediction beyond matched content and beyond model representations. |
| Geometry item/context support and available reliability/control results | Determine which distances and maps support interpretation. |

State exactly which participants, stories, seeds and conditions have finished. Do not treat missing results as zeros or call a partial panel the final population analysis. Interim inspection should not silently redefine multiplicity families around the results that happen to finish first.

Continue the remaining production folds and descriptive conditions automatically where dependencies permit. The interim report supplies timely feedback; it does not impose a mandatory pause after each small batch.

## 6. What can follow later, and what should remain intact

| Preserve in the main scientific evidence | Reasonable to schedule later |
|---|---|
| Encoding, decoding/grounding and geometry | Intermediate-layer depth profiles |
| Participant/story diversity and all semantic task families | Additional descriptive cross-comparisons |
| Matched constituent controls and identical comparison support | Passage-latent geometry as a secondary interpretation |
| Query-only and observation/training controls | Additional seed results after initial inspection, while retaining them for stability claims |
| Training-only selection and reserved final story | New datasets, atlases or architecture searches |

Several efficiencies are already adopted: intermediate layers use linear probes rather than the complete nonlinear grid; tuning settings are shared across refit seeds; encoding uses one combined search; full traces are exported for the relevant structured context parents. Preserve those decisions. No further general optimization campaign is requested before launch.

The numerical-signature deduplication in grounding geometry remains a scientific interpretation issue recorded in the scientific handoff. Current qualification verifies faithful computation of that estimator; it does not establish that value-dependent weighting is the best measurement. Do not silently replace it in a scheduling change. A later justified aggregation revision could reuse fitted models and traces rather than require every model to be trained again.

## 7. How the first-day evidence should guide revision

| Pattern | Appropriate next scientific question |
|---|---|
| Simple observation readouts work; structured recovery is weak | Is the structured architecture, projection, objective or fitting procedure limiting access? |
| Encoding predicts responses; decoding is weak | Which information is predictively useful but inaccessible under these decoder tasks/readouts? |
| Matched and control performance remain similar across readouts | Are task priors, persistent context, observation construction or generalization support dominating the measurement? |
| Reproducible human–model differences emerge | What distinction and representation view support the dissociation, and does it survive the relevant controls? |
| Coverage or reliability is insufficient | Which claim remains unresolved rather than disproved? |

A weak aggregate score should not automatically trigger new labels, a smaller scientific scope, or arbitrary extra training. Revisions should address a diagnosed issue and be documented. Keep story 11 reserved until the development decisions are fixed.

Reuse valid, unaffected outputs. A decoder revision need not invalidate annotations, frozen-state extraction, independent encoding fits or native geometry. Conversely, changing a target, training protocol or fitted preprocessing can invalidate dependent outputs; the engineering thread should identify that dependency boundary rather than relabel old receipts.

## 8. Venue assessment and unresolved evidence

The project is appropriate for COLING's stated computational cognitive modeling, psycholinguistics, semantics and interpretability scope. The eventual paper needs informative findings and defensible comparisons; exhausting every layer/view combination is not itself a quality criterion. The official [ARR reviewer guidance](https://aclrollingreview.org/reviewerguidelines) emphasizes whether the experimental sample and evidence support the claims. There is no acceptance guarantee, and this review did not redo the project's literature/novelty assessment.

The following were not established by this inspection:

- Current scheduler availability and sustained six/eight-GPU throughput.
- Scientific scores for every latest completed baseline and context comparison.
- Exact first-day completion time under a revised priority order.
- Full final-evaluation runtime.
- Convincing observation-dependent structured recovery or reproducible biological grounding across the full panel.

The engineering thread should report which early panels the chosen worker selections will complete, which existing fits they reuse, and what remains uncertain about their timing. It should retain the full development inventory and separate final release, and provide researcher-run commands through the established workflow when the concrete launch plan is ready.

**Requested action:** Prioritize complete primary comparisons, collect existing scientific scores now, add interim reporting without weakening the final report, and proceed with production. Preserve the next run window for evidence-driven revisions and completion rather than automatic recomputation of everything.
