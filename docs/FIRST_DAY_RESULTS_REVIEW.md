# First-day scientific results review

**Date:** 2026-10-08. **Evidence snapshot:** 2026-10-08 18:56:20 UTC.

**Subsequent measurement review:** Read [FIRST_PRINCIPLES_MEASUREMENT_REVIEW.md](FIRST_PRINCIPLES_MEASUREMENT_REVIEW.md) before making the next experiment decision. It identifies endpoint-timing, categorical-support and identity-abstraction limitations behind these results. The numerical summaries below remain records of the completed measurement; the next step requires revisiting measurement definitions, beyond the optimizer-level diagnosis emphasized here.

**Assessment:** Continue the research. This release contains a promising human–model geometry result, small positive model-feature encoding increments, and informative weaknesses in the current structured readout. It does **not yet establish reliable observation-specific structured decoding from fMRI, a structural encoding advantage, or stable anatomical grounding**. Those distinctions matter for the next experiment and the paper's claims.

**Recommended next action:** Retain the completed-release review boundary. Use the saved fits and observations to diagnose decoder generalization and quantify the geometry/encoding findings; then approve a coherent continuation or method revision. An unchanged expansion of every structured fit is not currently the best use of the next compute window. The broad scientific scope remains intact.

This is a scientific review of actual returned evidence, not a new experimental authorization or an engineering status update. It supplements [SCIENTIFIC_STATUS_HANDOFF.md](SCIENTIFIC_STATUS_HANDOFF.md), [FIRST_RUN_SCIENTIFIC_FEEDBACK.md](FIRST_RUN_SCIENTIFIC_FEEDBACK.md), and [PRODUCTION_RUN_PLAN.md](PRODUCTION_RUN_PLAN.md). It supersedes the earlier 82-item snapshot as the empirical basis for the conclusions below.

**1. Evidence inspected and what is complete**

The local [production evidence archive](../artifacts/production-evidence-latest.zip) passes its ZIP CRC check. All five JSON records share the completion record's generation identity. The selected manifest contains **1,602/1,602 completed logical items**, including the entire 1,050-item early panel and all 28 primary decoder system/fold panels. The full development inventory contains 10,285 logical items; completion of the selected release is not completion of that inventory.

| Branch | Actual support in this release |
|---|---|
| Ordinary decoding and controls | Nine participants and five final-layer models; outer folds 0 and 5, holding out stories 01 and 06; seeds 11, 29, 47; query prior, linear, MLP, structured, and structured disrupted-training controls. |
| Ordinary encoding | Nine participants on those same two heldout stories; constituent-matched structural comparisons and all five model-conditional chains. There are 342 available paired contrasts. |
| Native geometry | All nine participants and five final-layer models; complementary heldout contexts 01–05 and 06–10; raw and presentation-residualized views, with pooled and scope-matched item policies. |
| Learned grounding/context geometry | Participants 01/02/03, OLMo3-7B-base and Qwen27B; both complementary contexts and all three seeds. This is a fixed initial subset, not a favorable-score selection. |
| Final evaluation | Story 11 remains reserved. The remaining ordinary folds, other learned-grounding conditions, and intermediate-layer work are outside this completed release. |

Archive SHA256: `50999ba43b9583b21e5c0e722bf0adf7d521626dda57ef61cfaf7f39e2df6019`.

Generation: `51bf1c3c753405a203ccbafeaf6b9e1fa15dbe42401ae49eeb694b24b24f79be`.

The [review script](../artifacts/review_first_day_evidence.py) and [derived numerical record](../artifacts/first-day-scientific-review-2026-10-08.json) make the aggregation reproducible. They read the archive, check its identities, aggregate existing scores, and compute one explicitly exploratory partial-correlation diagnostic. They do not fit models or calculate new p-values.

Decoder summaries first average seeds within participant/model × story × task cells. Human summaries then average the nine participants and two stories equally. The underlying exported metrics retain their declared source/query weighting. Model summaries give the two stories equal weight. Geometry means describe the stated system pairs; they are not population estimates treating every pair as independent. No new significance or multiplicity claim is made here.

**2. What the three branches mean**

Encoding asks whether a representation predicts heldout fMRI better than its comparison representation. Decoding asks whether an observation helps answer a semantic question. Geometry asks whether distinctions among semantic items have a similar relative organization in two representations.

These can differ. Human and model patterns can agree about which meaning-bearing passages are similar while a trained decoder struggles to identify an exact participant or resolve a scoped proposition in a new story. An item-conditioned average can reveal coarse organization even when individual observations are noisy. Conversely, a decoder can answer many questions using the public question and candidate descriptions without recovering the relevant information from fMRI.

The current findings therefore do not contradict each other. They identify which measurements are yielding evidence and which interpretations remain unsupported.

**3. Decoding: the question-only baseline is the crucial reference**

Across the two ordinary heldout stories, candidate-dependent uniform chance averages **14.08%**. The trained query-only prior reaches **40.72%** without a brain or model observation. This large difference shows why an above-chance decoder score alone is insufficient here.

| Human readout | Matched accuracy | Matched NLL | Accuracy minus query prior | Accuracy minus mismatched observation |
|---|---:|---:|---:|---:|
| Query-only prior | 40.72% | 2.818 | — | — |
| Linear | 32.09% | 3.952 | −8.624 pp | +0.352 pp |
| MLP | 38.69% | 2.908 | −2.031 pp | −0.090 pp |
| Structured | 36.16% | 3.101 | −4.554 pp | +0.017 pp |

Here **pp means percentage points**; lower negative log likelihood (NLL) means more probability assigned to acceptable answers. NLL assesses the whole predictive distribution, so it can detect changes that leave the top-ranked answer unchanged.

Replacing the structured decoder's observation with a mismatched observation from the same story changes aggregate accuracy by only 0.017 pp. Its matched-minus-mismatched NLL is **+0.00084**, also providing no aggregate advantage for the matching observation. The structured decoder trails the query prior in **all 18 participant/story cells**, after averaging seeds. Its mismatch effect is +0.047 pp on story 01 and −0.014 pp on story 06.

This supports a specific conclusion: **the current structured brain readout has not demonstrated convincing unit-specific semantic recovery in aggregate**. It does not establish that the brain lacks these distinctions, or that the graphs are wrong.

There are two qualifications that should stay visible:

- The structured brain decoder exceeds its separately retrained correspondence-disrupted control by **2.367 pp**, with NLL lower by 0.03495. That is evidence of sensitivity to the training correspondence under this fixed fitting procedure. It is not equivalent to sensitivity to the correct test observation. Story-level information, persistent context, different generalization behavior, and weak local observation use can produce different results for these two controls.
- A within-story mismatch preserves story identity and some ongoing semantic context. Near invariance to it is stronger evidence against precise local recovery than against every possible use of narrative information. The ordinary query-prior, mismatch, and training-null results need to be interpreted together. Their architectures and fitting procedures are not perfectly capacity-matched causal interventions.

The simpler model readouts do show more consistent observation dependence:

| Frozen model | Linear: matched minus mismatch | MLP: matched minus mismatch | Structured: matched minus mismatch |
|---|---:|---:|---:|
| OLMo3-7B-base | +1.009 pp | +1.373 pp | +0.114 pp |
| OLMo3-7B-instruct | +1.137 pp | +1.430 pp | +0.158 pp |
| Qwen3.5-9B-base | +0.893 pp | +0.829 pp | −0.079 pp |
| Qwen3.5-9B-post | +1.315 pp | +1.160 pp | +1.011 pp |
| Qwen27B (`qwen38-27b`) | +0.762 pp | +0.957 pp | +0.224 pp |

For every model, the linear and MLP matched NLL also improves over mismatch. These are useful positive controls: the pipeline can detect observation-dependent information in real frozen-model representations. Nevertheless, every model/readout's aggregate accuracy remains below the query-only prior. The closest, Qwen3.5-post MLP, achieves 40.54% and slightly improves NLL over the prior (2.813 versus 2.818). There is no general structured-architecture advantage in this release, and the larger model is not uniformly superior.

Task families should remain separate in the paper. For the human structured decoder, binding accuracy is 15.92 pp below the prior and changes by −0.034 pp under the mismatch comparison; reference accuracy is 6.84 pp below the prior with a +0.144 pp mismatch effect. The positive state-update difference against the prior, +3.78 pp, rests on only 11 and 12 distinct sources in the two stories. Identity-update support is only four and twelve sources. Thousands of total queries do not turn these rare families into well-replicated findings.

**4. Training histories identify a concrete generalization problem**

The structured brain readout selects only **1–3 refit epochs** in every ordinary participant/fold condition. The model structured readouts select 7–11. Early selection alone would not diagnose a problem, but the saved inner histories provide stronger evidence.

At the selected learning rate, across 54 brain structured inner trajectories (nine participants × two outer folds × three inner folds):

- Mean training NLL falls from **2.81 at the validation-selected epoch to 1.44 at the last attempted epoch**.
- Mean validation NLL rises from **3.12 at its selected minimum to 5.91 at the last attempted epoch**.
- Median validation NLL rises from approximately **3.02 to 5.04**; one final attempted validation value reaches 27.14.

The returned test scores use the selected fitting policy; the bad late checkpoints are diagnostic history, not the deployed result. All trajectories worsening after the selected minimum is partly a consequence of early-stopping selection. The magnitude and train/validation divergence are the important observations.

This is evidence of poor generalization or unstable extrapolation under the current representation/readout combination. It is not evidence that the jobs simply stopped too soon. Arbitrarily increasing epochs would go against the observed validation behavior.

The inspected observation code standardizes input coordinates using training statistics and then applies local PCA. It does not whiten the resulting component scores to unit variance. The structured readout applies learned local projections and nonlinear/pairwise composition. Consequently, site variance, component scale, domain shift between stories, and growing composed logits are concrete diagnostic targets. None has yet been identified as the cause. Whitening or normalizing everything by default would itself change the measurement and could discard useful amplitude information.

The absence of an explicit query-only bypass in the structured architecture does not guarantee observation-specific behavior. Learned descriptors and functions of broadly similar observation distributions can still support stable answer preferences. Actual matched-control effects remain the empirical test.

**5. Encoding: positive model increments, mixed or negative structural increments**

The exported encoding statistic is the participant-level **mean change in voxelwise heldout correlation**, averaged over nine participants below. It is not absolute prediction accuracy, variance explained, or a spatially localized effect.

| Paired representation comparison | Story 01 mean Δr | Story 06 mean Δr |
|---|---:|---:|
| Concepts beyond presentation | +0.000553 | −0.000208 |
| Matched predicate–role binding beyond unbound constituents | +0.000134 | −0.001056 |
| Matched general binding | +0.000520 | −0.000545 |
| Matched scoped binding | −0.000774 | −0.001028 |
| Reference package | +0.000453 | +0.000370 |
| Scope package | −0.000060 | +0.001156 |
| Discourse package | −0.000341 | +0.000116 |
| State package | −0.000875 | −0.002389 |
| Joint package | −0.000216 | −0.000333 |

The matched binding contrasts address added structure beyond constituent information. The other named packages follow their existing comparison definitions; they should not all be described as equally isolated tests of structure.

| Model added beyond presentation + C + BC | Story 01 mean Δr | Story 06 mean Δr |
|---|---:|---:|
| OLMo3-7B-base | +0.001425 | +0.003731 |
| OLMo3-7B-instruct | +0.001243 | +0.003569 |
| Qwen3.5-9B-base | +0.001456 | +0.005137 |
| Qwen3.5-9B-post | +0.001663 | +0.004824 |
| Qwen27B | +0.001235 | +0.004693 |

All five models have positive mean increments on both stories. Most participants have positive increments within each condition. Adding exact binding features **after** the model has already been included yields negative means for all five models on both stories: approximately −0.00030 to −0.00095 on story 01, and −0.00256 to −0.00367 on story 06.

The defensible reading is that these model representations add some heldout predictive information under the measured readout, whereas the current categorical binding representation does not show a consistent additional benefit. A finite, regularized estimator can generalize worse when given sparse or poorly supported additional features. This does not mean that binding is absent from the brain, nor does it prove the model completely represents it.

The mean over all cortical voxels can conceal localized gains and losses. Before judging practical effect size, obtain **absolute baseline/augmented correlations, paired voxelwise distributions, and anatomically defined summaries**, with reliability information where available. Regions must be predefined or selected inside training data; choosing favorable test voxels and reporting their mean would overstate the effect.

Exact categorical conjunctions also cannot estimate a distinct coefficient for a combination absent from training. Training/test support, feature dimensions, selected regularization and group weights should therefore accompany this result. The present negatives do not test every possible compositional representation of the accepted graph.

**6. Native human–model geometry is the strongest positive result**

Native geometry forms semantic-item response patterns from eligible observations and compares the distances between those patterns. It does not rely on the trained structured decoder's evidence maps. A positive human–model RDM correlation means that pairs of items that are relatively similar in human responses tend to be relatively similar in model responses.

The following are pooled-item means across nine participants × five models. Each context contains five heldout development stories. All 45 system pairs have positive native correlations for each of these kinds in each context, in both raw and presentation-residualized views.

| Semantic kind | Items, contexts A/B | Raw ρ, A/B | Presentation-residualized ρ, A/B | Residualized + exploratory co-occurrence partial ρ, A/B |
|---|---:|---:|---:|---:|
| Concept | 34 / 41 | 0.269 / 0.401 | 0.386 / 0.593 | 0.318 / 0.573 |
| Predicate | 100 / 131 | 0.331 / 0.383 | 0.413 / 0.516 | 0.374 / 0.495 |
| Role | 59 / 78 | 0.345 / 0.337 | 0.393 / 0.530 | 0.344 / 0.494 |
| Reference | 47 / 38 | 0.316 / 0.449 | 0.393 / 0.620 | 0.330 / 0.572 |
| Scope | 24 / 18 | 0.408 / 0.452 | 0.451 / 0.615 | 0.429 / 0.575 |

Context A is stories 01–05; B is stories 06–10. These coefficients describe association, not a percentage of shared neural computation. Averaging 45 coefficients does not create 45 independent human observations.

The last column is a **new exploratory calculation in this review**, not an already declared inferential result. For each available human/model pair, the script verifies that the human–model, human–co-occurrence, and model–co-occurrence coefficients use exactly the same semantic items and identical co-occurrence parent. It then calculates the correlation between rank residuals:

`partial = (r_hm - r_hc * r_mc) / sqrt((1 - r_hc^2) * (1 - r_mc^2))`.

There are 2,070 computable native comparison cells across the supported kinds/policies; 90 other available native cells lack the required identical available co-occurrence comparison. All 45 partial coefficients remain positive within every row/context represented in the table. No p-value is inferred from this formula, and no distance entries are treated as independent samples.

**The available co-occurrence measure does not explain away the native alignment.** That is a substantive positive finding. However, this one control is not a complete model of lexical similarity, story identity, time separation, overlapping hemodynamic responses, or narrative context. Presentation residualization also does not remove all of those factors.

The role labels illustrate an interpretation limit. An item can be a conjunction such as `remember / experiencer / person`, rather than an abstract role independent of its predicate and filler. A role-RDM association therefore need not isolate role binding beyond lexical/constituent content. That is exactly why the matched encoding branch and appropriately matched geometry comparisons remain useful.

**7. Context stability and anatomical maps constrain the stronger claims**

Native geometry is less stable when semantic identities are compared across disjoint subsets of stories. For presentation-residualized brain patterns, the mean fixed split-story RDM correlations are:

| Kind | Within context A: independent story subsets | Within context B: independent story subsets |
|---|---:|---:|
| Concept | 0.107 | 0.004 |
| Predicate | 0.044 | 0.074 |
| Role | −0.001 | 0.134 |
| Reference | 0.192 | 0.118 |
| Scope | 0.101 | 0.041 |

These use the common supported items in their respective splits, sometimes fewer than in the main alignment table. They are one fixed context split, **not a repeated-measure noise ceiling**. It would be invalid to divide human–model alignment by them as an automatic reliability correction.

The combination of strong same-context alignment and weaker across-context stability makes a context-sensitive interpretation plausible. It does not establish that alignment is spurious, nor does it establish a stable, context-independent semantic code. Distinguishing shared context-dependent semantic organization from generic narrative/temporal effects is a central next analysis.

Learned grounding requires a separate assessment. In the context fits themselves, matched-minus-mismatched brain accuracy averages **+0.091 pp in context A and +0.134 pp in B** across the three participants and seeds. Corresponding NLL differences are approximately +0.0010 and −0.0031. These are small positive accuracy effects, not literally zero observation dependence. The context fits do not have the same training partitions as the ordinary fits, so the ordinary query-prior and retrained-null scores cannot be substituted as their controls.

There are positive learned human–model RDM correlations, but a common answer vocabulary and trained architecture can induce shared organization. Positive learned RDMs do not repair weak observation dependence or establish reproducible anatomical localization.

For the three-participant grounding subset, mean anatomical map correlations are:

| Kind | Across seeds, contexts A/B | Across participants, A/B | Across contexts |
|---|---:|---:|---:|
| Concept | −0.049 / 0.091 | 0.148 / 0.188 | 0.083 |
| Predicate | 0.208 / 0.102 | 0.195 / 0.207 | 0.129 |
| Role | 0.001 / −0.003 | 0.007 / 0.022 | 0.002 |
| Reference | 0.005 / −0.008 | 0.006 / 0.020 | 0.002 |
| Scope | −0.024 / −0.051 | 0.146 / 0.170 | 0.031 |

Across-seed summaries use six comparisons per context; across-participant summaries use three participant pairs × three seeds. Across-context summaries use three participants × three seeds, with only 10/54/26/7/10 common items for concept/predicate/role/reference/scope respectively. Those comparisons are descriptive and dependent.

Predicate maps show some reproducibility, so it would be inaccurate to call every map random. But role and reference maps are nearly uncorrelated across these changes, and concept/scope maps have weak seed stability. The ordinary human structured top-site replacement control also shows essentially no accuracy distinction between replacing selected versus equal-count random sites: selected-minus-random is **−0.013 pp**, with NLL difference **−0.0053**. Negative accuracy would mean greater damage from selected replacement; negative NLL goes in the other direction. This is not a clear aggregate faithfulness advantage. It is a separate ordinary-fit result, not a direct intervention on every context map.

**The current evidence does not support a NEURONA-style headline that this readout has reliably localized specific structured concepts or role bindings in human anatomy.** No anatomical ground-truth labels are required in principle, but observation dependence, specificity and reproducibility must carry the evidential burden. Those conditions are currently too weak for that headline. They are not new restrictions on the research's scientific scope.

**8. Annotation richness and measurable repeated support are different**

The graphs may contain an excellent interpretation of a unique event without that exact configuration recurring often enough for cross-story geometry or a categorical heldout predictor.

In the native pooled contexts, the available signature counts for complete configurations are **0 and 2**; identity has **1 and 1**, qualification **1 and 0**, and state update **2 and 2**. Those kinds cannot supply an ordinary RDM association requiring at least three common nonzero items. Scope matching further reduces support; for example predicate signatures fall from 100/131 to 33/53.

This is a limitation of the measured repeated object, not a zero effect or evidence that those phenomena do not exist. Ordinary task families can remain evaluable even when cross-context item geometry is unavailable. Likewise, native reference geometry has 47/38 items while grounding reference geometry has only 12/11; comparing their headline coefficients as if they measured identical item sets would be misleading.

There is no basis in these results for reopening the entire annotation review. If sparse exact configurations prevent a desired generalization test, consider a linguistically justified factorized or constituent-matched measurement using the accepted graph. State precisely what that representation measures, preserve the old result, and select any fitting changes within development training/validation. Relabeling to obtain favorable neural effects would compromise the study.

**9. Publication assessment**

The research remains well matched to COLING's computational cognitive modeling, psycholinguistics, semantics, and interpretability topics. The official main-conference call lists those areas and the October 12, 2026 ARR deadline. [COLING 2027 call](https://2027.coling-iccl.org/calls/main_conference_papers/).

My present assessment is:

| Proposed paper claim | Status from this release |
|---|---|
| Graph-indexed native brain/model representations have related semantic organization in the same narratives | Promising descriptive evidence, including persistence after the available co-occurrence adjustment. |
| Frozen-model features add heldout fMRI prediction beyond the current presentation/content baseline | Positive mean increments on both tested stories; needs absolute/spatial effect characterization and broader story evidence. |
| The structured readout reliably recovers brain semantic answers beyond public-query information | Not established; aggregate comparisons currently argue against this claim for the implemented readout. |
| Exact symbolic binding gives additional fMRI prediction beyond constituent content or model features | Not established in the tested folds. |
| Structured grounding identifies stable concept/role brain maps | Not established, particularly for role/reference maps. |
| Humans and models have the same compositional mechanism, or systematic unseen-combination generalization | Not tested by the positive geometry coefficients and not supported by the available exact-configuration geometry. |

There is **credible top-venue potential**, but this archive is not yet a complete paper-level demonstration of the original grounding headline. The strongest emerging contribution may be an account of the relationship among shared semantic organization, context dependence, and access to structured meaning. That can preserve the project's ambition and all three branches. It is a candidate interpretation to test, not a conclusion to impose on the remaining experiments.

A negative decoding result can be scientifically valuable if measurement sensitivity and generalization have been established. A poorly generalizing readout alone cannot support a strong negative claim about neural representation. Likewise, human–model correlations alone require an explanation of the linguistic distinctions measured and an assessment of competing sources of agreement. The relevant publication standard is whether the evidence supports a substantive claim; completing an arbitrary number of model/layer cells is not sufficient. [ARR reviewer guidance](https://aclrollingreview.org/reviewerguidelines).

This review does not certify novelty against an exhaustive new literature search, assign an acceptance probability, or assume every research branch must have a positive effect. The comparison with prior work should concern actual questions, methods and conclusions, as agreed in the scientific handoff.

**10. Recommended continuation**

The next step is one integrated diagnostic and continuation decision using the real completed experiment. The following workstreams can proceed concurrently, mainly from saved artifacts.

| Workstream | Immediate work | Decision it resolves |
|---|---|---|
| Decoder generalization | Export train/inner-validation/test observation norm and variance summaries by site/story; retained ranks/coverage; per-task losses; candidate vocabulary/support; score and pair/composed-logit ranges. Trace the existing mismatch pair identities and confirm it changes the intended observation. Relate these to the already inspected learning curves. | Whether poor generalization arises from scale, support, fitting, an implementation defect, or the information available to this readout. |
| Encoding interpretation | Export absolute baseline/augmented voxelwise metrics, predefined anatomical summaries, feature train/test support and dimensions, selected group weights/regularization, and valid paired response support. Use existing predictions where possible. | Whether small cortical averages hide interpretable effects, and whether sparse conjunctions or estimator behavior explain the structural penalties. |
| Geometry interpretation | Export the supported item RDMs, source/story occurrence weights and per-story signatures. Verify the exploratory partial calculation directly; compare structure against matched constituent/lexical/context references on identical items; assess disjoint-context alignment and uncertainty with appropriate dependent units. | How much alignment concerns structured meaning, and whether it persists beyond shared occurrences and local narrative context. |

The last workstream is an analysis specification, not a demand to accumulate every conceivable control. Prioritize alternatives that could actually explain the observed alignment: constituent content, common source/story/time structure, and context stability. Existing representation arrays and graphs should supply much of this work without extracting all model states again.

For decoding, a useful methodological option is an explicitly regularized observation contribution on top of a trained public-query baseline, evaluated against the same baseline with identical public information. Linear/MLP already have prior branches, so adding a prior branch by itself is not a demonstrated solution. Selection would need to assess conditional observation contribution and generalization, not simply force more observation sensitivity. A factorized structural representation or normalized local readout is another option **only if the diagnostics justify it**. These are choices to examine, not conclusions that a particular architecture must be imposed.

If there is a concrete implementation defect, correct it and rerun the affected dependency chain with new provenance. If the implementation is faithful but the measurement generalizes poorly, document the scientific revision and use inner development selection. Preserve the current results as the original measurement's evidence. Any new observational control must preserve the distinction between narrative context and local source information, rather than treating an arbitrarily shifted signal as an unqualified null.

Then continue the integrated production study with the adopted definitions. Remaining ordinary stories improve generalization evidence; broader context/participant grounding is worthwhile when the readout offers interpretable observation dependence and stability. Unaffected annotation, model extraction, preprocessing and qualified fits should be reused according to their identities. A second run does not require recomputing every unchanged result.

Story 11 remains reserved until the development design is fixed. All ten development stories have already contributed to the context-geometry review, even though only two ordinary outer folds are complete. The other ordinary folds are useful additional development evaluations, **not wholly unseen confirmatory evidence** after this review. Revisions and exploratory analyses should be identified accordingly. A single final story will provide an independent narrative check, not an estimate of generalization to an unlimited population of narratives.

No new cluster launch is authorized by this review. The existing user-operated transfer/submission workflow and completed-release stop remain the operational boundary. The implementation/cluster thread should propose the specific saved-artifact export and affected computation from the findings above, with a revised runtime estimate based on actual dependencies.

**11. Limits of this examination**

The archive supports detailed score, support, training-history and geometry-summary review. It does not contain the full raw fMRI, every saved checkpoint/observation, the RDM arrays, or the paired voxelwise maps needed for the requested diagnostic extensions. No new result has been inferred from missing data. I have not established a fatal data/clock/annotation defect, and successful execution alone would not exclude one.

The analysis has not converted the many existing conditional permutation p-values into final population inference or corrected families. Seeds, queries, voxels, distance pairs, participant pairs and model pairs are dependent in different ways. The completed-release numbers are useful empirical evidence; the final paper still needs its declared contrasts, uncertainty and limitations at the appropriate participant/story/item levels.

The immediate scientific priority is to understand why stable same-narrative semantic alignment coexists with weak observation-specific structured decoding and weak anatomical map reproducibility, while quantifying the small encoding effects. This is a concrete research result to investigate, not a reason to discard the accepted graphs or reduce the project to software completion.
