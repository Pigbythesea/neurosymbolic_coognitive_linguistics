# Scientific scope and status handoff

**Updated:** 2026-10-07

**Role:** Current scientific entry point for a new colleague or thread.

**Decision:** Proceed with the full integrated research program. Annotation review is closed; the main measurement definitions have been implemented. Empirical conclusions must come from the experiments.

This document consolidates the researcher's decisions and the current scientific definitions. It supersedes outdated decisions and pending items in [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md), which remains the historical rationale. It is neither a new literature review nor a formal preregistration. It reports no established brain–model finding and makes no publication guarantee.

Engineering readiness, transfer, jobs, runtime evidence and resource choices belong in [ENGINEERING_HANDOFF.md](ENGINEERING_HANDOFF.md), [CLUSTER_STATUS.md](CLUSTER_STATUS.md) and [compute.md](compute.md). Their changing status is deliberately not copied here.

## 1. The project and its intended contribution

The project asks **how structured linguistic meaning is available and organized in human brain recordings and modern frozen language-model representations**. Its common analytical object is an evidence-linked semantic graph of the actual text participants read. Concepts, event predicates, ordered participant roles, reference, discourse, scope and changes of state all remain part of the scientific scope.

Three complementary branches answer different questions:

| Branch | Question | Evidence it can provide |
|---|---|---|
| Encoding | Which annotated distinctions predict heldout fMRI responses, including beyond constituent content or a frozen model representation? | Predictive information associated with a semantic representation under a specified readout and comparison. |
| Decoding and grounding | Which semantic answers can be recovered from brain or model observations, and where does a structured decoder obtain supporting evidence? | Semantic accessibility and reproducible, task-conditioned grounding without concept-to-region training labels. |
| Representational geometry | How do semantic items and configurations relate within each representation, across contexts, and between humans and models? | Agreements and dissociations in representational organization, with the meaning of each geometry stated explicitly. |

These are peer scientific components. A positive encoding effect is not a prerequisite for studying decoding or geometry. Nor is the project an alignment leaderboard: recoverability, conditional prediction, context stability, anatomical organization and systematic differences can each be informative.

The graph is an explicit analytical language and source of supervision. The study does not assume that the brain stores literal graph nodes or executes the decoder's symbolic program. A concept type, a particular story entity, a predicate and an ordered role assignment are different objects. Story-specific IDs identify occurrences; they are not automatically shared semantic types. Taxonomic hierarchy, compositional depth and temporal/discourse progression must also remain distinct when an analysis invokes them.

The prospective contribution is an empirical account of **structured meaning**, supported by these connected measurements. Existing encoding, probing, graph and grounding methods are methodological precedents. Novelty should be assessed against actual prior questions, data, supervision, methods and conclusions, rather than inferred or dismissed from overlapping terminology. The eventual paper must make that comparison explicitly; this handoff does not certify novelty from a new search.

## 2. Research commitments and freedom to revise methods

The researcher explicitly rejected deadline-driven narrowing and repeated cycles of miniature demonstrations as the project's governing strategy. The agreed approach is a comprehensive, integrated implementation using real data. COLING is a possible venue; scientific readiness determines the venue and submission, rather than a particular deadline determining the research questions.

Meaningful correctness checks, numerical verification and informed optimization remain necessary. They do not replace the intended experiment set. Synthetic observations, placeholder annotations and successful software checks cannot supply evidence for the scientific claims.

The original [first-principles reference](Structured_Meaning_Brain_LLM_Human_Reference.md) remains useful for motivation. The [technical proposal](Structured_Meaning_Brain_LLM_Research_Proposal.md) supplies candidate methods and a literature registry, but its detailed prescriptions are not binding. Later researcher decisions govern. The [original shared conversation](https://chatgpt.com/share/6ac18a1b-4f5c-83e9-b542-b85d56aa0512) was identified by the researcher as central; it was not newly retrieved for this handoff. This record relies on the available discussion and repository evidence, not an asserted independent reading of inaccessible history.

Current defaults make measurements concrete and reviewable. They do not forbid scientifically justified alternatives. A change to semantic meaning, information availability, a population, a comparison or a generalization claim should be explained and versioned. Routine implementation choices need not become new permission checkpoints. Code establishes what was computed; it does not silently redefine the scientific objective when it disagrees with the decision record.

## 3. Dataset, labels and what is actually supervised

The core is the **Deniz reading dataset: nine participants, eleven stories, 23,655 source words and 1,217 annotation units**. The inspected release supports nine participants, superseding the six-participant subset described in the original proposal. Stories 01–10 are development stories; story 11 is the final heldout story, including its repeated recordings. Repeats improve measurement/reliability assessment but are not new participants or independent narrative contexts.

This corpus was chosen because it joins extended reading, recurring semantic structure, recorded responses and recoverable stimulus timing. Complementary datasets remain possible when they answer a named scientific question. The old list of candidate corpora is not a requirement to collect or analyze all of them.

Annotations are made from the recorded **stimulus text**, not from fMRI or model performance. The stimulus, token coordinates and timing information can be available locally while the large response arrays reside elsewhere. The same accepted annotations and deterministic compiled objects are joined to observations through story, unit, token and timing identities. Physical data location does not create a separate annotation problem or require annotating brain arrays.

The phrase “without ground truth” needs a precise object:

| Object | Available supervision |
|---|---|
| Encoding target | Measured fMRI response values. |
| Decoder answer | Annotation-supported semantic answers derived from the text. |
| Concept-to-brain location | No direct labels saying which parcel contains a concept or role binding. This association is learned indirectly through answer supervision and evaluated through heldout evidence and controls. |

The last row is the NEURONA-inspired opportunity that motivated the researcher's question. It does not mean that the entire study is unsupervised or that nothing is trained. Base language models remain frozen; encoders/readouts, descriptor embeddings and training-fold preprocessing are fitted. Anatomical ground-truth labels are unnecessary for testing reproducible semantic associations, but the absence of such labels makes accuracy, observation dependence, specificity and stability central to interpreting a learned map.

## 4. Annotation decision: accepted agent-authored graphs

Production moved from the earlier CLI/rule-constrained attempt to direct, progressive agent authoring over the complete corpus (`deniz-direct-v2`). The small earlier comparison and the legacy Luna/CLI pipeline are development history, not production labels. The decision concerned the quality of semantic interpretation and the annotation workflow, not a general theorem that conversation interfaces outperform a CLI.

A separate agent review covered all eleven stories and 1,217 units, using source-first interpretations before graph comparison. The adopted review made **56 targeted operations across 38 units**; 1,179 unit graphs retained their baseline values. The researcher subsequently inspected the review and closed annotation review. Another complete annotation or adjudication round is not a standing prerequisite for downstream work.

The scientific input is the immutable reviewed export pinned by [configs/semantics.json](../configs/semantics.json):

- Export: `data/annotations/deniz-independent-review-v1/exports/5cd83e0779bb5440da47`.
- Full build hash: `5cd83e0779bb5440da4761cbaf26dd9791966212bbc2bc787c1efd8669524751`.
- Graph uncertainties and **29 retained review questions** remain part of the accepted input, linked to the affected records.

Acceptance means that this is the agreed scientific substrate. It is not a measured semantic-accuracy score, human interannotator agreement, or proof that every reading is uniquely correct. Review coverage and small correction counts must not be presented as accuracy estimates. Independent agent contexts are not independent human raters.

The researcher reported choosing GPT 6.1 Sol with max reasoning for production authoring. The saved runtime provenance exposes a GPT-6-based Codex family, but not the exact revision or reasoning setting; those fields remain explicitly unavailable. Distinguish the researcher's report from machine-captured provenance. Saved requests, drafts, decisions and review records support auditability; they do not prove absence of every possible out-of-protocol access or prior familiarity in pretraining.

See [direct_annotation.md](direct_annotation.md), [independent_review_handoff.md](independent_review_handoff.md) and [reviewed_downstream.md](reviewed_downstream.md) for provenance. Older statements about pending human adjudication or selecting a mutable latest pointer describe earlier stages; the accepted pinned export above governs analysis. Future concrete annotation errors should receive source-supported, versioned corrections, without silently rewriting the accepted archive or selecting labels to improve neural effects.

## 5. What the deterministic compiler does

The agent makes semantic judgments; the compiler converts those accepted judgments into analysis objects. It makes **no LLM calls and learns no parameters**. It produces normalized graph histories, features, public queries/candidates, answer targets, timing support and semantic-occurrence links.

Normalization uses explicit accepted aliases, predicate-conditioned role maps and reviewed scope cases. Raw records, ordered and repeated arguments, evidence spans, qualifiers and uncertainty remain recoverable. Label similarity alone does not authorize merging meanings. In particular:

- An introduction records discourse content, not actual-world existence. A missing edge, null argument or unmentioned assertion is not a negative fact.
- Identity is scoped; there is no unrestricted identity union across reports, beliefs, conditionals or other contexts.
- Local polarity and the full operator structure are separate. Negation, modality, attribution and qualification are not flattened into a single Boolean truth label.
- A later clarification can change what is supported from that point onward. It does not retrospectively supply information to an earlier prefix.
- Uncertainty withholds the answers or feature support that depend on it. It need not invalidate unrelated questions in the same unit.

This separation keeps flexible semantic authoring compatible with reproducible analysis. A deterministic compiler can still implement the wrong measurement, so source-to-feature and source-to-query checks matter. Structural validation alone cannot certify semantic truth.

## 6. Timing: three different clocks

Participants viewed a timed stimulus; the analysis uses that presentation record rather than guessing where they were reading from the fMRI signal. The scientific alignment distinguishes:

| Clock | Current definition |
|---|---|
| Text exposure | Original source-word spans and their released/aligned presentation times. Missing times remain unresolved. |
| Interpretation availability | The endpoint of the **whole annotation unit** the annotator could see. Earlier evidence words do not backdate an interpretation that also used later qualifiers or context in that unit. |
| Neural measurement | Delayed fMRI response samples. Current FIR delays are 1–4 TRs, with TR = 2.0045 seconds, constructed on each story's original timeline before trimming or masking. |

Four unresolved annotation endpoints remain unavailable for timed fitting. The frozen-model decoder observation is extracted from the text prefix at a unit endpoint, before the decoder question is supplied and without future stimulus text. Encoding also has word-aligned model features; their temporal construction must be stated rather than confused with the unit-end decoder observation.

fMRI measures the slower blood-oxygenation response and mixes activity over time. Its delayed windows can include effects of later nearby text. A causal text prefix and a delayed brain window therefore do not have identical temporal integration. The present design supports **offline, temporally indexed semantic association**, not exact word-isolated neural events or demonstrated online concept localization. Endpoint conservatism and explicit delays address information availability; they do not prove a perfect model of each participant's hemodynamics. [extraction.md](extraction.md) records the onset, sampling and trim conventions and their evidential limits.

## 7. Encoding: predict measured responses and define the contrast

Encoding uses group/banded ridge regression to predict the released native cortical voxel responses. Semantic vocabularies, scaling and regularization selection are fitted within training partitions. Inner story folds select the shared penalty and group mixture using normalized prediction error across variable voxels. Heldout correlation and error describe prediction; measured repeat reliability is reported where repeated data exist, without automatically treating it as a universal noise ceiling.

The important measurement correction is **matching constituent content before attributing an improvement to its arrangement**:

| Group | Meaning |
|---|---|
| `C` | Newly introduced entity/literal content and current event predicates. It does not automatically repeat all previously introduced referents. |
| `BC` | Independently summed predicate, role and filler marginals from the same argument incidences as PB/PBR, including reference-dependent fillers, multiplicity and uncertainty eligibility. It removes the conjunctions, not their constituent content. |
| `PB/PBR` | Exact predicate–role–filler conjunctions, with local/reference channels. |
| `GB/GBR` | General role–filler conjunctions. |
| `B/BR` | Binding features that additionally carry scope information. |
| `R`, `S`, `D`, `U` | Reference, scope, discourse and state-update feature packages. |
| `L`; presentation | Annotated surface information; separately, released presentation controls such as word/letter counts, length variation and pauses. |

The primary binding comparison is **presentation + C + BC** versus **presentation + C + BC + PB + PBR**. Otherwise, retrieving an old referent could introduce both content and binding while the analysis called the gain “structure.” BC addresses that concrete confound; it does not control every lexical or contextual difference in natural narrative.

The general-binding comparison adds GB/GBR. The scoped-binding comparison adds B/BR and measures a combined binding-and-scope contribution. Reference, scope, discourse, state and joint comparisons remain meaningful feature-package questions; their names do not automatically establish isolated causal mechanisms.

For frozen-model conditioning, the declared sequence is presentation+C+BC, then +model, then +model+PB+PBR. The result is predictive contribution conditional on that model, layer, readout and sample support. Correlated feature-group contributions are not unique biological causes.

**All competing conditions use identical observation rows within a comparison**, including training and validation. The default support is the union of groups and model states required by that comparison. Uncertainty in an unrelated feature group does not remove rows. The older universal all-group intersection remains an explicit robustness option. Within a required group, unresolved contributions make its affected complete-vector rows unavailable; unknown values are not observed zeros. Effects from different masks cannot simply be subtracted as paired effects.

PB/PBR and related encoding columns are exact categorical conjunctions. Seeing separate constituents does not create a coefficient for an unseen conjunction. The historical implementation label `factorized` is not evidence of learned compositional generalization. That claim requires the separately specified tasks and holdouts below. Details are in [experiment_definitions.md](experiment_definitions.md) and [encoding_support.md](encoding_support.md).

## 8. Decoder architecture and anatomical association

The decoder receives an observation plus a public symbolic question and candidate descriptions. It does not receive the source passage, private evidence, answer graph or correct answer IDs as input. Targets come from the accepted graph. Candidates use available prefix records or declared task syntax; alternative role assignments are alternatives in the task, not newly asserted false events.

Tasks include content identification, ordered role fillers, reference resolution, full scope, separate local polarity, role-assignment discrimination, directed/scoped relation types, qualifier values and explicit composed traversals. Publicly indistinguishable descriptors are deduplicated; conflicting indistinguishable targets remain unscored. Where several answers are supported, the objective sums their acceptable probability rather than arbitrarily selecting one.

Four readouts receive the same public task information: query-only prior, observation-linear, MLP and structured. They test increasingly structured access to observations; their capacities are not asserted to be identical. Descriptor embeddings are learned within training data, not supplied by an undisclosed pretrained semantic encoder.

For brain decoding, native voxels are assigned through the released surface mappers to **Schaefer 200 parcels / 17 networks**. A parcel is a predefined cortical subdivision, not a concept class. The mapping uses maximum unique atlas-label vote from absolute, per-vertex-normalized mapper influence; ties and unmapped locations are excluded. This is an explicit assignment convention, not a probability model of anatomy. It covers only a subset of available native voxels, despite representing all 200 parcels. Encoding and native brain geometry retain the broader native-voxel view; parcel-based claims must report their own coverage.

The default preserves up to eight training-fitted PCA components **within each parcel** before local grounding. Frozen-model coordinates are partitioned into 32 reproducible groups with local projection. These are computational sites, not anatomical homologues of parcels. The shared local interface enables comparisons but does not equate brain/model dimensionality or information content.

The structured readout learns local nonlinear transformations, concept-conditioned unary site evidence and low-rank ordered site-pair evidence. Role binding combines anchor, filler and typed relation support; composition propagates support through explicit steps. Answer loss trains these mappings without a target parcel label. There is no direct query-only output bypass in this architecture, but it can still exploit learned priors. Architecture alone cannot establish neural dependence.

Heldout query-only/linear/MLP comparisons, test-time mismatched observations and separately retrained structured nulls address different alternatives. Test-time mismatch asks whether a fitted decoder needs the corresponding observation; disrupted-pair training asks what it can learn without the intended training correspondence. Within-story model mismatches can still share earlier prefix context and must be interpreted accordingly.

Successful heldout answers, dependence on the matching observation, semantic specificity and reproducibility across participants, contexts and seeds can support **a meaningful association between a semantic distinction and evidence at particular parcels in this decoder**. They do not establish a unique biological concept location, causal necessity, neural connectivity, or literal execution of the decoder's program. This evidential path preserves the intended no-anatomical-label contribution without substituting an attractive map for validation.

## 9. Geometry: five views with different scientific meanings

Human–model comparisons match semantic items and compare their distance structure, usually as representational dissimilarity matrices (RDMs). They do not assume that coordinates in independently fitted spaces have corresponding axes.

| View | What the signature represents | Interpretation limit |
|---|---|---|
| `native` | Measured brain or frozen-model observation coordinates associated with item occurrences, with training-fitted preprocessing. | Context-conditioned observations; co-occurring items may share a vector. |
| `grounding` | Learned unary site profiles or ordered site-pair evidence for a semantic item/relation. | Decoder-derived support. Model sites have no anatomical meaning. |
| `latent` | Learned retrieval-context vectors from queries explicitly linked to the semantic occurrence. | Matching the query to an item does not guarantee an item-specific vector. |
| `latent-passage` | Mean of captured query vectors in a passage/unit. | An explicit passage representation; co-occurring items share it. |
| `encoding-implied` | Heldout brain responses predicted by an encoding model, including defined group contributions. | Geometry of learned predictions, distinct from measured responses. |

The latent correction matters: assigning the mean of every query in a passage to every concept could make an apparent concept geometry mainly a passage geometry. `latent` now selects occurrence-linked retrieval queries without looking at whether their answers were correct. Missing eligible links/traces remain missing. `latent-passage` preserves the passage-level alternative transparently.

There is a further architectural limit: two structured queries at the same event anchor can share a latent; linear/MLP latent projections are query-independent. The correction does not manufacture item-separable embeddings. Ordered grounding profiles supply the more explicit role-specific site-pair view. Whole configurations do not have a separate grounding signature; their component bindings do, and their native/latent/encoding-implied views remain available.

Repeats are averaged within query, distinct matched queries within source/item, sources within story, and stories equally. Selection is not conditioned on successful decoding. Independently fitted latent coordinates are never averaged together; compare their RDMs over common supported items instead.

Both pooled labels and fully scope-matched labels remain available. Pooling retains scope provenance and marginalizes contexts; it does not reinterpret scoped content as actual-world fact. Scope matching can reduce repeated support. Native geometry has raw and training-fitted presentation-residualized variants; the latter is not a complete removal of lexical, narrative or co-occurrence effects.

The shared annotation and query architecture can themselves induce learned similarity. Agreement among learned views is therefore not independent proof of a brain-like organization. Native observations, controls, encoding predictions and cross-context reproducibility help distinguish these interpretations. Reliable dissociations are scientifically useful; missing support is not a measured dissociation.

## 10. Current experimental defaults and generalization

[configs/experiments.json](../configs/experiments.json) records the current full experiment definitions. It is an amendable plan, not evidence that those experiments have finished.

- **Participants and splits:** all nine participants; ten leave-one-development-story-out outer folds with inner story selection; a final fit on development stories and evaluation on story 11 after development choices are fixed. Repeats remain within their story partition.
- **Frozen-model panel:** repository IDs `qwen35-9b-base`, `qwen35-9b-post`, `olmo3-7b-base`, `olmo3-7b-instruct`, `qwen38-27b`. Exact model identities/revisions are in [frozen-models.lock.json](../manifests/frozen-models.lock.json). These replace the old fixed legacy roster. Base/post-trained comparisons and a larger-model condition are useful contrasts, not controlled causal estimates of training effects.
- **Layers:** final block output is primary; fixed quarter-depth positions provide descriptive profiles. Selecting a best layer would require training-only selection, not choosing from final-story performance.
- **Readout repetitions:** seeds 11, 29 and 47. Seeds quantify fit variability; they do not increase the participant count.
- **Context geometry:** complementary development partitions hold out stories 01–05 and 06–10 in turn. Multiple heldout stories share one fitted basis; context analyses require at least two supported heldout stories per item. Report actual common-item coverage and unavailable reliability estimates.
- **Compositional generalization:** explicit traversal tasks and stricter heldout-configuration machinery exist. The latter purges training stories containing selected configurations and checks remaining constituent support. Its scientific inventory is not automatically specified by the default experiment list. Exact heldout combinations and sufficient remaining stories must be established and recorded before claiming systematic generalization.

No heldout story fits a feature vocabulary, scaler, PCA, descriptor embedding, regularization choice, learning rate or stopping decision. Compiling annotations and checking source integrity for story 11 is preprocessing; selecting measurements for favorable story-11 effects would change its evidential role.

A contemporary frozen-model panel avoids making old baselines the entire scientific object. “Frozen” describes the evaluated representation model, not the trained mappings around it. Model size, tokenizer, context and training history remain differences to interpret rather than effects automatically isolated by this panel.

## 11. Statistical units and defensible claims

Queries, voxels, timepoints, model layers, repeats and seeds are not independent human participants. Decoder reports retain per-task and per-story accuracy, candidate-dependent chance, accuracy minus chance and NLL; aggregates balance repeated measurements, source/family contributions and stories. Encoding comparisons retain identical response support. Geometry comparisons report their actual shared semantic items and contexts.

Population intervals average seeds within participant/story cells and use the declared paired participant/story resampling structure. Crossed resampling requires the appropriate complete paired panel. One final heldout story cannot estimate variation over a population of new stories. Thousands of questions do not remove that limitation.

RDM inference permutes semantic labels within the specified analysis, not individual distance cells as if independent. Labelwise testing requires stated multiplicity families. The current configuration calls for declared BH families, but the paper's exact primary contrast families and map-level claims still need to be written explicitly; this handoff does not pretend they are already fully preregistered.

Interpret an effect at the level actually tested:

| Finding | Supported interpretation | Further evidence needed for a stronger claim |
|---|---|---|
| Binding features improve matched encoding | These conjunctions add prediction beyond the matched unbound constituent baseline. | Robustness to alternative representations/readouts and relevant confounds before claiming a uniquely neural binding mechanism. |
| Decoder exceeds query-only and mismatch controls | Matching observations contribute to semantic answer recovery. | Reproducible, specific maps before anatomical interpretation. |
| Stable parcel evidence | A reproducible decoder-associated anatomical pattern under the measured conditions. | Independent interventions or other evidence for causal necessity. |
| Human/model RDM association | Similar ordering of distinctions among the supported semantic items in the named views. | Context and method sensitivity before claiming a shared biological implementation. |
| Weak or absent effect | Limited evidence with this data, support, timing and readout. | Adequate reliability and sensitivity before treating the distinction as absent from a representation. |

Annotation validity, temporal integration, narrative confounding, rare structured combinations, partial anatomical coverage and learned-readout dependence are material limits. They guide the interpretation and targeted analyses; they are not reasons to discard the scientific scope or accumulate unrelated controls.

## 12. Decisions superseded and work that remains scientific

| Earlier assumption or pending item | Current decision |
|---|---|
| Technical proposal governs exact implementation | Researcher decisions and justified scientific definitions govern; proposal details are candidates. |
| Deadline requires a smaller pilot project | Full integrated study; venue follows readiness. |
| Production should use the earlier cheap CLI/rule pipeline | Direct agent-authored corpus plus accepted independent review supplies labels. |
| Human adjudication/downstream adaptation still pending | Review is closed and reviewed-graph compilation/analysis definitions have been integrated. |
| Every contrast needs an all-feature temporal mask | Comparison-specific shared support; all-group intersection is an explicit robustness option. |
| C alone fully matches structural constituent content | C retains introductions/current predicates; BC supplies matched unbound argument marginals. |
| Every latent signature is a concept embedding | Occurrence-linked `latent` and passage-level `latent-passage` are separate; neither guarantees item-separable coordinates. |
| No ground truth means no training | Semantic answers and fMRI responses supervise learned mappings; concept-to-parcel labels are absent. |
| Atlas, readouts, models and splits are all unresolved | Concrete current defaults exist; changes remain possible with explicit rationale. |
| Verification or packaging establishes scientific results | Those establish implementation evidence. Heldout experiments establish empirical findings. |

The next scientific phase is to evaluate the declared branches and synthesize their evidence, without requiring all of them to be positive. It should produce a clear account of:

1. **What was actually measurable:** valid temporal support, training/test label coverage, repeat reliability, common semantic items and parcel coverage for each relevant comparison.
2. **Which distinctions are accessible and predictive:** matched content/binding effects, model-conditional prediction, decoder task results and observation-dependence controls.
3. **Which organizations reproduce:** context/participant/seed stability, anatomical support, and agreement or dissociation across explicitly named geometry views.
4. **What extends beyond the default measurements:** a supported compositional holdout inventory, scientifically motivated sensitivity analyses, or a complementary dataset if a specific unanswered question warrants it.
5. **What the paper can claim:** a prospectively stated primary inference family where possible, clearly identified exploratory analyses, and contributions tied to empirical evidence and a substantive comparison with prior work.

Atlas/partition sensitivity, stronger item-specific latent readouts, further annotation quality measurement and additional corpora are scientific options, not newly imposed prerequisites. In particular, using the accepted corpus does not require another full reviewer loop. If the paper claims a quantified annotation benchmark quality, that specific claim would require corresponding independent measurement beyond the existing acceptance record.

## 13. How to continue without drifting

Read this handoff first, then [experiment_definitions.md](experiment_definitions.md), [reviewed_downstream.md](reviewed_downstream.md) and [encoding_support.md](encoding_support.md) for measurement detail. [analysis.md](analysis.md) describes the readouts and spatial interface, but its historical execution-status paragraphs should not be used as a current operational report. Consult the current configurations and model lock when resolving a concrete method choice.

Keep the older [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) for the original reasoning and pushbacks. Keep engineering status in its separate handoff. When a scientific decision changes, update the relevant definition here and say what it supersedes. When results arrive, distinguish an observed finding from a measurement definition or an interpretation still under examination.

**Continuing brief:** Study the availability and organization of structured linguistic meaning in human fMRI and modern frozen models, using the accepted stimulus graphs and complementary encoding, decoding/grounding and geometry analyses. Preserve scientific ambition, make the measured object explicit, and let real evidence determine the conclusions.
