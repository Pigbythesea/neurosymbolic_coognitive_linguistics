# Structured Meaning in Human Brains and Language Models
## Parallel semantic decoding, neural encoding, and multilevel representational analysis

> **Read first:** [Project handoff: scientific scope and design decisions](../PROJECT_HANDOFF.md) records the researcher's later decisions and is the primary high-level reference. This proposal supplies methodological options and references, not a binding implementation manual. Its claims of authority, fixed model panel, numerical defaults, and approval rules do not override the handoff or subsequent explicit researcher instructions.

**Document type:** Research proposal and implementation handoff  
**Version:** 1.0  
**Source-verification date:** September 30, 2026  
**Intended audience:** Research collaborators, computational neuroscientists, NLP researchers, and coding agents  
**Target venue:** COLING 2027, subject to submission eligibility and completion of the scientific evidence package  
**Execution status:** Proposed study. No neural datasets have been downloaded or analyzed for this proposal, no reported results belong to this project, and the proposed software interfaces have not been implemented.

---

## Navigation

1. [Executive proposal and decision charter](#s01)
2. [Foundations: what the two motivating papers establish](#s02)
3. [Related work and contribution boundaries](#s03)
4. [Research questions, alternative hypotheses, and evidence structure](#s04)
5. [Dataset program and acquisition requirements](#s05)
6. [Semantic annotation: concepts, events, reference, and discourse](#s06)
7. [Queries, supervision, and annotation quality](#s07)
8. [Neural preprocessing, timing, and anatomical scope](#s08)
9. [Language-model panel and representation extraction](#s09)
10. [Parallel measurement E: neural encoding](#s10)
11. [Parallel measurement D: semantic decoding](#s11)
12. [Grounding signatures and representational geometry](#s12)
13. [Compositionality, consistency, and neural-dependence experiments](#s13)
14. [Splits, leakage prevention, and cross-fitting](#s14)
15. [Statistical analysis and reliability](#s15)
16. [Integrated experiment matrix and result interpretation](#s16)
17. [Software architecture and repository specification](#s17)
18. [Data contracts and configuration defaults](#s18)
19. [Algorithms and proposed command-line interface](#s19)
20. [Dependency-based implementation roadmap](#s20)
21. [Compute, storage, staffing, and feasibility](#s21)
22. [Paper construction, venue fit, and release package](#s22)
23. [Risks, unresolved decisions, and completion criteria](#s23)
24. [Researcher and coding-agent handoff rules](#s24)
25. [Appendix A: conceptual explanations retained from the discussion](#a01)
26. [Appendix B: worked example and expected data flow](#a02)
27. [Appendix C: required engineering and scientific-integrity tests](#a03)
28. [Appendix D: decision log and evidence ledger templates](#a04)
29. [References and verified resource registry](#refs)

---

<a id="s01"></a>
## 1. Executive proposal and decision charter

### 1.1 Project summary

This project investigates how structured linguistic meaning is reflected in human fMRI responses and in frozen language-model representations. It connects concepts, role-bound events, cross-sentence reference, and relationships among events through a shared semantic annotation framework. It then studies those representations through **parallel neural encoding and semantic decoding measurements**, together with grounding analysis, representational geometry, and compositional generalization.

The scientific aim is to characterize the relationships among these measurements. A language model may predict substantial neural variation while preserving only some of the semantic distinctions accessible in the brain. A structured readout may improve semantic decoding while leaving aggregate neural encoding unchanged. Concept groundings may be reproducible across contexts, or may change systematically with role and discourse. Agreement and divergence between measurements are both scientifically meaningful outcomes.

The project does not require independently labeled cortical locations for every concept. It does require measured neural responses, trustworthy stimulus-grounded semantic annotations, and experiments that distinguish information extracted from those recordings from information supplied by the annotation ontology, question wording, architecture, or language-model prior.

### 1.2 Broad question

> **How are concepts, role-bound events, and discourse relationships organized in human neural responses and language-model representations, and what do parallel encoding, decoding, grounding, and generalization measurements reveal about their shared and distinct semantic structure?**

This question accommodates several representational descriptions. A concept atlas, a geometry of events, a pattern of encoding–decoding dissociations, or a map of context-dependent accessibility could each become an important empirical result. The proposal does not predetermine a single representational object or expected result shape.

### 1.3 Agreed design commitments

| Commitment | Operational consequence |
|---|---|
| Neural encoding and semantic decoding have equal scientific standing. | Implement and run both branches in parallel. Neither branch is a prerequisite for running or reporting the other. |
| Integrate the approved ideas. | Retain symbolic extraction, hierarchical meaning, brain grounding, LLM comparison, latent geometry, compositionality, and independent neural prediction within one coherent program. |
| Measurement independence is procedural. | Use separately fitted readouts and separate objectives. Shared stimuli and neural recordings mean the resulting statistics can remain dependent. |
| No single expected map or hierarchy is stipulated. | Compare several clearly defined representational signatures and competing explanations. Do not optimize all analyses toward one attractive visualization. |
| Reading is the preferred presentation modality. | Start from a verified multi-story reading resource; keep complementary reading datasets in the design. Listening is a documented alternative when access or semantic coverage requires it. |
| The complete experiment program is specified before outcome inspection. | Use a full registered model panel and planned contrasts. Engineering validation does not become a sequence of outcome-driven, narrowly scoped research pilots. |
| Existing fMRI is the neural evidence. | Synthetic text, synthetic graphs, and predicted fMRI have explicitly different evidential status from measured responses. |
| Grounding interpretations remain calibrated. | Report decoder-dependent support, activity-pattern relationships, and encoding associations separately. Anatomical necessity or a unique neural symbolic code is outside the existing-data claim. |
| No calendar-driven research compression is built into the design. | The roadmap is organized by dependencies and deliverables. Conference dates are administrative constraints. |

### 1.4 Intended contributions

**Scientific contribution.** Determine which semantic distinctions are accessible and predictive, how their organization varies with contextual scale, and where human–model agreement persists or breaks down.

**Methodological contribution.** Establish a jointly interpretable evaluation program whose encoding, decoding, and geometry branches remain independently testable. The individual techniques have substantial precedents; their disciplined integration is part of the proposed contribution.

**Resource contribution.** Release stimulus-aligned semantic annotations, audited query sets, compositional splits, provenance, and reusable analysis software, subject to the source datasets’ redistribution permissions.

**NLP contribution.** Characterize contemporary open-weight model representations across depth, context, and matched base/post-trained checkpoints, using structured meaning and human neural measurements as complementary evaluation targets.

### 1.5 Reading this document correctly

Statements attributed to references describe source-supported prior work. Sections labeled as proposed defaults specify this project’s intended implementation. Mathematical counterexamples and methodological deductions are reasoning, rather than results from the motivating papers. Entries marked **verification required** are unresolved dependencies and must not be silently converted into established facts.

---

<a id="s02"></a>
## 2. Foundations: what the two motivating papers establish

### 2.1 Schrimpf et al.: integrative neural prediction

Schrimpf et al. evaluate 43 language models against three neural language datasets: two fMRI datasets and one ECoG dataset. The same linguistic stimuli are supplied to humans and models. Linear readouts fitted on training stimuli predict held-out neural responses; the paper reports correlations normalized by estimated reliability. Its strongest ceiling-relative results concern sentence datasets, while naturalistic-story prediction is lower. Representational dissimilarity analysis without a fitted model-to-brain map is also included. [R01, pp. 2–3]

The methodological inheritance is stimulus-matched comparison, frozen representations, held-out neural prediction, and comparisons across models and datasets. The present project does not treat a high brain-prediction score as a complete account of semantic organization.

The paper explicitly identifies semantic parsing, conceptual representations, richer context, and more detailed model-to-anatomy relationships as future directions. Those passages motivate this proposal, while subsequent work constrains the novelty claims. [R01, pp. 8–9]

### 2.2 NEURONA: answer-supervised neural decoding with unsupervised intermediate groundings

NEURONA has two distinct data paths:

```text
Dataset construction:
image/video -> scene graph -> symbolic queries and annotated answers

Neural decoding:
measured fMRI + symbolic query
    -> parcel representations
    -> learned concept-grounding modules
    -> compositional executor
    -> predicted answer
```

The diagram on page 4 of the supplied paper illustrates this distinction. The system receives semantic answer supervision, while intermediate assignments of concepts to brain regions have no localization labels. Its training objective is cross-entropy on final answers. [R02, pp. 3–6]

**The released BOLD5000 scene-graph script explicitly invokes GPT-4o.** This was verified in the public code at the file revision recorded in [R03]. The paper describes a pretrained vision-language annotator more generally. This verification does not establish that every video-annotation setting used an identical configuration.

NEURONA does report prediction accuracy. Table 1 on page 7 gives overall fMRI-QA accuracy of 0.7041 for BOLD5000-QA and 0.7046 for CNeuroMod-QA; Table 2 evaluates unseen query compositions. The absent ground truth concerns the intermediate concept-localization maps, rather than the final semantic answers. [R02, p. 7]

The complete system includes learned parcel transformations, a small temporal convolutional encoder, concept classifiers, and symbolic execution. Accordingly, the full architecture is different from a single linear encoding map. [R02, Appendix G]

### 2.3 Important extensions already present in NEURONA

The following cannot be claimed as new merely by adding them:

| Existing element | Location in the supplied paper | Consequence for this project |
|---|---|---|
| Compositional query generalization, including argument swapping and role systematicity | Section 5.1 and Appendix D.3 | Extend the evaluation to language, discourse, and matched brain–LLM representations with stronger stimulus-level holdouts. |
| Cross-subject and cross-atlas analyses | Appendices A–B | Replication is essential evidence, rather than a standalone novelty claim. |
| Symbolic-query–fMRI contrastive embedding and retrieval | Appendix C.2 | A shared embedding or retrieval task alone is insufficient as the new contribution. |
| Correlations between subject, object, and predicate grounding profiles | Appendix D.4, p. 20 | Pairwise grounding correlation has a direct precedent. The proposed contribution requires a richer, independently validated multilevel geometry. |

NEURONA’s appendix labels some evaluations “cross-subject decoding,” but the described analysis fits a separate model to each subject and summarizes performance. That is different from training a decoder on some participants and testing it on an unseen participant. Keep those evaluation types separate in this project. [R02, Appendix B.1]

### 2.4 What the authors’ interpretive limits mean here

NEURONA describes its maps as model-dependent decoding patterns and explicitly says the work does not establish representational compositionality in neural activity. Passive viewing and automated annotations further limit conclusions about participant-driven reasoning. [R02, pp. 9–10]

This proposal preserves that distinction. It can support claims about reproducible neural information, decoder-dependent grounding, and structure-sensitive prediction. A claim that the brain literally implements the symbolic program would require additional evidence.

### 2.5 A specific reason to strengthen the geometry analysis

NEURONA’s guided grounding formulations combine predicate and argument evidence; Appendix G explicitly includes additive combinations. Appendix D.4 then examines correlations among related profiles. [R02, pp. 20, 25–26]

For the illustrative construction

\[
G_{\mathrm{guided}}=G_p+G_s+G_o,
\]

independent, zero-mean, equal-variance components already yield

\[
\operatorname{corr}(G_{\mathrm{guided}},G_s)=1/\sqrt{3}.
\]

This is a mathematical example, not a reanalysis of NEURONA’s reported data. It shows why correlation contributed by the construction should be separated from structure recovered from measurements.

The implementation must save primitive grounding outputs, guidance weights, and composed outputs separately. Geometry tests must identify which representation is being analyzed and which dependencies were imposed by its construction. The main text and appendix use related formulations with different intermediate definitions; a faithful NEURONA reproduction must document the exact pinned-code semantics rather than silently reconciling the equations.

---

<a id="s03"></a>
## 3. Related work and contribution boundaries

### 3.1 Closest experimental precedents

| Work | Relevant experimental contribution | Extension sought here |
|---|---|---|
| Schrimpf et al., 2021 [R01] | Model-to-brain prediction and representational comparisons across models and neural datasets. | Jointly analyze semantic levels, decoding, encoding, and geometry under shared stimulus partitions. |
| NEURONA, 2026 [R02] | Weakly supervised concept grounding and compositional QA from image/video fMRI. | Language and discourse semantics; comparable probes of frozen LLM states; independent activity- and encoding-based evidence. |
| QA-Emb, Benara et al., 2024 [R04] | LLM answers to semantic questions serve as interpretable features for fMRI encoding. | Explicit role bindings and cross-event relationships, plus reverse semantic decoding and multilevel geometry. |
| QA encoding models, Singh, Antonello, et al., 2025 [R05] | The authors report a compact 35-question model evaluated on fMRI and ECoG. | Treat strong interpretable QA features as a required baseline. Detailed replication settings require full-manuscript/code inspection. |
| Toneva, Mitchell, and Wehbe, 2022 [R07] | Computational controls separate aspects of composed meaning from individual-word information, using naturalistic neural data. | Identify which relational structures contribute and compare their organization across systems. |
| Kauf et al., published 2024 [R08] | Manipulates the inputs used for ANN feature extraction for 627 recorded sentences; lexical-semantic content strongly contributes to ANN–brain similarity. | Evaluate relation-sensitive distinctions with lexical content preserved and carefully separate edited-model inputs from actual neural observations. |
| Frankland and Greene, 2015 [R09] | Controlled sentence fMRI studies role-sensitive “who did what to whom” information. | Generalization to natural discourse and systematic comparisons with modern model states. |
| Chen et al., CCN 2024 [R10] | Six participants perform over 1,000 relation-verification trials; six semantic relation types use the same 60 objects. | Event-specific bindings, reference, and discourse dependencies in passive comprehension. A useful controlled comparison, rather than an assumed downloadable dependency. |
| Generative causal testing [R06] | Generates explanations and evaluates them through follow-up recordings with designed stimuli. | Existing-data convergent evaluation of structured meaning; do not describe computational edits as equivalent causal neural tests. |
| Wehbe et al., 2014 [R16] | Models multiple reading subprocesses, including characters and actions, during narrative reading. | The presence of story characters or action annotations alone cannot establish novelty. |
| Chen et al., language timescales [R15] | Examines language integration timescales across reading and listening. | Distinguish temporal span from the type of semantic composition; test their interaction. |
| Hernandez et al., ICLR 2024 [R26] | Studies linear relational decoding in transformer representations. | Compare relational information across brain and model measurements, beyond testing linearity in LLMs alone. |
| AMR/UMR [R20–R21] | Established graph-based semantic representations; UMR includes document-level reference, modal, and temporal information. | Reuse semantic representational ideas with neural alignment and evaluation-specific provenance. Avoid claiming a new semantic formalism solely because it contains graphs. |

A 2026 expanded semantic-relations preprint was located [R11]. Its complete content and trial-level data access were not verified. The detailed comparison above relies on the accessible 2024 report. The 2025 QA-encoding manuscript also remained incompletely retrievable in this verification session; its authors’ project description establishes its existence and broad claims. These are explicit literature-verification tasks before manuscript submission.

### 3.2 Defensible novelty formulation

The proposed contribution is the **experimentally grounded integration** of:

1. A shared multilevel semantic reference for recorded language stimuli.
2. Independently fitted neural encoding and semantic decoding branches with equal reporting status.
3. Concept-, role-, event-, and discourse-level geometry derived through distinguishable measurement routes.
4. Tests of systematic generalization, neural dependence, and architecture-induced geometry.
5. A modern open-weight model panel evaluated with matched stimulus access and frozen representations.

The paper’s strongest novelty will depend on the empirical finding produced by this program. Method integration and a dataset are meaningful deliverables, but a top-venue scientific paper should also reveal a substantive relation or dissociation that existing aggregate evaluations obscure.

### 3.3 Claims to avoid

Do not claim that symbolic annotations, linear fMRI prediction, concept decoding, RSA, role-sensitive brain representations, or LLM–brain comparison are individually new. Do not describe recent open-weight models as the first models whose internal states can be inspected. Their value here is a broader and more informative comparison panel.

The project also does not establish a universal semantic metric simply by computing cosine distances. It evaluates several hypotheses about what those distances reflect.

---

<a id="s04"></a>
## 4. Research questions, alternative hypotheses, and evidence structure

### 4.1 Research questions

| ID | Question | Measurements |
|---|---|---|
| RQ1 | Which concepts, role assignments, reference relations, and event links are recoverable from human neural responses and frozen LLM states? | Semantic decoding, calibration, task-specific confusion, held-out composition. |
| RQ2 | Which levels of semantic annotation explain held-out neural responses beyond lexical content and presentation-related variables? | Neural encoding, feature-group contrasts, joint models with LLM features. |
| RQ3 | How are concepts and structured semantic configurations organized within each representational system? | Grounding, activity-based, and encoding-implied geometries, with explicit estimands. |
| RQ4 | Which relationships replicate across contexts, participants, corpora, readouts, and annotation realizations? | Split-half/context replication, independent fits, occupancy-matched nulls, cross-corpus agreement. |
| RQ5 | How do encoding, decoding, and geometry agree or diverge across model depth, context, and post-training? | Paired profiles and contrasts; family-aware comparisons. |
| RQ6 | How does the relationship between semantic structure and temporal context vary across cortical regions and model representations? | Context-conditioned encoding/decoding, region-specific geometry, discourse-span analyses. |

### 4.2 Competing explanations

**Content-dominated account.** Apparent structural agreement is largely attributable to shared concepts, topic, and lexical associations. Structure-aware features offer limited reproducible information once those variables are accounted for.

**Structure-accessibility account.** Role and discourse information is present in both systems, while a structured readout makes it easier to access. Gains may be stronger for decoding than for aggregate neural prediction.

**Shared-organization account.** Brain responses and some model states exhibit similar relational geometry that survives lexical/contextual controls and independent data partitions.

**Partial-overlap account.** Systems agree at one semantic level and diverge at another. For example, event content could align while reference-sensitive or cross-event relationships differ.

**Measurement-induced account.** Attractive grounding structure mainly reflects architecture, ontology, readout training, or concept-label priors. It weakens under neural disruption or changes substantially across measurement routes.

These are competing explanations to assess; the proposal does not assume one is true.

### 4.3 Evidence dimensions

Use the following labels in code and reports:

- **E:** Predict measured fMRI from stimulus-derived semantic or model features.
- **D:** Predict stimulus-grounded semantic answers from measured fMRI or frozen model states.
- **G:** Estimate relationships among concepts or structured configurations within each representation.
- **K:** Test compositional generalization, contextual consistency, specificity, and neural dependence.

E and D have separate objectives, parameter stores, tuning records, and evaluations. G and K connect and interrogate their outputs as well as measured/native representations. No combined “overall brain-likeness” score is required.

### 4.4 What independence does and does not mean

An encoding fit is not obtained by algebraically inverting a decoding fit. A decoding loss does not optimize the encoding weights. A brain–LLM geometry-matching loss is absent from the registered independent-comparison analysis.

Nevertheless, E, D, and G may use the same stimuli, labels, and neural recordings. Their errors and statistics can therefore be dependent. Statistical analysis must preserve that dependence through paired and clustered inference. The phrase “independent measurements” refers to distinct fitted measurement routes, not a claim of probabilistic independence.

### 4.5 Plausible informative outcomes

| Encoding result | Decoding result | Geometry/replication result | Supported interpretation |
|---|---|---|---|
| Structure improves prediction. | Structure improves decoding. | Relational organization replicates. | Convergent evidence for structure-sensitive information under multiple measurement routes. |
| Little unique encoding gain over an LLM. | Reliable structural decoding. | Shared geometry appears in that LLM. | The LLM may already contain the relevant predictive information; interpretable structure identifies part of it. |
| Reliable encoding association. | Weak structured decoding. | Activity geometry is reliable. | Predictive association may be distributed or difficult to recover with the tested decoder, or labels may be too fine-grained. |
| Limited encoding gain. | Strong structured decoding gain. | Grounding geometry varies with architecture. | A useful structured decoder with limited evidence for a stable anatomical interpretation. |
| Both branches perform well on broad content. | Relational contrasts are weak. | Agreement disappears after content controls. | Broad semantic alignment is supported; stronger relational claims are unsupported. |
| Brain–model agreement is weak. | Human and model readouts each perform reliably. | Within-system geometry replicates. | A genuine representational divergence is a candidate explanation. |

---

<a id="s05"></a>
## 5. Dataset program and acquisition requirements

### 5.1 Choose neural recordings before building a large text corpus

Every neural analysis must ultimately be tied to text that participants actually encountered during scanning. An external text corpus can support parser development, annotation validation, or text-only representation training. It contributes no additional measured fMRI samples.

The dataset program has complementary roles: multi-story estimation, independent narrative corroboration, and reliable sentence-scale analysis. These dataset roles do not assign scientific priority to E or D.

### 5.2 Candidate registry

| Dataset key | Source-supported description | Intended use | Verification state |
|---|---|---|---|
| `deniz_reading` | The documented public subset has six participants, eleven stories, ten training stories and one test story, with TR 2.0045 s. The accompanying repository lists subject IDs `01, 02, 03, 05, 07, 08`. [R13–R14] | Multi-story reading; estimate and compare all measurement families. | Literature and repository documentation verified. File-level download, full metadata, preprocessing, and permissions remain unverified. |
| `wehbe_reading` | Eight participants read chapter 9 of *Harry Potter and the Sorcerer’s Stone*, one word at a time for 0.5 s. Original and updated data routes are documented. [R16] | Independent reading corroboration; recurring reference and event analyses; contiguous-block validation. | Author page verified. Choose and inspect an exact release before ingestion. |
| `pereira2018` | 384 and 243 sentences in two experiments; 10 unique participants across the two experiments. Sentences form short passages and were presented repeatedly. [R01, pp. 8–9; R17] | Repeated-sentence geometry and sentence/event-level analyses where labels are supported. | Counts and paradigm verified in the supplied paper. Exact downloadable representation and repetition availability require inspection. |
| `lebel_listening` | Eight participants listened to 27 narratives, approximately six hours; raw/preprocessed data, surfaces, timed transcripts, and encoding resources are documented. [R18] | Larger within-person alternative or cross-modality corroboration when justified by coverage/access. | Release documentation verified; local data acquisition not performed. |
| `narratives_listening` | The published collection reports 345 participants, 891 scans, and 27 stories; participant exposure is heterogeneous. [R19] | Broader population/corpus replication using a prespecified compatible subset. | Publication verified; exact subset and local access require inspection. |

A documented subset is not a guarantee that every original participant or every derivative file is available through the same release. Keep original-study counts, derivative-subset counts, acquired-file counts, and analyzed counts separate.

### 5.3 Acquisition routes

For Deniz reading, verify both author-documented routes:

```text
https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri
https://berkeley.app.box.com/v/Deniz-et-al-2019
```

Both links are documented by author-associated repositories. The GIN route timed out during verification; the Box route could not be accessed through the browsing tool. Neither failure proves that the dataset is generally unavailable. Neither route has been verified as a successfully downloaded dataset for this proposal. [R14–R15]

For Wehbe, follow the author page’s original Dryad route or its updated-data route. The author page describes differences in preprocessing and spatial normalization. Select one release deliberately and retain its documentation; do not combine arrays from different releases based on similar filenames. [R16]

For Pereira, recover the original study’s release and/or the neural-NLP benchmark representation, then check what is available: individual repetitions, averaged beta estimates, localizer masks, passage identities, and stimulus order. An average-response derivative does not support repetition-based distance estimation by itself. [R01; R17]

### 5.4 Required acquisition manifest

Each acquired dataset must have a machine-readable manifest containing:

| Field | Requirement |
|---|---|
| Source and version | Canonical resource identifier, release/version, download date, checksums. |
| Permission | Dataset terms, derivative-data terms, text/audio copyright conditions, citation requirements. |
| Participants | Pseudonymous IDs and actual available runs; no invented IDs or assumed subject matching. |
| Stimuli | Exact transcript text, story/passage IDs, modality, order, and timing source. |
| Neural arrays | Shape, voxel/vertex identity, units, TR values, acquisition times, preprocessing history. |
| Anatomy | Native/template space, hemisphere labels, masks, surfaces, registrations, and atlas transformations. |
| Reliability | Repetition IDs, repeated-story conditions, localizers, available motion/quality variables. |
| Existing splits | Official training/test assignments and any conditions already used for tuning in prior pipelines. |
| Overlap | Text fingerprints for overlaps with other candidate datasets. |

### 5.5 Corpus-selection rule

Use Deniz reading as the proposed multi-story corpus when file verification and semantic coverage support the planned analyses. Retain Wehbe and Pereira for their distinct experimental strengths.

If the reading corpus cannot be acquired or lacks sufficient independent instances of a targeted phenomenon, record the limitation and evaluate the already specified listening alternative. The decision must use access, stimulus coverage, annotation quality, and measurement reliability rather than favorable E/D results.

No speech model is required for a listening alternative: the LLM input remains the transcript. Acoustic covariates become additional feature groups and controls. A switch in presentation modality must be reflected in the hypothesis and manuscript wording.

### 5.6 Cross-corpus independence audit

Narrative datasets can reuse public stories. Compare exact transcript hashes, normalized n-gram overlaps, story titles, and manual matches. Shared stories and shared participants do not count as fully independent replication. Publish the overlap table and restrict the relevant replication claim accordingly.

For model-pretraining contamination, record that public narratives and literary text may have appeared in training corpora. The study examines pretrained representations; it cannot generally guarantee novel-to-model stimuli. Strong human–model generalization claims must not rest on an unverified absence of pretraining exposure.

---

<a id="s06"></a>
## 6. Semantic annotation: concepts, events, reference, and discourse

### 6.1 Three distinct meanings of hierarchy

**Taxonomic hierarchy:** a book is a physical object, which is an entity.

**Compositional hierarchy:** entities fill event roles; events participate in reference, temporal, causal, and other discourse relationships.

**Temporal hierarchy:** information is accumulated across words, clauses, sentences, and extended passages.

The project represents all three explicitly where supported. It does not equate a longer text window with more compositional structure or assume an ordered cortical progression for every hierarchy.

### 6.2 Proposed semantic scaffold

Use an AMR/UMR-informed typed graph, with an evaluation-specific provenance layer. AMR and UMR are methodological precedents for semantic graphs and document-level relationships. This project’s schema is a controlled subset adapted to alignment and testing; it is not a claim to have invented either formalism. [R20–R21]

| Component | Required fields | Examples |
|---|---|---|
| Concept type | Canonical label, sense/frame identifier when available, coarse semantic type. | `book`, `lend`, `return`, `person`. |
| Entity instance | Story-local ID, mentions, available attributes, type membership. | The particular book and the particular people in this story. |
| Mention | Exact character/word span, availability time, referring expression. | `Leah`, `he`, `it`. |
| Event instance | Predicate, role-labeled arguments, supporting spans, factuality, polarity, time information. | `lend(agent=Leah, recipient=Noah, theme=book)`. |
| Reference relation | Mention-to-entity or event-to-event reference, support, uncertainty. | `he -> Noah`; later `book` refers to the previously lent item. |
| Discourse relation | Typed ordered edge, endpoints, supporting text, explicit/inferred status. | `before(e1,e2)`, `motivates(e2,e3)`. |
| State update | Which relation or property becomes available, is revised, or is explicitly revoked. | A returned object changes the described possession state. |
| Provenance | Annotator/version, input-prefix boundary, source span, confidence, audit status. | A relation supported by the word `because`. |

Canonical roles should distinguish agent, patient, theme, recipient, experiencer, stimulus, location, and time where appropriate. Do not map every first argument to an agent. Preserve lexical frame roles as well as any coarser role mapping so alternative normalizations can be compared.

### 6.3 Scope and factuality

The graph must represent negation, reported speech, hypothetical events, and uncertainty. “Noah hoped to return the book” and “Noah returned the book” cannot share the same asserted-event label. A future intention is distinct from a completed event.

Temporal adjacency does not license causation. Include `causes` or `motivates` as confidently supported only when the text or adjudicated annotation supports that interpretation. Maintain separate `explicit`, `inferred`, and `unresolved` statuses.

Graph shape is flexible. Reference and discourse structures can contain reentrancy and different relation types. Only subgraphs whose semantics require acyclicity, such as a strict `before` relation, should receive a cycle prohibition.

### 6.4 Incremental availability

Each annotation has an `available_at_word` field recording when its supporting information has been presented. The annotation generator must only see the relevant prefix when producing an online-availability label.

Do not parse an entire story, discover a later revelation, and retroactively attach that interpretation to earlier neural time points. Full-document annotations may be retained in a separately named retrospective view. They cannot be used as though a reader knew the future.

The proposed annotation procedure is to present a growing prefix or a prefix plus a persistent, prefix-derived entity ledger to the annotator at clause/sentence endpoints. A ledger may contain only information available before the current endpoint. Its update history must be saved so future-dependent corrections are detectable.

### 6.5 Distinguish type identity from story identity

A story-local name or entity identifier does not transfer automatically across stories. Separate questions about a semantic type, such as `person`, from questions about a specific participant in a specific event.

For cross-story geometry, prioritize semantic types, predicate senses, role-conditioned types, and event schemas with enough independent occurrences. For within-story reference analyses, local entity instances are legitimate targets. Do not combine all names into a purported universal person-concept vector without defining the aggregation.

Absolute story position, global event IDs, and arbitrary database identifiers must not become semantic features. Keep them as join keys only. Time can enter explicitly registered temporal analyses and nuisance controls.

### 6.6 Annotation production and independent audit

Proposed open-weight annotators are `Qwen/Qwen2.5-32B-Instruct` for production and `meta-llama/Llama-3.3-70B-Instruct` for independent audit. Their official model cards were verified [M13–M14]. They are outside the evaluated checkpoint panel, although their model families overlap with panel members; family-related annotation bias therefore remains a control target.

Freeze model revisions, prompts, schema, generation settings, context policy, and normalization rules before outcome evaluation. Generate constrained JSON with source spans and explicit uncertainty. A JSON-valid output is not automatically semantically correct.

Use human adjudication for a stratified sample of complete contexts and for disputed role/reference/discourse labels. Proposed audit budget: approximately 600 contexts distributed across corpora, semantic levels, rare predicates, negation, and long-distance dependencies, with two independent ratings and adjudication. This is a planning default, not a measured requirement or a guaranteed sufficient sample size. Report confidence intervals for annotation error by phenomenon and preserve a fully adjudicated evaluation subset.

All test-set annotation work is blinded to E/D predictions and brain maps. Annotation repair changes the annotation version, triggers affected evaluation reruns, and is logged. It must not be targeted to examples where a favored model fails.

### 6.7 Coverage audit

Before fitting neural targets, summarize unique concepts, role combinations, reference distances, event-link types, lexical realizations, independent contexts, stories, participants, and repeated neural observations.

A proposed eligibility rule for a reported concept-level cross-story geometry is at least 20 distinct contexts spanning at least three stories, with sufficient observations in each estimate partition. These numbers are transparent planning defaults. Publish the full coverage distribution and uncertainty, and retain insufficiently sampled concepts as descriptive results with an explicit eligibility flag. This rule does not define a biological threshold.

For the single-chapter corpus, analogous within-chapter criteria must be labeled as such. A within-chapter result cannot satisfy a cross-story criterion merely because it has many windows or questions.

---

<a id="s07"></a>
## 7. Queries, supervision, and annotation quality

### 7.1 Query families

| Family | Target | Example | Crucial control |
|---|---|---|---|
| Concept occurrence | Whether a concept is mentioned or an event is described in a bounded window. | “Is a book mentioned in this segment?” | Lexical occurrence and query-only baselines. |
| Event role | Which entity/type fills a specified role. | “Who lent the book?” | Same entities with a different role assignment. |
| Binding verification | Whether a particular predicate–argument configuration is supported. | “Did Leah lend the book to Noah?” | Role-reversed and predicate-matched alternatives. |
| Reference | Which earlier entity/event a current expression refers to. | “Does ‘he’ refer to Noah?” | Recency, gender/number, mention frequency, and candidate-prior baselines. |
| Temporal relationship | Relative event order explicitly supported by the passage. | “Did the reminder occur before the return?” | Surface mention order versus described event order. |
| Explicit causal/motivational relationship | A supported relation between events or propositions. | “Is the reminder motivated by Leah’s stated need?” | Connector presence and concept-matched unrelated event pairs. |
| Composed discourse query | A combination of reference, role, and event-link information. | “Did the recipient of the loan later return the same item?” | Constituent questions, broken reference links, and graph-depth controls. |

Query complexity must be tracked separately from semantic level. A query with more logical operators is not automatically about a more abstract concept.

### 7.2 Logical query representation

Use a typed abstract syntax tree (AST). Store canonical structured queries separately from optional natural-language renderings. Example operators include:

```text
Mentioned(concept, window)
VerifyEvent(predicate, ordered_role_bindings, scope)
ChooseRole(event_descriptor, role, candidate_set)
Corefers(mention_descriptor, entity_descriptor, scope)
EventRelation(relation, event_a_descriptor, event_b_descriptor)
And(query_a, query_b)
Exists(variable_type, predicate_query)
```

The query compiler is deterministic. It must reject malformed argument roles, unavailable references, and unsupported operator types. Save the canonical AST hash so paraphrased surface questions cannot silently cross a query holdout.

### 7.3 Decoder access contract

At evaluation, a neural decoder may receive the measured neural input, a query, the query’s legal candidate descriptors, and the temporal scope required to interpret it. It must not receive the source passage, the full annotated stimulus graph, the answer, or evidence spans that reveal the answer.

A frozen-LLM decoder receives cached stimulus representations and the same query/candidate information. It must not invoke the LLM again after seeing the query in the representation-accessibility analysis.

The full graph is used to generate labels, audit semantics, and construct E-branch features. A loader that includes that graph in a D-branch model input would invalidate the task.

### 7.4 Candidate descriptors and identity leakage

Candidate lists may themselves reveal information. For role decoding, a list of two person names is a legitimate task input only if an identical candidate list is provided to the query-only control. Randomize answer order with a saved seed and balance correct positions.

For unseen story-specific entities, use permutation-equivariant candidate scoring rather than an ever-growing classification head indexed by global entity IDs. Query descriptors may contain a presented name or a neutral mention descriptor; they must not contain the correct binding or a description such as “the person who returned the book” when that relationship is the target.

Include a fixed-vocabulary concept/type task and a candidate-based instance task as distinct outputs. They test different generalization properties.

### 7.5 Open-world labels and negatives

For a closed-window mention task, absence of a mention can define a negative. For event truth or discourse interpretation, absence from an incomplete graph does not establish falsity.

Use `supported`, `contradicted`, and `undetermined` where appropriate. Define the annotation semantics clearly: these labels concern textual support within the stated scope. They are not measurements of a participant’s belief or comprehension.

Generate challenging alternatives by role reversal, referent substitution, predicate substitution, and event-link alteration. Revalidate each negative against the source. A role reversal can also be true elsewhere in the passage; in that case it is not a valid negative for that scope. Ambiguous alternatives are excluded from binary scoring or retained under `undetermined`.

Use balanced binary or controlled-choice tasks where their labels are justified, and report the label distribution for every query family. Avoid a benchmark dominated by easy absence questions.

### 7.6 Loss weighting and effective sample size

Several queries can be generated from the same neural window. They add supervision diversity without adding independent neural observations.

Training should first weight each source window, then average its query losses within query family, then average across the registered families. Cap repeated near-identical query templates or downweight them. Evaluation reports both per-family scores and a macro summary; the macro summary does not replace the separate results.

Log all of the following: number of participants, stories, distinct source windows, distinct semantic instances, queries, and template types. Never use query count as the sole sample-size description.

### 7.7 Annotation outcomes to release

Release inter-annotator agreement, adjudication outcomes, role/reference/discourse error rates, ambiguity rates, and the provenance of every evaluation answer. For graph extraction, report component-level precision/recall and complete-event binding accuracy, rather than a single graph similarity number alone.

Preserve alternative valid readings when adjudicators agree that the passage is ambiguous. Score against an explicit acceptable-answer set where appropriate. Do not force ambiguity into an artificial unique answer to make the benchmark easier to implement.

---

<a id="s08"></a>
## 8. Neural preprocessing, timing, and anatomical scope

### 8.1 General policy

Use documented released preprocessing where it is adequate for the analysis, preserving its provenance. Additional preprocessing must be specified by dataset and applied without using semantic test labels. Standardize representation and timing interfaces, while respecting different acquisition paradigms.

The naturalistic encoding literature aligns stimulus features to scan times and models delayed BOLD responses. This is the methodological basis for temporal alignment here. [R18; R22]

### 8.2 Required inspection

Check array orientation, TR units, dropped volumes, run boundaries, synchronization offsets, censoring, spatial space, and whether the data are time series or sentence-level estimates. Verify that the timestamp assigned to a volume corresponds to the intended acquisition reference.

Check whether a supplied “reading” file is truly the reading condition. Confirm subject identifiers through metadata rather than filename resemblance. Verify repetition identities before averaging or treating observations as independent.

For updated Wehbe data, the author documentation describes native-space data without smoothing, detrending, or masking. The adapter must therefore explicitly provide the selected preprocessing rather than assuming a normalized, fully processed derivative. [R16]

### 8.3 Proposed preprocessing defaults

| Component | Proposed default | Required safeguard |
|---|---|---|
| Spatial smoothing | No additional smoothing for multivariate pattern analyses. | Document any smoothing already present in the source. |
| Motion/quality | Use released confounds and acquisition-quality information where available. | Censoring cannot silently destroy stimulus–volume correspondence. |
| Drift | Preserve released filtering; for minimally processed data, use a declared low-order run drift model. | Report the passband and sensitivity of discourse-timescale conclusions to filtering. |
| Scaling | Fit feature scaling on training data; save transforms. | No test-derived PCA, feature selection, or neural reliability threshold. |
| Run-specific nuisance fitting | Permitted for label-free offline preprocessing, with the convention recorded. | Distinguish such preprocessing from cross-validated learned prediction. |
| Voxel selection | Anatomical/localizer masks; any additional reliability selection uses independent localizers or training folds. | The same selection applies to all models in a comparison. |
| Missing/censored data | Explicit masks propagated into all designs and losses. | Never fill missing neural observations with model predictions and label them measured. |

Aggressive high-pass filtering can attenuate slow discourse-related variation. Report the temporal bandwidth that survives preprocessing rather than claiming sensitivity to arbitrarily long timescales.

### 8.4 Anatomical analysis views

Maintain a full available cortical/brain measurement view, a region-based view, and a parcel-based decoder view.

When a participant-specific language localizer exists, use it for functionally defined language regions. When it is absent, use an external anatomical/probabilistic definition and label it accordingly. An anatomical mask is not a participant-specific functional localizer.

Include language-associated, default-mode/association, and sensory comparison regions where supported by coverage and the selected atlas. These are registered analysis masks; the proposal does not assume that every discourse effect will fall inside a narrow language mask.

A proposed decoder resolution is Schaefer-200 where valid transformations exist, with a coarser network-level sensitivity analysis. If the release only supports a different valid atlas or native-space mask, record that limitation and use an explicit adapter. Do not silently infer voxel correspondence or pad unrelated parcels to make them appear equivalent.

Within-parcel multivariate information should be retained. A parcel mean can erase distinctions among concepts that use the same broad network. Proposed input reduction is up to eight training-fitted components per parcel, with the actual rank bounded by the available voxels and training samples.

### 8.5 Temporal feature construction for encoding

Let a word or semantic update become available at time \(\tau_i\). Construct a stimulus feature process using observed onset/duration or update times. Bin features to the actual scan grid by duration-weighted overlap, then create a finite-impulse-response design.

Proposed lag settings are one through four TRs, recorded in seconds for each dataset. For a roughly 2 s TR, these correspond approximately to 2–8 s. These are initial modeling defaults; any lag choice is selected or compared using training/validation partitions, and is applied consistently across matched feature spaces.

\[
\widehat{Y}(t)=\sum_{g}\sum_{\delta\in\mathcal D}X_g(t-\delta)W_{g,\delta}+\epsilon(t).
\]

Do not cross run boundaries when constructing delays. Trim or mask incomplete boundary support. Sentence-level beta datasets use their presentation/estimation structure rather than receiving an additional arbitrary HRF convolution.

Use a causal availability view for substantive claims about information available at an endpoint. A symmetric smoothing/downsampling filter may use later words; if used to reproduce an existing benchmark, record that support and keep its interpretation separate from the causal-availability analysis.

### 8.6 Neural observation windows for decoding

For a source segment ending at \(t_e\), a proposed decoder input samples BOLD at the registered delayed times, approximately \(t_e+2\) through \(t_e+8\) seconds where acquisition permits. For longer source segments, use a fixed, documented temporal aggregation rather than changing the input window by query difficulty.

This is an **offline decoding measurement**. Later BOLD samples can contain responses to subsequently presented text because the hemodynamic response overlaps in time. A prefix-restricted annotation alone does not eliminate this contamination.

Required analyses are:

- A declared source-time support and acquisition-time support for every decoder example.
- Controls using nearby/suffix text features to estimate how much local temporal context could explain a result.
- Matched windows for competing queries, with boundary and lag sensitivity specified in advance.
- A distinction between recently presented event content and discourse-state labels that depend on earlier material.

The study should not claim millisecond online construction or a uniquely isolated neural response to one word. Long-range content recoverable from a BOLD window may reflect ongoing representation, reactivation, correlated continuation, or temporally overlapping processing; the planned controls constrain these explanations without guaranteeing complete separation.

### 8.7 Anatomical support is different from physical necessity

A parcel ablation measures dependence of a particular decoder on the information available through that parcel. Correlated parcels and out-of-distribution masking can affect the result. Train-without-region comparisons and conditional perturbations can strengthen the interpretation, but neither is equivalent to a biological lesion.

For linear decoders, distinguish predictive weights from activation-pattern interpretations. Haufe et al. explain why discriminative weights can differ from the underlying neural patterns. [R23]

---

<a id="s09"></a>
## 9. Language-model panel and representation extraction

### 9.1 Prespecified panel

The panel contains historical reference models, modern decoder families, matched base/post-trained pairs, and within-family scale contrasts. It is a controlled comparison set, rather than a claim to include every newly released model. Official model cards for the following identifiers were checked; execution still requires exact revision hashes, local loading verification, and applicable access terms. [M01–M12]

| ID | Checkpoint | Comparison role |
|---|---|---|
| M01 | `openai-community/gpt2-xl` | Historical autoregressive reference linked to Schrimpf-era results. |
| M02 | `google-bert/bert-base-uncased` | Historical bidirectional-encoder reference. |
| M03 | `Qwen/Qwen2.5-1.5B` | Smaller within-family base model. |
| M04 | `Qwen/Qwen2.5-7B` | Base model for scale and post-training comparisons. |
| M05 | `Qwen/Qwen2.5-7B-Instruct` | Matched-family instruction-tuned comparison. |
| M06 | `Qwen/Qwen3-8B-Base` | More recent base-model generation. |
| M07 | `Qwen/Qwen3-8B` | Corresponding post-trained model. |
| M08 | `meta-llama/Llama-3.1-8B` | Independent base-model family. |
| M09 | `meta-llama/Llama-3.1-8B-Instruct` | Base/post-training comparison within Llama. |
| M10 | `google/gemma-2-2b` | Smaller base model in another family. |
| M11 | `google/gemma-2-9b` | Larger base-model comparison. |
| M12 | `google/gemma-2-9b-it` | Corresponding instruction-tuned comparison. |

Add context-independent lexical embeddings, random features, and selected random-initialization architecture controls as baselines. These do not replace panel members. Any unavailable checkpoint is recorded as unavailable; substitution requires a documented scientific reason before outcome comparison.

### 9.2 Frozen-state policy

Keep foundation-model weights frozen in E, D, and independent G analyses. Readout training is permitted and saved separately. Fine-tuning on fMRI would answer a different question and is outside the registered core comparison.

Use raw stimulus text for all checkpoints, without question-specific instructions, generated explanations, or answer tokens. A separate native-chat behavioral QA task may be useful, but its results must be labeled as prompted task performance and kept apart from representation accessibility.

For models with thinking/non-thinking generation modes, the frozen-representation analysis performs a forward pass over stimulus text and does not generate a reasoning trace. Generation-mode settings matter for any separately registered behavioral task or annotation pipeline. [M07]

### 9.3 Matched information access

For a word endpoint, extract states only from the prefix available at that endpoint. For BERT, this requires prefix-restricted forward passes; processing a whole sentence and extracting an earlier token would expose future text. A retrospectively contextualized BERT analysis may be retained as an explicitly separate condition.

No test question is supplied before caching the state. Do not let one model reread an entire passage with the query while another is evaluated through a frozen stimulus-only representation.

### 9.4 Context conditions

Proposed shared conditions are the last 32 words and the last 128 words, including the current word. Proposed extended context is 512 words for the ten modern checkpoints. A separately registered 2,048-word condition can investigate longer discourse for compatible models if coverage and resources justify it.

The 128-word condition provides a shared comparison across the complete panel. Token limits are checked per checkpoint and per example. An overlength example must be flagged; it cannot be silently truncated and still labeled as the same context condition. Matched comparisons use identical eligible stimulus subsets and report coverage.

These windows are computational manipulations, not claims about a human reader’s exact memory span. Earlier portions of a story were still presented to the human participants.

Strict sliding-context extraction must avoid stale key/value-cache information from words outside the declared window. Reuse a cache only when it is exactly equivalent to the specified input computation; otherwise recompute the relevant prefix. Approximate streaming policies require a separate condition label.

### 9.5 Layer definitions

Use six prespecified depths: input embeddings and block-output positions nearest 20%, 40%, 60%, 80%, and 100% of model depth. De-duplicate indices where necessary. Record the exact module path, residual-stream location, and whether a terminal normalization has been applied.

“Last layer” must have a consistent implementation definition. Some libraries expose final normalized states in an output tuple while intermediate states correspond to raw block outputs. Use hooks or explicit adapters to distinguish these representations. Keep terminal-normalized output as a separate feature when it differs.

The registered six-depth grid is evaluated across the entire model panel. Dense all-layer profiles may be cached and reported as an additional analysis when computationally available; they do not license test-set layer selection.

### 9.6 Temporal and semantic pooling

Store word-aligned hidden states, with exact word-to-subtoken offsets. A proposed default is the last subtoken state for each current word; subtoken-mean pooling is a declared sensitivity condition.

For E, aggregate word-aligned features to scan times using the same temporal support rules as other stimulus features. For D, create fixed window representations from the cached states, preserving a small number of ordered bins where relevant. For G, extract concept/event-associated states using the independently defined source spans and endpoints.

Do not compare an average over the full story on one side with a local event response on the other without an explicit contextual matching analysis.

### 9.7 Representation controls

Required controls include a bag of lexical content, a context-independent embedding representation, a randomized feature representation with matched dimension, and restricted-context variants. Selected random-initialization controls estimate architecture-related structure; their interpretation remains distinct from trained models. The existence of nontrivial untrained-model brain prediction is a precedent in Schrimpf et al. [R01, pp. 5–7]

For instruction-tuned/base comparisons, use identical raw text, layers, context policy, precision, and evaluation examples. Treat observed differences as checkpoint-associated effects; available releases generally do not isolate every aspect of the post-training procedure.

---

<a id="s10"></a>
## 10. Parallel measurement E: neural encoding

### 10.1 Objective

Predict measured held-out neural responses from stimulus-derived features. This branch quantifies which semantic and model features capture neural variation and how their contributions relate to structured decoding and geometry.

Its success criterion is reproducible, interpretable prediction and informative feature comparisons. The project does not require a global state-of-the-art encoding score.

### 10.2 Feature groups

| Code | Feature group | Proposed construction |
|---|---|---|
| `N` | Presentation and nuisance features | Word/letter rate, word length, timing, sentence boundaries, available motion/drift terms; acoustic groups only for listening. |
| `L` | Lexical/control features | Content-word counts, context-independent embeddings, lexical frequency or a documented corpus proxy, syntactic indicators where relevant. |
| `C` | Flat semantic content | Concept/predicate occurrence without role or cross-event links. |
| `B` | Within-event binding | Ordered predicate–role–argument features and complete event configurations. |
| `R` | Reference structure | Mention/entity links, antecedent type, reference distance, persistence/reintroduction markers. |
| `D` | Discourse relationships | Typed ordered event/proposition links and annotated update structure. |
| `Q` | QA-style interpretable features | Fixed semantic question features modeled after QA-Emb/QA encoding baselines. |
| `H[m,l,c]` | Frozen model states | Model `m`, depth `l`, context `c`, aligned to the same stimuli. |

Local presentation variables and lexical-semantic variables are retained as separate groups, allowing different controls to answer different questions. A word-frequency feature must have a declared external source or be estimated only from training text; missing licensed resources are not silently replaced by fabricated values.

### 10.3 Numeric representations of structure

Implement two specified routes.

**Sparse interpretable route.** Use indexed concept, predicate, role, and relation features. Ordered bindings have distinct indices. Preserve the mapping from each column to its semantic definition. Vocabulary selection and pruning use training coverage or an externally frozen ontology.

**Dimension-controlled route.** Apply a fixed, seeded, label-independent projection to those sparse features, or use a separately trained text-only graph encoder. Maintain independent projections for feature groups so role permutations remain distinguishable. Trainable text-only graph encoders use training text and external text under a documented policy, without neural targets.

The proposed compact dimensions are `C=512`, `B=1024`, `R=512`, and `D=512`, bounded by available rank and vocabulary. These are engineering defaults. Explicit sparse features are retained for interpretable coefficient analyses; projected/hash coordinates are not assigned individual concept names.

An unordered sum of concept embeddings is the flat control. It is inadequate as the sole implementation of binding because role-swapped events can have the same sum.

**Support for genuinely new combinations.** A one-hot column for a complete event that never occurs in training has no independently estimable coefficient. Hashing that unseen column does not solve this generalization problem. Retain the exact-conjunction route as an interpretable feature condition and implement a factorized condition for the combination-holdout experiments.

For the factorized condition, obtain a fixed concept vector `v(c)` from a declared lexical feature source or training-text-only co-occurrence/SVD representation. A proposed train-only default uses 128 dimensions, bounded by training vocabulary rank; save its vocabulary, co-occurrence window, and fitted decomposition. For each event, concatenate the predicate vector, role-indexed argument vectors with missing-role masks, and projected ordered predicate–argument tensor products. Use independent fixed projection seeds per role and retain an equivalent flat lexical/content control built from the same vectors. Emit these descriptors at the event's registered availability time, rather than collapsing all story events into one unordered bag.

This supplies shared features to new combinations of familiar components while preserving role identity. It introduces a declared lexical/statistical representation prior, whose contribution is evaluated through the matched controls. Complete event identity and novel lexical concepts remain separate generalization conditions. Report unavailable or all-zero feature support explicitly; an unsupported one-hot column cannot establish that the corresponding semantic distinction is absent from neural activity.

### 10.4 Registered model comparisons

| Encoding condition | Features | Question answered |
|---|---|---|
| E0 | `N + L` | How much prediction is attributable to presentation and lexical variables? |
| E1 | `N + L + C` | What does flat concept content add? |
| E2 | `N + L + C + B` | What does within-event structure add? |
| E3 | `N + L + C + R + D` | What do reference and discourse features add without explicit event-binding features? |
| E4 | `N + L + C + B + R + D` | What is captured by the complete semantic description? |
| E5 | `N + L + Q` | How does a strong flat QA-style interpretable representation perform? |
| E6 | `N + L + H[m,l,c]` | What is predicted by each frozen LLM state? |
| E7 | `N + L + C + B + R + D + H[m,l,c]` | Which semantic information is complementary to an LLM representation? |
| E8 | E4 with registered structure-preserving-content corruptions | Does the relation assignment matter beyond the constituents? |
| E9 | E7 with the same structural corruptions | Does structure-sensitive information remain after including LLM states? |

Split E4 by deleting `B`, `R`, or `D` one group at a time for interpretable group-specific contrasts. Fit the reduced model again rather than simply zeroing coefficients in the full model.

### 10.5 Estimator

Use subject-specific ridge/banded ridge regression. A convenient objective is

\[
\min_{\{W_g\}}\left\|Y-\sum_g X_gW_g\right\|_F^2
+\sum_g\lambda_g\|W_g\|_F^2.
\]

Banded ridge permits separate regularization strengths for feature groups; Himalaya provides an established implementation route. [R22]

Proposed regularization search: 21 log-spaced ridge values from \(10^{-4}\) to \(10^6\), with group-wise mixtures selected through a fixed-budget multiple-kernel/banded-ridge procedure. Feature scaling, PCA, group normalization, and regularization selection occur within the relevant training/inner-validation split.

Preserve the same target voxels and temporal rows across feature comparisons. Use voxel batches for memory control and store out-of-fold predictions, rather than only summary correlations.

### 10.6 Scores

Report held-out Pearson correlation and coefficient of determination separately:

\[
R^2=1-\frac{\sum_t(Y_t-\widehat Y_t)^2}{\sum_t(Y_t-\overline Y_{\mathrm{test}})^2}.
\]

The test mean appears here as part of the evaluation statistic, not as a fitted predictor. Also retain prediction error relative to a training-derived constant baseline when cross-run calibration is relevant.

Do not identify \(R^2\), squared correlation, and ceiling-normalized correlation with one another. Negative \(R^2\) values are retained. Estimate reliability/noise ceilings only when the data support the estimator and state the assumptions. Avoid clipping values above a noisy estimated ceiling.

A contrast such as

\[
\Delta R^2_{B}=R^2(N,L,C,B,R,D)-R^2(N,L,C,R,D)
\]

quantifies incremental out-of-sample prediction under the specified model family. It is not a causal variance component or proof that a particular brain region implements a symbolic binding operator.

### 10.7 Encoding–decoding separation

The E branch uses features extracted from text and frozen model states. It does not consume a D branch’s predicted semantic answers as if they were independently supplied stimulus features.

Encoding-implied concept maps can be compared with D-derived groundings later. Their shared stimulus labels and fitted neural data are recorded, and any claim of independent validation uses appropriate separated fits or held-out observations.

---

<a id="s11"></a>
## 11. Parallel measurement D: semantic decoding

### 11.1 Objective

Recover stimulus-grounded semantic answers from measured neural responses and from frozen LLM representations. Use common semantic query families and evaluation partitions. The brain and model decoders are fitted separately.

The targets are textual semantic labels. A correct answer demonstrates recoverable information under the measurement procedure. It does not establish that the participant consciously answered the question or spontaneously represented every annotation.

### 11.2 Input representations

For brain decoding, retain parcel-local multivariate temporal features. Proposed shape before learned adaptation is `[batch, parcel, delayed_sample, component]`, with explicit masks for unavailable components/volumes.

For model decoding, retain a fixed stimulus-window state or ordered temporal bins of frozen activations. Proposed shape is `[batch, temporal_bin, hidden_dimension]`. A global, dimension-matched adapter supports basic readout comparisons.

Brain parcels and model tokens/units are different coordinate systems. Do not label model token positions as cortical analogues. Cross-system comparisons focus on task distinctions, within-system structure gains, and concept distance matrices.

### 11.3 Decoder families

| Decoder | Specification | Purpose |
|---|---|---|
| D0: query/candidate prior | Query AST, surface form where used, and legal candidates; no brain/model features. | Detect question and candidate-set shortcuts. |
| D1: low-rank conditional linear readout | Input-dependent score with query/candidate-specific weights. | Strong capacity-controlled reference with no explicit compositional executor. |
| D2: compact noncompositional neural readout | Matched-budget MLP or small attention readout over inputs and query features. | Test whether gains arise from structure rather than general nonlinear capacity. |
| D3: modular neuro-symbolic readout | Typed concept modules and a differentiable executor. | Test whether explicit structure supports semantic decoding and systematic recombination. |
| D4: text-informed modular variant | D3 with frozen textual descriptions of concept labels. | Quantify the contribution of semantic label priors; keep separate from the label-ID-only variant. |

The D1 baseline must contain query–input interaction. A linear classifier over a simple concatenation of a query vector and fMRI vector cannot make the neural feature slope depend on the query. A suitable low-rank form is

\[
s(Y,q,a)=(A Y)^\top(B z(q,a))+b(q,a),
\]

where \(z(q,a)\) describes the query and candidate answer. This remains a low-capacity baseline while allowing different questions to read different neural information.

### 11.4 Modular grounding architecture

**Brain-local encoder.** Apply a parcel-local adapter and small temporal encoder to obtain \(e_p\in\mathbb R^{128}\). No cross-parcel mixing occurs before parcel-local grounding scores are computed. This restriction makes the coordinate of a grounding score interpretable as access to that parcel’s input. A globally mixed encoder can be an explicit variant, but its output token position cannot be interpreted as a purely local source.

**Primitive grounding.** For a concept \(c\), compute a primitive evidence profile

\[
g_c(p)=w_c^\top e_p+b_c.
\]

For relation-sensitive evidence, use a low-rank ordered-pair function such as

\[
u_r(p,q)=(A_r e_p)^\top(B_r e_q)+a_r^\top e_p+b_r^\top e_q+\beta_r.
\]

Factor parameters by relation family or concept embedding to avoid a dense parameter tensor for every rare lexical predicate. Preserve ordering so agent/patient reversal is representable.

**Argument routing.** The query’s requested arguments select soft evidence weights over support sites. A role-specific pooled representation can be

\[
v_{c,k}=\sum_p\operatorname{softmax}_p(g_{c,k}(p))e_p,
\]

where \(k\) indexes an argument role. A bounded-capacity event module combines the ordered role representations and predicate identity. A discourse module combines two event-level representations and a typed directed link.

**Execution.** Compile the query AST into modular operations. Use explicitly defined probability/logit semantics for conjunction, existential pooling, and candidate choice. Normalize pooling for candidate/site count so a larger parcel count does not automatically increase an existential score.

The precise executor is part of the method. The proposed default maps atomic scores through a sigmoid, uses `min` for conjunction, `max` for disjunction and existential aggregation over valid candidates, and a softmax over scores for candidate selection. These piecewise-differentiable operations have defined subgradients and do not increase an existential score merely because identical sites are duplicated. Use masked candidates only. An empty candidate domain returns the operator-specific declared value and is flagged in the evaluation record.

A prespecified smooth alternative replaces `min`/`max` with temperature-controlled soft aggregations, with temperature selected on inner validation. Report this as a different executor condition. Neither operator choice supplies evidence that the brain implements those logical operations.

For open-world textual support, compute separate supported and contradicted evidence, then use a three-way calibrated answer head including `undetermined`. Do not obtain textual contradiction by complementing low support. Formal negation is executed only within a query scope whose annotation semantics defines the negation; negated event descriptions are also represented explicitly in the graph.

Use a shared two-layer event MLP with width 128 over the predicate code and role-ordered pooled vectors, with explicit missing-role masks. Use a separate width-128 two-layer discourse MLP over two ordered event vectors and a relation-type code. Event descriptions supplied through the query may identify the events being compared; the full ground-truth event graph remains unavailable to the decoder. Preserve ordered input bins in the local temporal encoder so repeated occurrences can be distinguished to the extent supported by the observations. Alternative event/temporal architectures require named sensitivity conditions.

Logical consistency created by an operator is reported as an architectural property; performance on new supported compositions tests whether its inputs contain useful evidence.

### 11.5 Model-side structural decoder

Use the same query grammar, concept vocabulary policy, modular functions, training examples, and readout-budget protocol. Model input adapters operate on the cached temporal bins or state vectors rather than requiring an anatomical parcellation.

For a model-side grounding-profile variant, partition native hidden coordinates into a fixed number of groups or use explicitly declared support slots. Grouping is arbitrary and must be repeated with registered alternative partitions. Learned-slot geometry is labeled as readout-mediated.

Absolute D scores across brains and LLMs do not equate acquisition noise, pretraining history, or native capacity. Report within-system gains from structure and matched semantic error patterns alongside absolute scores. These comparisons answer which information is accessible under the readout, rather than ranking biological and artificial intelligence.

### 11.6 Training objective and defaults

Use cross-entropy for categorical answers, binary cross-entropy for valid binary tasks, and explicit masking/acceptable-answer sets for ambiguity. Balance losses at the source-window and query-family levels.

Proposed defaults are hidden dimension 128, three random seeds, AdamW, batch size 32 source windows, and a maximum of 100 epochs. Tune learning rate over `3e-4` and `1e-3`, weight decay over `1e-5` and `1e-4`, and dropout over `0.0` and `0.1`, using a declared fixed-budget inner-validation protocol. Early stopping uses validation loss with patience 10. These are proposed implementation settings, not claimed optimal values.

Cap concept-module and readout parameter budgets across the structured and noncompositional families; report exact parameter counts, optimizer steps, label exposure, and effective input dimensions. Data adapters may differ in shape, so “matched capacity” requires documentation rather than an assertion based on equal hidden size alone.

D3’s core training uses final-answer supervision. Any additional direct supervision of primitive concept presence is an explicitly labeled auxiliary-loss experiment. No variant receives anatomical concept labels because such labels are absent.

### 11.7 Measurements

Report macro accuracy, balanced accuracy where applicable, per-family macro F1, log loss, calibration, and confusion matrices. For candidate-choice tasks, record candidate-set size and chance/prior baselines. Retrieval metrics are separate from semantic QA.

Compositional generalization is evaluated using the splits in Section 14. Report role-binding accuracy and exact composite-answer correctness separately from constituent success. Avoid claiming that correctly answering two constituent questions establishes their correct binding.

### 11.8 Grounding outputs that must be saved

For each held-out example, save primitive concept evidence, ordered relation evidence where feasible, argument-routing weights, intermediate event/discourse states, final logits, predicted answers, and the exact source/query identifiers.

Do not save only the final attention or guidance vector and label it the concept’s neural representation. Preserve the decomposition required to separate input-derived evidence from architecture-imposed combinations.

### 11.9 Independent operation of E and D

D can produce informative outcomes when E shows limited unique structural gain, and E can produce informative outcomes when D struggles with fine-grained queries. Both branches are run and reported. Neither branch is demoted to a post hoc validation tier based on the other’s performance.

---

<a id="s12"></a>
## 12. Grounding signatures and representational geometry

### 12.1 Preserve several estimands

The concept–concept comparison proposed in the discussion is retained. After estimating concept-associated signatures, compute relationships within the brain-derived representation and within each model representation, then compare the resulting relationship structures.

Several signatures answer different questions. Keep them available rather than deciding in advance that one must become the project’s principal empirical object.

| Signature family | Construction | What similarity means | Important limitation |
|---|---|---|---|
| G1: grounding profile | Held-out concept-conditioned primitive evidence across brain parcels or declared model support sites. | Similar distribution of readout evidence under the specified decoder. | Architecture, biases, and support-site definitions can shape the result. |
| G2: activity-pattern signature | Concept-conditioned patterns estimated from measured neural responses or native frozen model activations. | Similar observed/native activity associated with the concept or configuration. | Naturalistic contexts and noise can confound the estimate. |
| G3: learned latent signature | Internal concept/event states of a fitted readout or embedding model. | Similarity within the trained representational coordinate system. | Geometry may be non-identifiable from task performance and may change under reparameterization. |
| G4: encoding-implied signature | Predicted response pattern associated with a registered semantic feature/configuration under a fitted E model. | Similar associations in the encoding model’s predicted neural space. | It remains model-implied and depends on feature correlations and the reference context. |

Also retain stimulus–stimulus RDMs. They connect the concept-level analysis to measured responses for actual instances and provide a complementary diagnostic without replacing the concept-level question.

### 12.2 G1: concept-conditioned grounding profiles

For concept \(c\), participant \(s\), and held-out occurrence set \(I_c\), define

\[
u^{G1}_{s,c}(p)=\frac{1}{|I_c|}\sum_{i\in I_c}g_{s,c}(Y_i)_p.
\]

A context-balanced implementation first averages within story/context strata, then averages across strata. Report the weighting policy and occurrence counts.

Save both the raw profile and a contrast profile relative to matched concept-absent or neural-mismatched windows. A mismatch contrast can help remove a static concept/site bias, but it is not a complete remedy for correlated contextual information.

Generate groundings from out-of-fold examples. Do not estimate a profile only from the training examples used to learn its module. For signatures associated with a role or discourse condition, define the conditioning set explicitly, such as `return:agent`, `lend:recipient`, or `reference:reintroduced_entity`.

Comparing two parcel profiles answers whether their information is similarly distributed under the decoder. It does not establish that two concepts occupy adjacent cortical locations or use identical fine-grained neural patterns.

### 12.3 G2: concept-conditioned measured/native patterns

Estimate concept-associated patterns from measured neural data and native model states using the same semantic instance definitions. Two complementary estimators are proposed:

**Matched occurrence contrasts.** Compare windows containing the target concept/configuration against windows matched on lexical content, topic, duration, and relevant context. Matching rules are developed on training data and applied without looking at neural test effects.

**Conditional linear contrasts.** Estimate responses associated with the target label while accounting for registered nuisance/content variables. Use separate estimation partitions and expose uncertainty and collinearity. These are conditional associations; they are not pure isolated concept responses.

Do not residualize using semantic variables whose contribution is the target of the analysis. A control that removes the entire relevant event meaning can make the intended effect unidentifiable. Report the estimand before and after each control set.

For LLM patterns, compare the same stimulus instances, role contexts, and extraction endpoints. Use lexical controls and role-preserving/role-changing contrasts to assess how much a signature is driven by the visible word itself.

### 12.4 G3: non-identifiability of a freely learned latent

If a linear readout produces \(Wh\), an invertible transformation \(A\) permits

\[
h'=Ah,\qquad W'=WA^{-1},\qquad W'h'=Wh.
\]

Predictions are unchanged while many cosine distances between latent vectors can change. Therefore, predictive accuracy alone does not uniquely identify a latent metric.

A learned latent is still a legitimate object of study when its construction is declared. Analyze its RDM within each independently fitted model, test stability across seeds/architectures, and compare it with G1/G2/G4. Do not average raw latent coordinates from separately fitted networks unless an explicit training-only alignment establishes correspondence. RDM aggregation across common concept labels avoids requiring coordinate identity.

Any contrastive brain–text embedding extension is labeled `alignment_trained`. Its geometric agreement cannot serve as independent evidence that brain and model geometries agree, because agreement was optimized. NEURONA already includes such a retrieval extension. [R02, Appendix C.2]

### 12.5 G4: encoding-implied concept and configuration maps

For interpretable sparse features, examine the fitted concept/configuration coefficients or a defined predicted-response contrast. For a nonlinear or interacting feature representation, specify a background distribution and compute contrasts averaged over supported contexts.

Do not generate an impossible graph with every concept absent and interpret a single-feature insertion as an observed neural response. Counterfactual feature contrasts remain model-derived, and their validity depends on the reference distribution.

G4 can be compared with D-grounding profiles and G2 patterns using independent partitions where feasible. Agreement offers converging evidence conditional on shared labels and modeling assumptions. G4 never replaces the measured neural side of G2.

### 12.6 Distance matrices

For any declared signature family \(G\), a cosine dissimilarity matrix is

\[
D_{B}^{G}(c_i,c_j)=1-\frac{u_{c_i}^{G}\cdot u_{c_j}^{G}}
{\|u_{c_i}^{G}\|\|u_{c_j}^{G}\|}.
\]

Construct the corresponding matrix \(D_{M,m,l}^{G}\) within each model/layer. Brain and model vector dimensions can differ because their RDM rows/columns refer to the same concepts or configurations.

Register centering, scaling, norm thresholds, and missing-value policies. Raw cosine, centered correlation distance, and noise-aware distance estimates are different metrics. Report sensitivity rather than selecting whichever produces the strongest agreement.

Use noise-normalized/cross-validated distances only when the data support the required independent estimates. The RSA toolbox documentation gives relevant estimators and assumptions. [R24]

For independent partitions \(a,b\), an illustrative cross-validated distance is

\[
d_{ij}^{\mathrm{cv}}=(u_i^{a}-u_j^{a})^\top\widehat\Sigma^{-1}
(u_i^{b}-u_j^{b}).
\]

A negative estimate is possible and should not be clipped. The noise covariance is estimated independently with shrinkage. Repetitions of the same stimulus can support a clearer noise-unbiased interpretation than different natural contexts containing the same concept. The latter require an explicit assumption that the sampled contexts estimate the same conditional representational object; report this as cross-context generalization where appropriate.

### 12.7 Four semantic distinctions to test

Separate semantic similarity, co-occurrence/association, role relationship, and discourse relationship. A giving–receiving association differs from a giving–lending similarity. An event causing another event is a directed relationship.

A symmetric RDM cannot directly encode causal direction. For directional questions, retain ordered event/configuration signatures or fit directed relational operators in a separately identified analysis. For example, test whether a training-fitted relation operator generalizes from event/argument states to held-out targets. Relational operators have prior work in LLMs [R26]; their novelty here would come from the matched neural comparison and semantic scope.

### 12.8 Brain–model geometry comparisons

For common eligible concepts/configurations, compare RDM upper triangles using rank correlation and appropriately dependent inference. Report separate results for each signature family, semantic level, participant, model, and depth.

Construct competing explanation matrices for lexical overlap, concept frequency, topic/co-occurrence, role match, and typed semantic relationships. Fit multivariable RDM models with concept-aware held-out validation where supported. Do not treat the many matrix entries as independent observations.

Test whether role/discourse relationships explain agreement after content controls, and whether the same relationships recur in separate stories or corpora. A model may agree with one aspect of geometry and disagree with another; preserve the distinction.

### 12.9 Visualization policy

Distance heatmaps, neighborhood summaries, multidimensional scaling, and cortical projections are descriptive views of the registered matrices/maps. The statistical evidence comes from the original high-dimensional estimates and held-out tests.

A two-dimensional embedding must report its distortion or stress where available. Apparent cluster separation after dimensionality reduction is not an independent validation result. Do not select visualized concepts based on attractive clustering without labeling the selection.

---

<a id="s13"></a>
## 13. Compositionality, consistency, and neural-dependence experiments

### 13.1 Systematic recombination

Construct evaluation conditions in which constituent concepts and relevant relation types occur in training, while their tested combinations are held out. Distinguish new surface wording, new query templates, new combinations, and unseen concept types.

A role-swap task tests whether the system preserves argument assignments. A reference task tests whether distinct expressions are connected to the correct entity. An event-chain task tests whether relationships remain coherent when multiple links must be used. These are different tests and should receive separate scores.

### 13.2 Content-preserving structural controls

For feature and query analyses, create controls that preserve concept inventories while changing the relation assignments:

| Control | What is preserved | What is changed |
|---|---|---|
| Role permutation | Predicate and argument concepts. | Ordered role assignments. |
| Reference rewiring | Mention strings, candidate inventory, and approximate distance distribution. | Mention-to-entity links. |
| Event-link shuffle | Event nodes and relation-type counts. | Which ordered event pairs are linked. |
| Relation-label shuffle | Graph nodes, edge endpoints, and edge count. | Link semantics. |
| Flat concept representation | Concept content and overall window. | All explicit binding and cross-event structure. |
| Scope/factuality alteration | Most lexical material. | Negation, attribution, or modality where labels are unambiguous. |

Controls must preserve plausible nuisance distributions and avoid introducing obvious artifacts. Some alterations will produce unsupported or contradictory text descriptions; their role is to test computational sensitivity, not to generate an unmeasured human experimental condition.

### 13.3 What computational edits can demonstrate

When the brain response is recorded for the original text, compare whether original structured features predict that response better than corrupted features. This establishes the utility of the original representation for the recorded target under the encoding model.

For D, changing a query about the same observed stimulus creates a valid task contrast when its answer is known from the source. The neural input remains the same and the answer changes according to the query’s truth conditions.

For an edited sentence never shown to participants, model behavior and predicted fMRI can be studied, but the human neural response to that edited sentence is unknown. Do not enter predicted responses into the measured-brain evaluation table.

### 13.4 Neural dependence

Run query-only and candidate-prior baselines for every task. Then compare intact neural pairing with disrupted pairing that respects temporal dependence and relevant nuisance structure.

Use two distinct disruption analyses:

**Evaluation-time disruption.** Apply the trained decoder to mismatched/circularly shifted neural windows. This tests whether its predictions depend on the correct recording.

**Refit-under-null disruption.** Refit the pipeline on appropriately disrupted stimulus–neural pairing and inspect its performance and geometry. This tests how much semantically plausible structure the architecture and labels can generate without the true neural correspondence.

The second analysis is essential for the concern that an apparently meaningful map could be produced from priors alone. Use multiple block-preserving disruptions with seeds registered in advance. Where full refitting is expensive, predefine a bounded null suite that covers every decoder family and semantic level; retain the full real-data panel.

### 13.5 Consistency and specificity

Measure consistency across independent contexts, stories, participants, and fits. Measure specificity through distinctions between different concepts and sensitivity to role/reference changes.

A decoder that always assigns a concept to one fixed parcel can have perfect within-concept consistency. NEURONA’s consistency metric explicitly rewards concentration of repeated region selections. [R02, pp. 8–9] Therefore, report consistency together with concept discrimination, neural dependence, site-occupancy controls, and context-sensitive changes.

Null models should preserve concept prevalence, number of selected sites, and overall site occupancy where the comparison requires it. Uniform random region assignment alone is often a weak reference for an architecture with strongly nonuniform support-site preferences.

### 13.6 Architecture-induced structure

For each fitted readout, calculate how much of the tested geometry is present in:

1. Primitive evidence before guidance/composition.
2. Guidance weights and deterministic compositions.
3. A prior-only or disrupted-neural version of the same model.
4. Independently measured/native activity estimates.

This decomposition helps identify whether an appealing event-level relationship was learned from neural information or inherited from an additive/concatenative construction. It does not require excluding structured models; it clarifies their contribution.

### 13.7 Regional support tests

Compare full-input decoding, region-restricted decoding, train-without-region decoding, and carefully interpreted evaluation-time masking. Use registered region groups and shared tuning budgets.

Report both unique and redundant information where the design permits it. If removing one region has little effect, the information may remain available elsewhere. If zero-masking causes a large effect, distribution shift may contribute. Neither result alone establishes biological necessity or absence of representation.

### 13.8 Annotation-prior tests

Repeat selected analyses with independently produced annotations, a human-adjudicated subset, and label-ID-only versus text-description concept representations. Examine whether brain–model agreement is concentrated in the annotator’s model family.

Preserve the same scientific questions across annotation versions. A changed ontology that answers a different question should be treated as a different analysis rather than a simple robustness rerun.

---

<a id="s14"></a>
## 14. Splits, leakage prevention, and cross-fitting

### 14.1 Split unit hierarchy

The split registry contains corpus, participant, story, run, passage, source-window, event instance, query AST, and repetition identifiers. The appropriate split unit depends on the claim.

All repetitions, questions, and paraphrases associated with the same source stimulus must stay together in a stimulus-generalization split. Overlapping neural windows and overlapping semantic contexts cannot cross folds unless the analysis is explicitly designed around their dependence.

### 14.2 Multi-story reading protocol

Preserve an official held-out story when the release defines one. Use the remaining stories for nested story-level cross-validation and model selection. Proposed development evaluation is leave-one-story-out, with inner validation drawn from the remaining training stories.

After freezing analysis choices, refit on the permitted training stories and evaluate the sealed story. Separately report the broader out-of-fold story results, since one sealed story provides limited diversity. The sealed story is not repeatedly inspected to redesign the method.

If the downloaded release’s official split differs from its published derivative description, document the discrepancy and choose a reproducible split policy before fitting.

### 14.3 Single-chapter reading protocol

Use contiguous blocks with explicit temporal/context purge intervals. Proposed block size is defined in advance from the chapter’s length and the maximum feature/observation support, rather than chosen to maximize performance. At least several non-overlapping eligible blocks are needed for meaningful within-chapter evaluation.

If a held-out block appears earlier than later training text, the later examples’ contextual prefixes may contain that held-out material. For strict inductive evaluation, purge any training examples whose entire declared input support intersects a held-out block. Do not reset context in a way that changes only one system without recording the change.

Call this within-chapter held-out-block generalization. It does not demonstrate generalization to new stories.

### 14.4 Sentence/passage protocol

For Pereira, split by passage, with all sentence repetitions of a passage assigned together. This prevents neighboring sentences from supplying held-out passage content to the training readout.

Evaluate instance-level geometry using actual independent repetitions only when those repetitions are available. If only averaged responses are acquired, use an appropriate non-repetition estimator and state the limitation.

### 14.5 Compositional holdouts

Create separate registries for:

- Held-out ordered role combinations with familiar constituent types/predicates.
- Held-out predicate–argument type combinations.
- Held-out reference configurations and distance/type combinations.
- Held-out event-link motifs with familiar relation types.
- New concepts, if supported, as a separately named zero-shot/type-generalization task.

A strict combination holdout excludes training source examples containing the held-out configuration, including labels and semantic features in E and D. It also checks contextual support: an excluded event appearing in a training example’s prefix can violate a strict novelty claim.

The foundation LLM may have encountered analogous structures during pretraining. The generalization claim concerns the fitted brain/model readouts and the experimental stimulus partition, rather than guaranteed novelty to the pretrained model.

Holdout assignments can use stimulus annotations and coverage counts under a fixed algorithm. They cannot use neural effect sizes or model success to select especially favorable test motifs.

### 14.6 Purging physical and semantic support

Every example has two support records:

**Stimulus support:** all words, context, and graph updates used to construct its input features or labels.

**Neural support:** all acquisitions, temporal filters, and lagged samples used for its target/input.

For strict splits, reject cross-fold intersections that violate the claimed independence. For long-context analyses on short corpora, this may reduce the eligible set sharply; report the reduction rather than weakening the split without disclosure.

### 14.7 Cross-fitting requirements

Fit scalers, PCA, vocabulary selection, feature pruning, covariance estimators, matching procedures, and hyperparameters using the relevant training partition. Inner validation must have its own fitted transforms where those transforms can influence selection.

Compute D-derived grounding profiles on out-of-fold data. Compute G1/G3 RDMs within each fitted coordinate system and aggregate common labeled relationships; do not directly concatenate incompatible latent vectors across folds.

For G2, use separate occurrence/recording partitions for signature estimation where possible. For G4 validation against G2, preserve independent measurements for the validation side. Shared labels and residual methodological dependence are recorded in the evidence ledger.

### 14.8 Participant and corpus transfer

Three distinct experiments are possible:

| Experiment | What is fitted? | What transfers? |
|---|---|---|
| Within-participant generalization | Separate readout per participant. | New stimuli within that participant. |
| Cross-participant replication | Separate readouts and independently estimated geometries. | Scientific relationships or labeled RDM structure. |
| Cross-participant decoder transfer | Readout fitted on some participants, with declared alignment. | Predictions on an unseen participant. |

Do not report the first or second as the third. Anatomical registration alone does not establish correspondence of arbitrary learned latent coordinates. Any functional alignment uses training-only shared stimuli and is an additional fitted component.

### 14.9 Leakage audit report

Produce an automated report containing duplicate texts, shared contexts, repeated stimuli, overlap halos, AST duplicates, candidate-order imbalance, train/test concept coverage, transform-fit IDs, and any blocked examples. A paper-ready split diagram should be generated from the same registry.

---

<a id="s15"></a>
## 15. Statistical analysis and reliability

### 15.1 Units of inference

Participants, stories, and independent stimulus contexts are the relevant replication units. Queries, voxels, seeds, and model layers provide measurements within those units. They do not multiply the number of independent humans or stories.

Use paired tests for matched E/D/model contrasts. Use participant- and story-aware resampling or hierarchical models, respecting which factors are crossed and which are nested. With a small participant cohort, emphasize uncertainty and consistency of effects rather than apparently precise query-level p-values.

### 15.2 Encoding statistics

Store per-participant, per-story, per-region scores and paired feature-group differences. For voxelwise inference, use a declared multiple-comparison procedure and preserve spatial/temporal dependence in the relevant null. Aggregate ROI results independently of test significance thresholds.

Block-preserving nulls should account for autocorrelation and the duration of source/neural support. A permutation applied at individual TR resolution can produce an inappropriate reference for a smooth time series.

### 15.3 Decoding statistics

Bootstrap or randomize at the source-window/story level, preserving all queries for an example together. For instance-choice tasks, preserve candidate sets and their legal answer distributions. Report uncertainty for each semantic level and important contrast.

Classifier seed variation is computational uncertainty and is summarized separately. It does not substitute for participant or stimulus replication.

### 15.4 Geometry statistics

RDM entries sharing a concept are dependent. Use simultaneous row/column relabeling for appropriate concept-permutation tests, with frequency/type restrictions when required by the null. Avoid shuffling individual matrix cells.

For regression on RDM explanations, use a node-aware held-out design where possible: hold out concepts/configurations and evaluate relationships among held-out nodes, rather than letting the same concept dominate both training and test edges. Report when sparse coverage makes this design unavailable.

Estimate agreement within each participant and across independent context partitions. A group-average RDM alone can conceal instability. For brain maps, use nulls that preserve relevant spatial/network structure; arbitrary voxel permutation is not a valid generic anatomical null.

### 15.5 Noise ceilings and reliability

Report raw scores before normalization. Use repeated stimuli to estimate measurement reliability where available. Cross-subject predictability can provide a different ceiling-like reference under explicit assumptions; it is not interchangeable with a within-subject repetition ceiling. Schrimpf’s paper uses an extrapolated intersubject procedure, which should be described accurately when reproduced. [R01, p. 9]

An observed geometry difference may reflect unequal reliability. Compare reliability-matched subsets or present noise-aware uncertainty analyses before interpreting a low brain–model correlation as a representational divergence.

### 15.6 Multiple comparisons

Before the final evaluation, register hypothesis families for E feature contrasts, D structure/generalization contrasts, G agreement/structure contrasts, and regional/context interactions. Apply a declared correction within and, where necessary, across families corresponding to the claims.

For model-depth profiles, use a prespecified depth grid and simultaneous intervals or a family-level test rather than choosing the best test layer. A validation-selected layer can be evaluated on a separate held-out set and clearly labeled as selected.

### 15.7 Cross-measurement relationships

Compare E, D, and G through paired model/layer/context profiles. Account for multiple checkpoints within a family and multiple layers within a checkpoint. A scatterplot with every layer treated as an independent model overstates sample size.

Use within-family/base–post-trained contrasts and leave-family-out sensitivity where meaningful. Associations across a dozen models support qualified empirical comparisons; they do not establish a universal scaling law.

Avoid a requirement that E and D correlate positively. Their association, lack of association, or semantic-level dissociation is an empirical outcome.

### 15.8 Default resampling budget

A proposed budget is 2,000 clustered bootstrap replicates for confidence intervals and 1,000 valid permutations for planned score comparisons, with exact/randomization methods used where the small number of clusters makes them preferable. Computationally expensive full-pipeline refits use a separate registered budget and cannot borrow the p-value resolution of cheap score-only permutations.

Record the random seed, exchangeability assumptions, and effective number of distinct permutations. Never report more p-value precision than the procedure supports.

---

<a id="s16"></a>
## 16. Integrated experiment matrix and result interpretation

### 16.1 Registered study modules

| Module | Data/inputs | Manipulation or comparison | Output | Claim supported |
|---|---|---|---|---|
| X1: parallel semantic measurements | All eligible reading corpora, all retained semantic levels. | E0–E7; D0–D4. | Encoding and decoding profiles with uncertainty. | Predictive association and information accessibility. |
| X2: structural specificity | Same recorded stimuli. | Role, reference, and event-link controls. | Paired E/D changes and task confusions. | Sensitivity to relation assignments beyond concept inventories. |
| X3: compositional generalization | Registered combination holdouts. | Novel combinations with familiar constituents. | Held-out E/D performance by novelty type. | Readout generalization to new semantic configurations. |
| X4: multilevel geometry | G1–G4 and stimulus RDMs. | Concept, role, event, discourse, and lexical-control relationships. | Within-system RDMs and brain–model comparisons. | Organization conditional on the defined estimand. |
| X5: reproducibility and neural dependence | Independent contexts/participants and disrupted-neural controls. | Separate fits, site-matched nulls, annotation variants. | Stability, specificity, and null-relative effects. | Data-constrained rather than purely prior-driven organization. |
| X6: model depth/context/post-training | Full fixed checkpoint panel. | Six depths; common contexts; modern extended context; matched pairs. | Within-system and cross-measurement contrasts. | Where checkpoint representations agree or diverge. |
| X7: anatomical/context profiles | Registered masks and parcel views. | Region-restricted/retrained readouts and discourse-span conditions. | Region-specific E/D/G profiles. | Distribution of available/predictive information under the models. |
| X8: modality/population extension | Conditional listening resource or compatible additional cohort. | Matched semantic analyses with modality controls. | Explicitly qualified replication/extension. | Generality beyond the initial reading program. |

X1–X7 constitute the planned reading study. X8 is conditional on a documented coverage/access or scientific rationale, rather than a mandatory expansion into multimodal modeling.

### 16.2 Model-panel execution policy

At the shared 32/128-word contexts, evaluate every registered checkpoint at all six depths on the applicable E/D/G tasks. The ten modern checkpoints additionally receive the registered 512-word context. This yields 204 nominal representation conditions before data-specific eligibility checks:

\[
12\times6\times2+10\times6=204.
\]

This count describes the model representation grid, not independent models or subjects. An encoding feature family that does not use an LLM is fitted once per neural split, not redundantly once per model. A brain-only decoder is likewise fitted once per participant/split/decoder configuration. Model-only decoding is not duplicated across participants who saw identical text.

Atlas, null, and auxiliary-representation sensitivity analyses use a prespecified coverage design. The plan does not require a wasteful Cartesian product of every hyperparameter, atlas, random seed, and annotation variant. Every scientific contrast must nevertheless have its registered coverage and comparable tuning budget.

### 16.3 Joint interpretation without forced convergence

The final report presents a structured result profile. It may identify strong E with weak D for a semantic level, strong D with little incremental E, or stable within-system geometry with limited cross-system agreement.

Convergent positive evidence strengthens an interpretation when the branches constrain different alternatives. Shared labels or architecture-induced relationships are recorded so agreement is not counted repeatedly as though it were fully independent.

A null or divergent result should be interpreted against annotation quality, semantic coverage, noise, and readout capacity. These diagnostics help distinguish an unsupported strong claim from a failed implementation or an underidentified comparison. They should not be used to keep changing analyses until a preferred conclusion appears.

### 16.4 Illustrative claims at different evidential strengths

**Supported by D plus neural-dependence controls:** “Role assignments are recoverable from the recorded responses with the tested readout.”

**Supported by E feature contrasts:** “The annotated binding features add held-out predictive information beyond the registered content controls.”

**Supported by independent-context G analyses:** “The estimated relational organization is reproducible across contexts under the stated representation definition.”

**Supported by reliable within-system geometry and matched brain–model contrasts:** “This model family captures some semantic relationships while diverging on others.”

**Not established by this observational program alone:** a unique anatomical location for a concept, biological necessity of a parcel, literal execution of the symbolic program by the brain, or the neural response to an unpresented counterfactual stimulus.

---

<a id="s17"></a>
## 17. Software architecture and repository specification

### 17.1 Architectural objective

Implement a reproducible analysis system with shared data contracts and independently executable scientific branches. The repository should support the entire registered study. A notebook-only collection of fitted models would make leakage auditing, resumption, and collaborator handoff unnecessarily difficult.

The proposed package name is `sembrain`. This name and all interfaces below are specifications to implement. No existing installation or command availability is claimed.

```text
sembrain/
  README.md
  PROPOSAL.md                         # This document, versioned with the project.
  pyproject.toml
  environment.lock                   # Exact environment after compatibility validation.
  configs/
    study.yaml
    datasets.yaml
    models.yaml
    annotation.yaml
    preprocessing.yaml
    encoding.yaml
    decoding.yaml
    geometry.yaml
    inference.yaml
    contrasts.yaml
  schemas/
    corpus.schema.json
    semantic_graph.schema.json
    query.schema.json
    artifact.schema.json
  src/sembrain/
    cli.py
    registry/                        # Model, dataset, ontology, and contrast registries.
    provenance/                      # Hashing, manifests, environment and lineage records.
    datasets/                        # Corpus adapters; no scientific estimator logic.
    annotations/                     # Prefix extraction, graph validation, audit ingestion.
    queries/                         # Typed AST, deterministic compiler, negative generation.
    splits/                          # Story/passage/block/composition splits and purging.
    neural/                          # Confounds, masks, local features, temporal alignment.
    representations/                 # Frozen checkpoint adapters, hooks, pooling, caches.
    features/                        # Lexical and structured stimulus feature families.
    encoding/                        # Independent E fitting, prediction, group comparisons.
    decoding/                        # Independent D fitting, execution, and grounding export.
    signatures/                      # G1–G4 estimators and context-balanced aggregation.
    geometry/                        # Distances, RDM comparisons, directed relations.
    controls/                        # Label, structure, neural, and architecture nulls.
    inference/                       # Clustered uncertainty, randomization, multiplicity.
    reporting/                       # Tables, figures, claim/evidence and coverage reports.
    workflow/                        # Task DAG, scheduler adapters, resumable execution.
  tests/
    unit/
    integration/
    leakage/
    numerical/
    provenance/
  workflows/
    local/
    slurm/
  docs/
    data_dictionary.md
    ontology.md
    annotation_manual.md
    analysis_registry.md
    decisions.md
    limitations.md
  data/                              # Local paths; excluded from Git by default.
    manifests/
    source/                          # Read-only downloaded originals.
    derivatives/
    annotations/
    queries/
    splits/
  artifacts/
    features/
    fits/
    predictions/
    signatures/
    distances/
    statistics/
    reports/
  logs/
```

### 17.2 Separation of concerns

Dataset adapters expose recordings, presentation events, text, and provenance in a common format. They do not decide which semantic hypotheses to favor. Annotation code produces labels and stimulus features; it cannot access neural predictions. Split code defines permissible training and evaluation observations before downstream feature fitting.

E and D consume the same versioned split registry and compatible stimulus identifiers. Each branch owns its fitted parameters, validation decisions, predictions, and error records. G2 can be estimated from eligible measured/native features without waiting for D. G1 depends on fitted decoding modules; G4 depends on fitted encoding models. This dependency structure preserves the different meanings of the signatures.

### 17.3 Suggested implementation stack

Use a Python-based stack with array computation, sparse matrices, tabular storage, neuroimaging I/O, frozen-model inference, and statistical modeling. Suitable components include NumPy/SciPy, pandas or an equivalent table library, PyArrow/Parquet, HDF5 or Zarr, PyTorch, Transformers, NiBabel, Nilearn, scikit-learn, Himalaya, and RSA-related utilities. Banded ridge and dissimilarity tools have existing public implementations. [R22, R24]

The exact compatible versions must be established from actual installation and numerical tests, then locked. This proposal does not assert that an untested combination of current package releases is compatible. Avoid depending on undocumented internal library outputs for layer definitions.

Choose one workflow implementation after infrastructure inspection. A lightweight explicit DAG or an established workflow engine is acceptable. Cluster submission should be a thin execution layer; scientific identity must not depend on a scheduler job number.

### 17.4 Artifact identity and invalidation

Every artifact receives a content-derived identity incorporating its inputs and relevant configuration. At minimum, record:

```text
artifact_id
artifact_type
schema_version
source_dataset_version
source_checksums
annotation_version
ontology_version
query_version
split_id
preprocessing_id
representation_id
estimator_config_hash
fit_seed
code_commit
environment_lock_hash
parent_artifact_ids
created_at
execution_status
```

A different annotation version invalidates dependent queries, structured features, relevant fitted readouts, and their analyses. A different checkpoint revision invalidates its representation cache and downstream model analyses. A changed plot style does not invalidate neural fits.

Cache lookup must compare all scientifically relevant fields. Matching a filename or checkpoint nickname is insufficient. A failed or incomplete artifact must never satisfy a dependency merely because an output file exists.

### 17.5 Reproducibility and logging

Save per-job resource use, wall time, numerical precision, device identity, random seeds, package versions, warnings, and exceptions. Retain the full model/configuration manifest for failed runs as well as successful runs. Atomic writes and completion markers prevent partially written arrays from being treated as finished predictions.

Where exact determinism is unavailable, document the source and quantify seed variability in the registered fits. Resuming an interrupted job must either restore optimizer/sampler state or explicitly create a new fit identity.

The final report should be regenerated from saved predictions and manifests. Values must not be manually transferred into manuscript tables without a provenance link.

---

<a id="s18"></a>
## 18. Data contracts and configuration defaults

### 18.1 Stable join keys

Use explicit string identifiers for corpus, subject, session, run, story, passage, word, source window, query, repetition, and split. IDs are metadata and join keys. They are excluded from model tensors unless a specific field is registered as an experimental input.

Maintain one authoritative text normalization with a reversible mapping to the original transcript. Tokenizer-specific offsets belong in a separate table. Never identify observations solely by row order across independently generated files.

### 18.2 Required table contracts

| Table | Required fields | Integrity requirement |
|---|---|---|
| `corpora` | Corpus ID, release, source URL, checksums, modality, license status, access status. | Actual release inspection determines availability. |
| `runs` | Subject/session/run IDs, story, TR, scans, initial discarded scans, presentation origin, source derivative. | Scan and event clocks must share a validated origin. |
| `words` | Stable word ID, original text, normalized text, character offsets, onset, duration, story/run. | Ordered, reversible, non-overlapping character spans. |
| `annotations` | Typed graph records, support spans, availability, scope, confidence, audit, ontology version. | References resolve within an available prefix. |
| `windows` | Source span, endpoint, context policy, neural support interval, eligibility, grouping IDs. | Physical and semantic support are explicit. |
| `queries` | Query ID, window ID, AST, candidates, answer, family, negative type, support/audit metadata. | The model-facing view excludes answers and evidence. |
| `splits` | Observation/group ID, outer role, inner fold, purge reason, held-out composition. | All shared-support constraints are checked. |
| `fits` | Branch, system, feature families, model/layer/context, subject where applicable, tuning results, seed. | Training/validation identities are immutable. |
| `predictions` | Observation ID, fit ID, prediction, target reference, measurement family, aggregation weight. | Every scored prediction identifies its untouched target. |
| `signatures` | Estimand G1/G2/G3/G4, concept/configuration, system, coordinates, partition, coverage. | Coordinate systems and independence partitions are explicit. |
| `statistics` | Contrast, estimate, uncertainty, null definition, resampling unit, correction family. | Inferential population is stated. |

Separate physically stored label tables from the D-input data-transfer object. Merely asking downstream code to ignore an `answer` field is a weak safeguard.

### 18.3 Array shapes and masks

Use shape names in code and assert them at module boundaries:

```text
Measured continuous fMRI:       Y[scan, neural_feature]
Parcel-local fMRI features:     B[scan, parcel, local_component]
Decoder neural observation:    B_window[batch, lag_or_bin, parcel, local_component]
Word-level LLM features:        H[word, hidden_dimension]
Encoding features:             X[scan, feature]
Structured E feature families:  {family_name: X_family[scan, feature]}
Window-level model inputs:     H_window[batch, ordered_bin, hidden_dimension]
Unary groundings:              G_unary[batch, candidate_site, concept]
Ordered pair groundings:       G_pair[batch, site_from, site_to, predicate]
Concept signature:             U[concept_or_configuration, coordinate]
Symmetric distance matrix:     D[concept_or_configuration, concept_or_configuration]
```

Sparse role/event features need not be densified globally. Store valid-bin, missing-parcel, candidate, and target-eligibility masks separately. A zero vector is not a universally valid missing-data representation. Shape padding must never be interpreted as observed activity or as an additional candidate entity.

### 18.4 Example query record

This is an invented schema illustration, not an observation from a neural dataset:

```json
{
  "query_id": "illustration:q002",
  "window_id": "illustration:w001",
  "family": "binding_verification",
  "ast": {
    "operator": "VerifyEvent",
    "predicate": "lend",
    "roles": {
      "agent": "Leah",
      "recipient": "Noah",
      "theme": "book"
    },
    "scope": "asserted_event"
  },
  "candidate_descriptors": [],
  "label_record": {
    "answer": true,
    "evidence_text": "Leah lent Noah a book.",
    "audit_status": "illustrative_only"
  },
  "provenance": {
    "source_kind": "constructed_example",
    "neural_observation_available": false,
    "ontology_version": "proposal-v1"
  }
}
```

The compiler emits a model-facing object containing the query AST, legal candidates, and a neural/model feature key. `label_record` and `provenance` remain in evaluation/audit storage. The illustrative record is automatically excluded from neural fitting because no observed response exists.

### 18.5 Proposed study configuration

The following YAML is a configuration contract. Null values intentionally mark unresolved release/revision dependencies. A production-run validator must reject them for any selected input.

```yaml
study:
  name: structured_meaning_brain_llm
  proposal_version: "1.0"
  execution_ready: false
  measurement_branches: [encoding, decoding, geometry, structural_validation]
  branch_priority: equal
  foundation_models_frozen: true
  shared_encoding_decoding_objective: false
  brain_model_geometry_alignment_loss: false
  analysis_registry_locked: false

data:
  preferred_modality: reading
  selected_corpora:
    - deniz_reading
    - wehbe_reading
    - pereira_sentences
  conditional_corpora: [lebel_listening, narratives_listening]
  require_source_checksums: true
  require_cross_corpus_overlap_audit: true
  require_release_license_review: true
  root: "${SEMBRAIN_DATA_ROOT}"

annotation:
  ontology_version: proposal-v1
  context_policy: prefix_with_prefix_derived_ledger
  production_checkpoint: Qwen/Qwen2.5-32B-Instruct
  production_revision: null
  audit_checkpoint: meta-llama/Llama-3.3-70B-Instruct
  audit_revision: null
  generation_temperature: 0.0
  allow_uncertain_labels: true
  test_audit_blinded_to_predictions: true
  proposed_double_audit_contexts: 600

representations:
  registry: configs/models.yaml
  checkpoint_revision_required: true
  raw_stimulus_only: true
  query_before_cache: false
  depths: [embedding, block_20pct, block_40pct, block_60pct, block_80pct, block_100pct]
  shared_context_words: [32, 128]
  modern_context_words: [512]
  strict_context_equivalence: true
  word_pooling: last_subtoken
  overlength_action: mark_ineligible_for_matched_comparison

neural:
  additional_spatial_smoothing: false
  parcel_view: schaefer_200_if_valid_mapping
  parcel_components: 8
  learned_transforms_fit_on_training_only: true
  encoding_lag_trs: [1, 2, 3, 4]
  decoder_observation_policy: explicit_offline_support
  future_stimulus_contamination_audit: true

encoding:
  estimators: [ridge, banded_ridge]
  alpha_log10_min: -4
  alpha_log10_max: 6
  alpha_count: 21
  compare_groups: [nuisance, lexical, concepts, bindings, reference, discourse, qa_features, llm]
  scores: [pearson_r, heldout_r2, paired_incremental_r2]
  retain_negative_r2: true

decoding:
  families: [query_prior, conditional_linear, compact_neural, modular, description_informed]
  latent_width: 128
  seeds: [11, 29, 47]
  optimizer: adamw
  learning_rates: [0.0003, 0.001]
  weight_decays: [0.00001, 0.0001]
  dropout_values: [0.0, 0.1]
  batch_source_windows: 32
  maximum_epochs: 100
  validation_patience: 10
  loss_weight_unit: source_window
  expose_full_graph_to_model: false

geometry:
  estimands: [G1_grounding, G2_activity, G3_learned_latent, G4_encoding_implied]
  distances: [cosine_dissimilarity, correlation_distance]
  crossnobis: only_with_valid_independent_pattern_estimates
  crossfit: true
  average_unaligned_latents_across_fits: false
  report_directed_relations_separately: true
  concept_context_minimum_proposed: 20
  concept_story_minimum_proposed: 3

inference:
  bootstrap_replicates: 2000
  score_permutations: 1000
  expensive_refit_null_budget: null
  cluster_units: [participant, story]
  correction_registry: configs/contrasts.yaml
  require_coverage_report: true

execution:
  artifact_root: "${SEMBRAIN_ARTIFACT_ROOT}"
  failure_policy: record_and_stop_affected_dependencies
  silently_substitute_inputs: false
  log_negative_and_null_results: true
```

The expensive-refit null budget must be registered after infrastructure profiling and before final outcome inspection. It is intentionally separate from the number of score-only permutations. Corpus-specific rules override inapplicable global settings explicitly: sentence-level beta data do not receive continuous-time HRF convolution, and a single chapter uses blocked rather than story-level evaluation.

### 18.6 Model registry contract

Every M01–M12 entry needs checkpoint ID, resolved commit revision, tokenizer revision, access/license status, architecture type, context limits, raw-block hooks, normalization details, precision, and successful load-test status. Model cards establish candidate existence; local validation establishes execution readiness.

```yaml
M06:
  checkpoint: Qwen/Qwen3-8B-Base
  revision: null
  tokenizer_revision: null
  role: evaluated_frozen_model
  architecture: causal_decoder
  context_conditions_words: [32, 128, 512]
  hook_specification: null
  license_reviewed: false
  locally_load_verified: false
  precision: bfloat16
  generated_reasoning_allowed_in_representation_task: false
```

Apply the same schema to all entries. Avoid hard-coding architecture-specific layer paths until inspected against the pinned implementation.

---

<a id="s19"></a>
## 19. Algorithms and proposed command-line interface

### 19.1 Common preparation algorithm

```text
1. Inspect each selected dataset release and create immutable source manifests.
2. Normalize source text with reversible offsets and align presentation/scanning clocks.
3. Generate prefix-limited semantic annotations and complete the blinded quality audit.
4. Construct query families, semantic feature specifications, and coverage reports.
5. Register stimulus groups, outer evaluation partitions, inner folds, and composition holdouts.
6. Purge observations with forbidden shared textual, temporal, or neural support.
7. Lock ontology, prompts, split definitions, checkpoint panel, contrasts, and analysis budgets.
8. Materialize source-only frozen representations and timing transforms.
9. For each fold, fit all learned preprocessing on that fold's allowed training observations.
10. Publish the prepared artifacts to independent E, D, G2, and control tasks.
```

Semantic annotation and stimulus-only checkpoint inference can be computed for held-out stimuli without fitting to their neural targets. Learned vocabulary transformations, feature selection, normalization, and any semantic parser adaptation must still follow their registered training/held-out boundaries. The role of each preprocessing operation must be explicit.

### 19.2 Encoding algorithm

```text
For each corpus and participant:
  For each outer split:
    Load allowed training, validation, and test neural observations.
    Build timing-aligned nuisance, lexical, structured, QA, and LLM feature groups.
    Fit fold-specific feature transforms using permitted training observations.
    For each registered E condition:
      Select penalties and any allowed feature settings through inner grouped validation.
      Refit on the allowed outer training pool using the selected settings.
      Predict each outer-test neural observation once.
      Save predictions, coefficients, transforms, validation history, and provenance.
    Score paired conditions using identical eligible observations.
    Export G4 signatures with the encoding-implied label and estimation partition.
```

A validation-selected setting may use the outer training pool for final fitting. The sealed outer-test set remains outside selection. Interpret group-removal contrasts using refitted reduced models, rather than zeroing a block in an already fitted joint model and calling that unique explained variance.

### 19.3 Decoding algorithm

```text
For each permitted system:
  Brain system = a participant's recorded responses.
  Model system = a checkpoint/layer/context cache.
  For each outer split and D family:
    Materialize feature-only inputs and legal query/candidate objects.
    Fit input transforms on training observations only.
    Train with source-window-balanced loss and grouped inner validation.
    Keep the foundation representation frozen.
    Evaluate final answers on the untouched outer-test queries.
    Save all answer probabilities, candidate masks, errors, and training metadata.
    For modular fits, export primitive, guidance, and executed grounding tensors separately.
    Evaluate registered input-disruption and candidate-permutation controls.
```

A brain-only decoder is independent of the number of LLMs in the panel. Model-side fits are independent of how many people read identical stimuli. This saves substantial work without reducing scientific coverage.

### 19.4 Geometry algorithm

```text
For each registered signature estimand:
  Identify the legal independent contexts/partitions and coordinate system.
  Apply the prespecified concept/configuration eligibility criteria.
  Estimate signatures separately across contexts, participants, and fits.
  Compute within-system distances in each valid coordinate system.
  Retain coverage, covariance/reliability information, and missing-entry masks.
  Compare RDMs using shared concept/configuration identities and matched coverage.
  Apply lexical, topic, co-occurrence, frequency, and architecture controls.
  Estimate uncertainty at context/concept/participant/story levels as appropriate.
  Summarize reproducibility and brain–model agreement without treating matrix cells as IID.
```

For G3, compute distances within each fitted latent and aggregate distance-level summaries. Do not concatenate raw latent coordinates across independently trained decoders. For G4, mark the signatures as predictions implied by an encoding model, including when no isolated concept stimulus was recorded.

### 19.5 Proposed CLI

These commands describe the software contract to be implemented. They are not claims about an already available tool:

```bash
python -m sembrain audit sources --config configs/study.yaml
python -m sembrain audit clocks --config configs/study.yaml
python -m sembrain annotate generate --config configs/annotation.yaml
python -m sembrain annotate validate --config configs/annotation.yaml
python -m sembrain annotate import-human-audit --input data/annotations/adjudications.parquet
python -m sembrain queries compile --config configs/study.yaml
python -m sembrain splits build --config configs/study.yaml
python -m sembrain audit leakage --config configs/study.yaml
python -m sembrain registry lock --config configs/study.yaml
python -m sembrain features build --config configs/study.yaml
python -m sembrain workflow plan --config configs/study.yaml --output artifacts/task_manifest.json
python -m sembrain workflow run --manifest artifacts/task_manifest.json --branch encoding
python -m sembrain workflow run --manifest artifacts/task_manifest.json --branch decoding
python -m sembrain workflow run --manifest artifacts/task_manifest.json --branch geometry
python -m sembrain workflow run --manifest artifacts/task_manifest.json --branch structural_validation
python -m sembrain inference run --config configs/inference.yaml
python -m sembrain report build --config configs/study.yaml
python -m sembrain audit release --config configs/study.yaml
```

The workflow planner reports unresolved prerequisites, memory estimates, nominal and eligible condition counts, dependency reuse, and total planned fits. It can run while execution is blocked. A training command must refuse unresolved dataset provenance, split-integrity errors, or unpinned selected checkpoints.

### 19.6 Result tables required for handoff

Produce machine-readable tables for E scores, D scores, grounding reliability, geometry comparisons, structural controls, participant-level results, model-family contrasts, annotation quality, and exclusions. Each manuscript result must map to a contrast ID and saved artifact IDs.

Do not report only a pooled “brain alignment” number. A collaborator should be able to trace a finding to a semantic level, context condition, neural view, estimator, data split, and interpretation category.

---

<a id="s20"></a>
## 20. Dependency-based implementation roadmap

### 20.1 Execution logic

The roadmap specifies a complete scientific program and its prerequisites. Each work package has an auditable output and a completion test. Completion tests concern correctness, coverage, and reproducibility; they do not require a favorable scientific result.

```text
WP0: registry and research contracts
  ├── WP1: source acquisition, access, clocks, anatomical metadata
  ├── WP2: ontology, prefix annotation, human audit
  └── WP3: checkpoint adapters and extraction semantics

WP1 + WP2 → WP4: queries, coverage, grouped/compositional splits
WP1 + WP3 + WP4 → WP5: fold-safe neural/model/structured features

WP5 ──→ WP6E: encoding fits and predictions
     ├→ WP6D: decoding fits, answers, and groundings
     ├→ WP7A: measured/native activity geometry
     └→ WP8: matched controls and null execution

WP6E → WP7B: encoding-implied geometry
WP6D → WP7C: grounding and learned-latent geometry

WP6E + WP6D + WP7A/B/C + WP8 → WP9: inference, replication, joint interpretation
WP9 → WP10: paper, reproducible release, evidence audit
```

WP6E and WP6D are peer branches. Their execution order may depend on resource availability; their scientific status does not.

### 20.2 Work packages

| Package | Inputs and responsibility | Deliverable | Completion criterion |
|---|---|---|---|
| WP0: contracts | Proposal, source registry, researcher decisions. | Locked schema, initial analysis registry, decision log, task ownership. | Every planned claim has an estimand and an evaluation route. |
| WP1: data | Official releases and release documentation. | Dataset manifests, source hashes, timing audit, anatomical/behavioral metadata inventory. | Selected corpora can be read and joined without unresolved clock or provenance errors. |
| WP2: semantics | Original stimuli and approved ontology. | Prefix graphs, prompts, independent annotations, human adjudication, error/coverage report. | Model-facing labels have traceable support and known uncertainty. |
| WP3: models | Fixed M01–M12 panel, compatible environment. | Pinned checkpoint registry, validated extraction adapters, resource profiles. | Context, token, layer, and normalization contracts pass their tests. |
| WP4: evaluation | WP1/2, query compiler, planned semantic contrasts. | Canonical queries, candidate controls, split manifests, overlap/purge report. | Every scored item obeys group and support separation; all exclusions are enumerated. |
| WP5: features | WP1/3/4, feature specifications. | Immutable caches and fold-fitted transforms. | A feature tensor is reproducible from its manifest and contains only permitted information. |
| WP6E: encoding | WP5, E condition registry. | Held-out neural predictions and feature contrasts. | All eligible registered E conditions are completed or explicitly marked failed/unavailable. |
| WP6D: decoding | WP5, D condition registry. | Answer probabilities, primitive and composed groundings, systematicity scores. | All eligible registered D conditions satisfy input isolation and comparable evaluation. |
| WP7: geometry | Eligible native/measured states and relevant E/D fits. | G1–G4 signatures, distances, directed-relation analyses, coverage and stability estimates. | Every geometry has a stated coordinate system, estimation partition, and interpretation. |
| WP8: controls | Shared prepared artifacts and registered nulls. | Structural, neural-dependence, annotation, and architecture-control results. | Nulls preserve the dependencies they are designed to control and are distinguishable by purpose. |
| WP9: inference | Predictions/signatures from independent corpora, contexts, and participants. | Corrected contrasts, uncertainty, replication comparisons, claim/evidence ledger. | Inferential units and dependencies are respected; contradictory results are retained. |
| WP10: dissemination | Complete evidence ledger and source permissions. | Manuscript, machine-readable results, code/configuration release, data/annotation cards. | Every headline claim traces to saved evidence and a stated limitation. |

### 20.3 Decisions requiring researcher approval

The research lead signs off on ontology changes, semantic eligibility criteria, dataset substitutions, changes in claims, grouping or contamination assumptions, additions/removals of scientific contrasts, and substantial reductions to the fixed panel. These decisions cannot be delegated to an agent’s preference for easier positive results.

Engineering choices such as chunk sizes, scheduler queue selection, lossless compression, and numerically equivalent batching can be made by a coding agent within the contract. A change in precision, context approximation, anatomical interpolation, or feature normalization can affect the measurement and must be logged and validated accordingly.

### 20.4 Resource profiling without outcome-driven redesign

Profile I/O, memory, frozen inference throughput, and readout training time on a declared training-only resource benchmark. This is an engineering measurement used to schedule the full design. It does not select models, semantic levels, or datasets according to whether they produce the desired scientific effect.

A verified failure of an input contract should stop the affected branch while other independent work proceeds. A statistically weak result is a scientific result and does not automatically trigger a new ontology, a narrower dataset, or an unregistered favorable subgroup.

### 20.5 Final evaluation freeze

Before examining sealed evaluation outcomes, archive the source and annotation manifests, all selected checkpoints and hooks, split logic, eligibility rules, estimators and tuning spaces, null definitions, inference code, and intended contrasts. Development-set analyses should be clearly separated from confirmatory evaluation. Subsequent corrections must describe their cause, affected outputs, and whether the final test had already been inspected.

---

<a id="s21"></a>
## 21. Compute, storage, staffing, and feasibility

### 21.1 Feasibility assessment

The project can be implemented using existing neural recordings, frozen models, conventional encoding estimators, and comparatively small task-specific readouts. It does not depend on training a foundation model or collecting a new fMRI cohort.

The most consequential work lies in semantic annotation, verified data access/alignment, split construction, and interpretability controls. A powerful coding agent can automate much of the infrastructure. Scientific supervision remains necessary for annotation semantics, inferential assumptions, and the relationship between evidence and claims.

The established encoding experience assumed for the research team is directly useful. The additional engineering emphasis is a typed semantic resource, compositional query execution, independently defined signatures, and dependable multimodule provenance.

### 21.2 Hardware planning envelope

The following values are planning estimates from tensor sizes and proposed workloads, rather than measured requirements for this project.

A configuration with access to one or more 48–80 GB GPUs, 64–128 GB of host memory for substantial preprocessing/encoding jobs, and approximately 1–2 TB of working storage would be a useful starting resource envelope. Actual requirements depend on accessible derivatives, neural feature counts, concurrent jobs, and cached model weights. Multi-GPU access helps with the larger proposed annotators and with parallel execution.

An 8-billion-parameter model requires approximately 16 GB merely to store two-byte weights. Activations, temporary buffers, key/value caches, and framework overhead require additional memory. The proposed 32B and 70B annotation checkpoints require correspondingly larger resources; two-byte weights alone are approximately 64–65 GB and 140 GB. Do not treat those weight-only calculations as guarantees that a model fits on a specific device arrangement.

Quantization may be appropriate for annotation generation if its quality is audited and recorded. For the frozen representational comparison, precision changes define a measurement condition and should be controlled. Avoid comparing a quantized checkpoint against full-precision checkpoints without documenting the asymmetry.

### 21.3 Feature-cache estimate

For an illustrative corpus total of 50,000 word endpoints, an average hidden dimension of 4,096, the 204 registered representation conditions, and two-byte storage:

\[
50{,}000\times4{,}096\times204\times2
=83{,}558{,}400{,}000\text{ bytes},
\]

approximately 83.6 GB in decimal units, before compression and metadata. These are assumed dimensions and corpus size. They are not a measured count of words or features in the selected releases.

Use actual model dimensions and eligible context coverage to generate the final estimate. Streaming word-to-TR aggregation and storing registered event/window summaries can reduce intermediate storage; preserve enough information to reproduce the registered G and D analyses. Store one copy of each stimulus-only model representation regardless of participant count.

### 21.4 Fit-count accounting

Encoding cost scales with the number of participants, outer splits, model representation conditions, and required feature/contrast fits. Brain-side D cost scales with participants, splits, decoder configurations, and seeds. Model-side D cost scales with model representation conditions, splits, decoder configurations, and seeds, without a participant multiplier for identical stimuli.

Write these counts into the task manifest before launch. Reuse factorization, feature caches, and nuisance fits when mathematically valid. Do not reuse an outer-fold transform fitted on forbidden data merely to save time.

A nominal number of GPU hours cannot be responsibly specified before actual corpus sizes, extraction policies, and hardware are profiled. Strict prefix extraction, especially for bidirectional BERT or sliding windows with exact context resets, can be substantially more expensive than one full-story forward pass. The resource planner must implement the declared semantics rather than silently replacing them with a cheaper computation.

### 21.5 Human annotation and research effort

The proposed 600-context double audit requires two independent judgments per context. At an illustrative ten minutes per judgment, this corresponds to roughly 200 person-hours, before adjudication and ontology development. Actual duration will vary with discourse complexity and the scope of a “context.” This calculation makes the labor visible; it is not an assertion about the necessary final audit size.

Useful responsibility areas are neural-data/encoding, semantic annotation/decoding, model extraction/software infrastructure, and statistical/interpretive review. A small team can combine roles, provided independent annotation checks and researcher review are preserved.

### 21.6 Feasibility risks and workable responses

Insufficient cross-story concept coverage may restrict particular concept-level analyses while leaving event-instance and stimulus-level analyses valid. A missing dataset release may require a documented source substitution. Limited GPU concurrency changes scheduling. None of these conditions automatically removes the parallel E/D design.

A lightweight decoder that generalizes poorly can still be compared with the registered more expressive readouts to diagnose accessibility limits. Report those differences transparently. Do not describe every failed readout as proof that the corresponding meaning is absent from the brain or LLM.

---

<a id="s22"></a>
## 22. Paper construction, venue fit, and release package

### 22.1 Intended scientific narrative

The manuscript should introduce the need to distinguish semantic content, binding, and discourse organization when comparing language models with human neural responses. Explain the common semantic resource and the parallel measurements, then present the relationships actually found among E, D, G, and structural generalization.

The paper’s final emphasis can reflect the strongest well-supported finding after the complete analysis. The study design remains broad enough to support encoding–decoding dissociations, context-dependent concept organization, model-family differences, or reproducible relational structure. An attractive concept map alone is insufficient evidence for the full claim.

### 22.2 Planned figures and tables

| Output | Purpose | Safeguard |
|---|---|---|
| Framework figure | Show source annotation and separately fitted E/D branches, with G1–G4 dependencies. | Measured fMRI, predicted fMRI, annotation labels, and learned maps have different visual labels. |
| Data/coverage table | Describe participants, stories, independent contexts, annotation accuracy, and semantic coverage. | Question counts are not substituted for neural sample counts. |
| Parallel measurement figure | Show E and D by semantic level/model/context with comparable coverage. | No unregistered composite score or primary/secondary ordering. |
| Structural generalization figure | Show role/reference/discourse-specific changes under holdout and control conditions. | Changes to computational inputs are distinguished from actual changed stimuli. |
| Geometry figure | Compare within-system relationships and cross-system agreement. | Native, grounding, learned-latent, and encoding-implied geometries are separately labeled. |
| Anatomical/support figure | Describe where the tested readouts obtain information or predict activity. | Grounding and coefficient maps are not labeled as causal neural localization. |
| Reliability/null table | Summarize stability, neural dependence, architecture baselines, and replication. | Nulls retain their distinct interpretations and sampling units. |

The appendix should include the full ontology, exact schemas, audited examples, model/revision/hook registry, complete condition coverage, participant-level results, hyperparameter spaces, preprocessing and exclusion reports, and remaining failures.

### 22.3 COLING 2027 fit and administrative constraints

The official call includes computational cognitive modeling, model interpretation, discourse, semantics, and language resources, all relevant to this proposal. It lists October 12, 2026 as the latest ARR submission deadline and December 23, 2026 for commitment after meta-reviews. Long papers allow up to eight main-body pages; short papers allow four, with a mandatory limitations section and additional references/appendix space. Recheck the official call before submission. [R27]

This proposal is sized as a full research study. The conference year does not provide an extra year of preparation. As of the document date, the latest eligible submission is close; completion and eligibility must be assessed against actual project status, rather than assuming a calendar because the meeting is in 2027.

The general main-track fit is stronger than a claim that this English-only study directly addresses every emphasis of the special theme. Do not add multilingual scope solely for venue branding without an appropriate paired neural-data design.

If that submission cycle is missed, preserve the scientific program and choose a later suitable NLP or computational-neuroscience venue after checking its current call. This document does not predict acceptance or guarantee novelty relative to every subsequent publication.

### 22.4 Release deliverables

Release code, locked configurations, checkpoint metadata, reproducible download/adaptation scripts, versioned annotations where permitted, canonical queries, split manifests, machine-readable predictions/summary results where consent allows, and data/annotation/model-analysis cards.

A public package should distinguish derived annotations from original copyrighted text. For a licensed narrative, releasing word offsets, source identifiers, permissible short examples, and reconstruction scripts may be appropriate when full-text redistribution is restricted. Verify each source’s terms rather than assuming that public download access grants every redistribution permission.

Neural data use and participant consent require institutional review appropriate to the research setting. This proposal does not make a legal or ethics-exemption determination. Keep participant IDs pseudonymous, avoid identity or sensitive-trait inference, and limit the claims to the recorded language tasks and consented research uses.

### 22.5 What would make a strong submission

A strong submission would provide a clearly specified, reusable measurement program and at least one reproducible empirical finding about structured meaning that conventional aggregate comparisons obscure. It should demonstrate what the fMRI contributes, what the semantic scaffold supplies, and how model representations agree or diverge across semantic levels.

Every proposed branch need not produce a positive effect. Every reported conclusion does need an appropriate target, valid uncertainty, and a transparent account of alternatives. Avoid a result narrative that counts mutually dependent analyses as several independent confirmations of the same claim.

---

<a id="s23"></a>
## 23. Risks, unresolved decisions, and completion criteria

### 23.1 Risk register

| Risk | Why it matters | Required response | Effect on claims |
|---|---|---|---|
| Deniz release access or metadata incomplete. | Preferred reading design depends on matched recordings, text, clocks, and mappings. | Verify the source manifest; document a substitute or restricted analysis before outcome selection. | No claim about an uninspected release or absent condition. |
| Sparse compositional combinations. | Natural narratives may not supply clean factorial contrasts. | Publish coverage, use valid familiar-constituent holdouts, preserve unidentifiable cells as such. | Limit generalization to observed semantic coverage. |
| Annotation errors or ontology bias. | Semantic targets and graphs can embed systematic mistakes. | Independent annotator/human audit, evidence spans, alternate normalization and label-prior controls. | Framework findings remain conditional on audited distinctions. |
| Query-only shortcuts. | Correct answers may reflect wording/candidate priors. | Query-only baselines, balanced candidates, neural disruptions, hard supported contrasts. | Demonstrate the incremental role of the representation. |
| Future text leakage. | Full-story parsing or model states can expose unavailable information. | Prefix availability, strict cache tests, timed-support audit. | Separate retrospective analyses from prefix-constrained analyses. |
| BOLD temporal mixing. | A delayed recording can contain responses to nearby or later material. | Explicit offline support, neighboring/suffix controls, lag sensitivity. | Avoid word-exact or online-decoder claims. |
| Readout-induced geometry. | Training can create distances and shared patterns. | Multiple estimands, primitive/composed separation, null fits, native/measured corroboration. | State the estimator-dependent nature of each geometry. |
| Low neural reliability. | Weak agreement may be measurement-limited. | Report reliability and valid ceilings, repeated-stimulus analyses where available. | No strong absence claim from an unreliable estimate. |
| Pretraining exposure to stories. | Model knowledge may include the source narrative. | Document corpus familiarity, novel combination tests and local role/reference analyses. | Limit claims about genuinely novel language understanding. |
| Unequal tuning or capacity. | Apparent structural gains may reflect parameter or search differences. | Capacity/tuning tables, matched baselines, within-system contrasts. | Do not interpret raw score gaps as pure structure effects. |
| Anatomical overinterpretation. | Decoder weights, support maps, and measured patterns answer different questions. | Maintain estimand labels and appropriate map transformations. | No unique localization or causal-necessity assertion. |
| Multiple testing and dependent units. | Large condition grids can produce misleading significance. | Registered contrasts, clustered inference, family-level sensitivity, complete coverage reporting. | Match inferential strength to participants/stories/models actually sampled. |

### 23.2 Unresolved items to settle before production fitting

The research team must verify the exact reading-data files and access terms, applicable subject/run coverage, native-to-atlas mappings, available independent language localizers, actual repeated stimuli, and cross-corpus overlap. It must resolve checkpoint revisions and access, validate layer hooks, finalize ontology mappings and the audit protocol, select legal negative-generation rules, and lock the expensive-refit null budget.

These are bounded implementation decisions. None requires seeing favorable test-set outcomes. Record their resolution in Appendix D’s decision template and the machine-readable registry.

The 2025 QA-theory follow-up and the 2026 expanded semantic-relations preprint also require a full-text novelty review before the manuscript’s final contribution statements. The present proposal explicitly separates the verified earlier studies from the incompletely retrieved follow-ups. [R05, R11]

### 23.3 Technical completion criteria

The full selected dataset program has a versioned manifest and validated alignment; all eligible registered model conditions have verified feature caches; E and D can execute independently; all learned preprocessing is fold-safe; G1–G4 have correct provenance and coordinates; required controls have completed or documented failures; statistical outputs identify their valid inferential units; every result is reproducible from artifact lineage.

A missing optional modality extension does not invalidate the reading study. An unresolved label leak, timing error, or invalid evaluation split does invalidate the affected result and must be repaired before it is used.

### 23.4 Scientific completion criteria

The evidence package answers the registered questions to the extent supported by the available data, reports the unsupported or underidentified comparisons, and distinguishes reliable divergence from measurement failure. It includes uncertainty, independent-context/participant checks, and representation-sensitive controls.

Completion does not require a leaderboard win, universal brain–model agreement, identical anatomical maps across people, or a single concept hierarchy. A well-supported dissociation is an acceptable outcome. A visually coherent map that is explained entirely by a label or architecture prior does not substantiate a brain-representation claim.

### 23.5 Change-control rule

A change following test inspection must be classified as error correction, robustness analysis, or a new exploratory analysis. Preserve previous outputs and document whether the change alters the relevant scientific claim. Confirmation of a revised claim requires an appropriate untouched evaluation source or an explicit exploratory qualification.

---

<a id="s24"></a>
## 24. Researcher and coding-agent handoff rules

### 24.1 Researcher responsibilities

The researcher owns the semantic ontology, claim definitions, interpretation of ambiguous language, source selection, evaluation assumptions, and final scientific conclusions. They approve the complete analysis registry and any change that alters what a measurement means. They review a sample of program execution traces and verify that semantic interpretations remain faithful to the source stimuli.

### 24.2 Coding-agent responsibilities

The coding agent implements the specified contracts, obtains permitted public inputs through their documented access routes, constructs reproducible manifests, writes and runs integrity tests, executes the registered conditions, records failures, and produces auditable outputs. It should identify ambiguities with a concrete proposed resolution and the affected files or analyses.

The agent must not silently substitute a dataset, checkpoint, label definition, context policy, query family, or feature geometry. It must not relabel simulated/predicted responses as measured fMRI. It must not remove unfavorable models or conditions from the coverage report.

### 24.3 First handoff actions

Read Sections 1–4 for scientific purpose and evidential boundaries, Sections 5–9 for source/semantic/representation contracts, and Sections 17–20 for implementation dependencies. Create the repository and registry, inspect actual sources, and produce a machine-readable readiness report listing verified inputs and blockers.

Then implement the common data contracts, typed query compiler, split/purge logic, and checkpoint adapters. Parallelize the E and D implementations once their shared feature contracts are established. The complete planned model panel and contrasts should appear in the workflow manifest before final evaluation.

### 24.4 Definition of a satisfactory handoff state

Another researcher should be able to identify the origin of every dataset and checkpoint, understand each annotation and target, execute the registered experiment graph, trace a result to its inputs, and see which claims remain uncertain. Another coding agent should be able to resume from a task manifest without reconstructing scientific choices from informal chat history.

This document is the initial authoritative reference. Subsequent scientifically approved decisions update its version or a linked decision log. Informal instructions that conflict with a locked evaluation contract must be resolved explicitly before execution.

---

<a id="a01"></a>
## Appendix A. Conceptual explanations retained from the discussion

### A.1 Why do we need scores when concept-localization ground truth is unavailable?

There are observable targets at two levels. E predicts actual recorded fMRI. D predicts semantic answers supported by the stimulus. Anatomical labels for intermediate concepts are unavailable. Therefore, direct prediction scores and indirect grounding analyses can coexist without inconsistency.

A sentence can establish that Leah lent a book to Noah. It cannot establish which parcel must encode lending. Successful held-out decoding constrains the usefulness of a learned grounding, while its anatomical interpretation remains model-dependent. This is the distinction made by NEURONA’s supervised final answers and unsupervised intermediate assignments. [R02, pp. 6–9]

### A.2 Does this have to become an accuracy competition?

No. The project uses scores as measurements supporting scientific comparisons. Its contribution can be a reproducible distinction among semantic levels or a dissociation between measurements. A structured representation may improve relational D while adding little aggregate E, or may provide a transparent account of information already contained in LLM features.

All eligible E and D results are reported in parallel. Their relative values and relationships are findings, rather than criteria chosen in advance to designate one branch as the most important.

### A.3 What is the useful shortcut provided by weak supervision?

Existing stimulus–fMRI pairs can supervise semantic answers without a separate experiment localizing every concept. This enables a broad and practical research program.

The trade-off is interpretive uncertainty about intermediate representations. A fixed concept-name embedding could be semantically plausible and perfectly consistent while ignoring neural data. Neural disruptions, independent contexts, stronger query controls, and alternative signature estimators help distinguish that case from a genuinely data-constrained organization.

### A.4 What exactly does concept–concept cosine similarity mean?

It means similarity between the specified vectors under the specified coordinate system. For grounding profiles, it concerns shared distributions of decoding evidence. For measured activity patterns, it concerns shared response patterns. For freely learned latent vectors, it depends on the selected representation and readout constraints. For encoding-implied maps, it concerns predictions associated with a semantic feature or configuration.

Cosine proximity does not by itself imply anatomical proximity, causal dependence, conceptual identity, or a direction of influence. Directed semantic links require separate analysis. A high-dimensional distance matrix also does not imply that a two-dimensional plot faithfully displays all its relationships.

### A.5 Why retain stimulus geometry as well as concept geometry?

Concept estimates aggregate multiple stimulus instances. Their uncertainty depends on context balance, lexical realization, and the available repeated occurrences. Stimulus-level geometry keeps the comparison closer to specific measured observations; concept-level geometry asks how those observations organize around a semantic description.

Both are legitimate and can be mutually informative. Agreement increases confidence when their estimation procedures constrain different alternatives. Neither needs to replace the other or predetermine the final result’s shape.

### A.6 Why is modern LLM agreement helpful without being a correctness oracle?

A modern model provides an independently inspectable representational system. Comparing its semantic relationships with human neural relationships can reveal shared organization or informative divergence.

Agreement can also arise from common text statistics, annotation conventions, or readout structure. Independently fitted mappings and label/architecture controls help identify those contributions. The project therefore treats brain–LLM agreement as a measured relationship whose interpretation must be earned.

### A.7 How much does listening complicate a language study?

The model can still receive text transcripts. Listening adds acoustic presentation variables and modality-specific interpretation; it does not automatically require a speech-model research project. Reading removes the physical sound signal from the principal input but retains visual presentation, timing, and hemodynamic issues.

The dataset decision should balance modality match, neural reliability, semantic coverage, and source accessibility. The preferred design uses reading resources. A larger listening resource remains an explicit alternative when it materially improves the achievable semantic tests. [R12–R19]

### A.8 What does compositionality mean operationally here?

The study distinguishes constituent occurrence, role binding, reference, and relationships between events. New combinations should be evaluated with familiar constituents where possible, and role-changing edits should affect the relevant outputs while meaning-preserving changes should preserve them.

A symbolic executor necessarily enforces some structural regularities. Generalization of its predictions from actual neural inputs, with appropriate controls, is the empirical test. Literal symbolic execution in biological tissue is a stronger claim that the existing-data design does not settle.

### A.9 Are the measurements independent?

Their estimators and objectives are separately fitted, and one branch’s predicted output is not silently used as ground truth for another. That is procedural independence.

They still share stimuli, annotations, neural recordings, and sometimes preprocessing. Those shared dependencies belong in the statistical and interpretive account. Several agreeing analyses should not be counted as several independent experiments when they recycle the same evidence.

---

<a id="a02"></a>
## Appendix B. Worked example and expected data flow

### B.1 Constructed text

The following passage is an invented illustration:

> Leah lent Noah a book. Noah promised to return it. Because Leah needed the book back, she reminded him. Noah returned it.

It has no associated neural recording and must never enter a neural evaluation as though it did.

### B.2 Semantic records

An illustrative graph contains story-local entities for Leah, Noah, and the book. Event records include a completed lending event, a promise whose content is a future return, Leah’s stated need, the reminder, and a completed return. The promised event and the later completed event retain different factuality and occurrence records, even if a later annotation links them.

Roles for lending are agent=Leah, recipient=Noah, theme=book. Roles for the completed return include agent=Noah and theme=book. The pronouns have reference annotations only where the prefix supports the resolution. The explicit `because` construction supports a motivational connection between the stated need and reminder.

The graph must not infer that the reminder caused the eventual return merely from their sequence. A later returned-book event does not retroactively make a future intention an already completed action.

### B.3 Prefix availability

After the first sentence, the lending event is available. After the second sentence, the promise and intended return are available; the completed return is still unavailable. After the third sentence, the need, reminder, and supported reference/motivational links are available. The last sentence supplies the completed return.

Each state update has a supporting span and availability endpoint. Full-story annotation can describe relationships retrospectively, but an online-availability view must retain the earlier uncertainty.

### B.4 Queries and lawful inputs

| Query | Supported target in the stated final-passage scope | Information the decoder receives |
|---|---|---|
| Is a book mentioned? | Yes. | Representation plus the concept-occurrence query. |
| Did Leah lend the book to Noah? | Supported. | Representation plus the requested ordered binding. |
| Who returned the book: Leah or Noah? | Noah. | Representation, query, and the same candidate list given to D0. |
| Does the final “it” refer to the book? | Supported in this example. | Representation and legal mention/entity descriptors. |
| Was the reminder motivated by the stated need? | Supported by the explicit construction. | Representation and event/relation descriptors. |
| Did the reminder cause the completed return? | Undetermined from the text alone. | The same legal input types; no target-revealing graph. |

The answer, full passage, correct reference graph, and evidence text are excluded from the D-input object. E is allowed to use the stimulus-derived graph because its task predicts measured neural responses from stimulus information.

### B.5 How an actual recorded passage would flow through E

For an actual dataset passage, each graph update becomes a timed feature event. The lexical and flat-concept control contains the relevant words/concepts. Binding features retain the ordered lending roles. Reference features identify the linked mentions under their declared information-availability policy. Discourse features record the explicit need–reminder relationship.

After timing alignment and fold-safe transformation, E conditions predict the corresponding measured fMRI. The E2–E1 contrast asks what within-event structure contributes under that feature representation. E4–E2 and registered group deletion contrasts examine reference/discourse contributions. E7–E6 examines information contributed by the structured features beyond a given frozen LLM state.

All predictions are evaluated against the actual held-out recording. A role-swapped graph can be tested as a computational control for prediction of that original recording. It does not generate a measured response to a different sentence.

### B.6 How an actual recorded passage would flow through D

The brain decoder obtains a declared fMRI observation window, legal query, and candidates. The model decoder obtains pre-query cached states for the matched text window and the same legal query/candidates. Both predict the semantic answer.

A modular fit exposes primitive concept scores, ordered role routing, event evidence, and final answer scores. Correctly identifying both people and a lending event is distinguished from assigning the people to the correct roles.

An input-mismatched control tests how performance changes when the query remains fixed and the representation comes from an incompatible but appropriately matched source. A full-refit disrupted-neural control tests what organization the architecture and supervision can learn without the true stimulus–brain pairing.

### B.7 How geometry is formed across many real passages

Across sufficiently many independent passages, estimate signatures for lending, returning, requesting, and related concepts. Also estimate role-conditioned signatures and event configurations where coverage permits. Balance or model lexical and context distributions under the registered policy.

Compute within-brain and within-model relationships separately for each estimand and partition. Compare their relation structure after estimating reliability and accounting for shared concept content. A finding that concepts cluster similarly while role-bound configurations differ would be informative; so would reproducible agreement at both levels or a lack of reliable geometry at one level.

No single illustrative sentence can establish such an organization. It demonstrates the data flow and the required distinctions.

---

<a id="a03"></a>
## Appendix C. Required engineering and scientific-integrity tests

These tests are implementation requirements. Passing them establishes that a pipeline obeys its specification; it does not establish a positive neuroscientific finding.

| Test ID | Test | Expected outcome |
|---|---|---|
| T01 | Dataset manifest round trip. | Every loaded array/text record maps to an actual source file and checksum. |
| T02 | Word-offset reconstruction. | Original words and character spans can be reconstructed without silent normalization shifts. |
| T03 | Scan/presentation impulse alignment. | A synthetic event appears at the expected positive hemodynamic delay, with no off-by-one or reversed lag. |
| T04 | Beta-versus-continuous dispatch. | Already estimated sentence responses are not reconvolved as continuous observations. |
| T05 | Prefix annotation invariance. | Appending unseen future text cannot change a stored prefix-derived annotation. |
| T06 | Entity-ledger provenance. | Every ledger item was available when the current annotation was generated. |
| T07 | Factuality and negation. | A promise, completed event, negated event, and reported hypothetical remain distinguishable. |
| T08 | Unsupported causal inference. | Temporal adjacency alone never creates a confirmed causal label. |
| T09 | Source/evidence isolation. | D-input serialization contains no answer, full graph, evidence text, or source passage. |
| T10 | Candidate permutation. | Reordering candidates permutes choice probabilities correspondingly, rather than changing semantic predictions. |
| T11 | New local entity identity. | A new story-local ID cannot be exploited as a globally trained class label. |
| T12 | Negative-label validation. | A generated role reversal that is also true in the scope is rejected as a binary negative. |
| T13 | Repetition/group leakage. | All source repetitions and related query renderings stay in the registered group. |
| T14 | Context-support purge. | A training example whose declared context contains held-out material is removed in the strict split. |
| T15 | Neural-support purge. | Overlapping BOLD/filter/lag support does not cross a prohibited split boundary. |
| T16 | Transform isolation. | Changing held-out targets cannot change a trained scaler, PCA, vocabulary, or selected hyperparameter. |
| T17 | BERT future invariance. | Future suffix changes do not affect a prefix-constrained extracted state. |
| T18 | Exact sliding context. | Cached extraction matches an independent forward pass on the declared window to the numerical tolerance. |
| T19 | Layer-hook validation. | Recorded features correspond to the documented block output and normalization state. |
| T20 | Token-overflow handling. | Overlength inputs are reported, and matched analysis coverage is updated without silent truncation. |
| T21 | Model-only representation caching. | Query changes do not modify the cached stimulus representation. |
| T22 | Ordered role sensitivity. | Swapping agent and recipient produces distinct binding features and ordered module inputs. |
| T23 | Flat-content invariance. | The corresponding unordered concept control remains unchanged when only the binding is swapped. |
| T24 | Count-normalized execution. | Duplicating identical support sites does not spuriously raise the default existential score. |
| T25 | Open-world support. | Low support does not automatically become textual contradiction. |
| T26 | Source-window loss weights. | Replicating every query of one source window does not increase that window’s total training weight. |
| T27 | E/D independence. | E runs without D fits, and D runs without E fits; both use compatible source/split contracts. |
| T28 | Native/predicted separation. | A G4 encoding-implied map cannot be loaded as a measured G2 response. |
| T29 | Latent-coordinate test. | Applying an invertible latent transform with compensated readout preserves answers while potentially changing cosine geometry. |
| T30 | Additive-guidance null. | Independent synthetic components exhibit construction-induced correlation, and the analysis attributes it to the operator. |
| T31 | Fixed-map consistency null. | A fixed concept-only assignment is perfectly stable but fails the required neural-dependence criterion. |
| T32 | Parcel-locality test. | Before registered cross-site combination, changing a distant parcel cannot alter a primitive local encoding. |
| T33 | Crossnobis eligibility. | The estimator refuses an invalid absence of independent pattern estimates/covariance inputs. |
| T34 | RDM resampling. | A concept permutation acts on both rows and columns; matrix cells are never independently shuffled for an IID test. |
| T35 | Contrast masking. | Paired condition scores use identical eligible target sets, with coverage differences reported separately. |
| T36 | Condition coverage. | The nominal model grid equals 204, and every excluded or failed cell has an explicit reason. |
| T37 | Seed identity. | Seed changes create distinct fit IDs and do not create additional human observations. |
| T38 | Interrupted-job recovery. | Partial files cannot satisfy dependencies; restart behavior is logged and reproducible. |
| T39 | Result provenance. | Every table entry resolves to prediction/signature artifacts, fit IDs, and a contrast definition. |
| T40 | Source permission audit. | Release packaging excludes source material whose redistribution permission is unresolved. |

Synthetic arrays are appropriate for numerical and leakage tests. Mark them as synthetic in the artifact registry and prevent their use in scientific neural result tables.

---

<a id="a04"></a>
## Appendix D. Decision log and evidence ledger templates

### D.1 Decision record

```yaml
decision_id: DEC-0001
date: null
researcher_approver: null
question: "Which exact Deniz reading release is used?"
options_considered: []
source_evidence: []
selected_option: null
rationale: null
affected_artifacts: []
analysis_outcomes_seen_before_decision: false
requires_protocol_version_change: false
status: unresolved
```

A decision should be understandable without recovering a private conversation. Source-evidence fields identify actual papers, files, metadata, or tests. Unresolved items remain visible in the readiness report.

### D.2 Claim/evidence record

```yaml
claim_id: CLAIM-RQ2-001
claim_text: null
semantic_level: binding
measurement_families: [encoding, decoding]
contrast_ids: []
corpora: []
participants: []
source_partition: null
prediction_artifact_ids: []
uncertainty_artifact_ids: []
required_controls: []
control_outcomes: []
shared_dependencies_between_measurements: []
alternative_explanations: []
reliability_limitations: []
claim_status: pending
```

Do not fill the final claim before examining the evidence. A registered question and contrast can be specified in advance; the direction and strength of the eventual claim remain empirical.

### D.3 Coverage record

Every scientific output includes counts of source stories/passages, participants, independent stimulus instances, repeated measurements, eligible concepts/configurations, source windows, queries, and effective model conditions. Include excluded counts by reason and the final inferential population.

### D.4 Ready-for-handoff checklist

The repository points to the current proposal version; source access and licenses are recorded; exact checkpoints and environment are pinned; ontology and annotation audits are available; all model-facing inputs obey the isolation contract; the complete task manifest exists; E/D/G/K dependencies are explicit; unknowns and failures are enumerated; another team member can reproduce a selected result from saved artifacts.

---

<a id="refs"></a>
## References and verified resource registry

Reference identifiers are portable within this Markdown document. Source locations below are supplied for retrieval and verification. Public webpages and model cards can change; production execution must preserve the actual accessed revision, release, or file checksum. The verification notes distinguish inspected literature or documentation from unverified downloadable payloads.

### Motivating papers and direct code verification

<a id="ref-r01"></a>
**[R01] Schrimpf, M., Blank, I. A., Tuckute, G., Kauf, C., Hosseini, E. A., Kanwisher, N., Tenenbaum, J. B., and Fedorenko, E. (2021).** *The neural architecture of language: Integrative modeling converges on predictive processing.* Proceedings of the National Academy of Sciences, 118(45), e2105646118. DOI: `10.1073/pnas.2105646118`. Source supplied by the user as `schrimpf (2).pdf`; full main paper inspected. Key locations: pp. 2–3, model/data and readout evaluation; pp. 8–9, limitations, dataset descriptions, and methods. Public location: `https://doi.org/10.1073/pnas.2105646118`.

<a id="ref-r02"></a>
**[R02] Wang, Y., Hsu, J., Adeli, E., and Wu, J. (2026).** *Neuro-Symbolic Decoding of Neural Activity.* ICLR 2026; framework name NEURONA. Source supplied by the user as `ICLR-2026-neuro-symbolic-decoding-of-neural-activity-Paper-Conference.pdf`; main paper and relevant appendices inspected. Key locations: p. 4, framework diagram; pp. 6–9, supervision, accuracy and grounding consistency; p. 18, retrieval; p. 20, grounding correlations; pp. 25–26, implementation and guidance definitions. Public locations: `https://arxiv.org/abs/2603.03343`; `https://github.com/PPWangyc/neurona`.

<a id="ref-r03"></a>
**[R03] NEURONA BOLD5000 scene-graph generation script.** `src/create_dataset/create_bold5000_scene_graph.py`, in the authors’ repository. The inspected function explicitly calls `model="gpt-4o"`. Retrieved Git blob SHA: `44125c8d4677f84f67f1be7b47c541ade3d1b8c7`; this identifies the file content and is not a repository commit SHA. Location: `https://github.com/PPWangyc/neurona/blob/main/src/create_dataset/create_bold5000_scene_graph.py`. This verifies the released BOLD5000 annotation script; it does not establish every configuration used for every dataset in the paper.

### Closely related semantic and neural-modeling work

<a id="ref-r04"></a>
**[R04] Benara, V., et al. (2024).** *Crafting Interpretable Embeddings by Asking LLMs Questions.* QA-Emb. Location: `https://arxiv.org/abs/2405.16714`. The interpretable QA-feature approach is a close precedent for the E branch; it must receive a substantive comparison rather than a passing citation.

<a id="ref-r05"></a>
**[R05] Singh, C., Antonello, R., et al. (2025).** *Evaluating scientific theories as predictive models in language neuroscience.* Preprint. Location: `https://www.biorxiv.org/content/10.1101/2025.08.12.669958v1`; authors’ project: `https://microsoft.github.io/automated-brain-explanations/`. The project’s description of a compact QA encoding model and fMRI/ECoG evaluation was inspected. Full manuscript retrieval remained incomplete. Verify detailed experiments, author list, and any later publication version before final comparison.

<a id="ref-r06"></a>
**[R06] Antonello, R., et al.** *Generative causal testing to bridge data-driven models and scientific theories in language neuroscience.* Preprint initially posted in 2024; inspect the relevant version for citation year and final publication status. Location: `https://arxiv.org/abs/2410.00812`; author project: `https://microsoft.github.io/automated-brain-explanations/`. Used here for the distinction between model-generated hypotheses and experiments involving additional recorded neural responses.

<a id="ref-r07"></a>
**[R07] Toneva, M., Mitchell, T. M., and Wehbe, L. (2022).** *Combining computational controls with natural text reveals aspects of meaning composition.* Nature Computational Science, 2, 745–757. DOI: `10.1038/s43588-022-00354-6`. Location: `https://www.nature.com/articles/s43588-022-00354-6`.

<a id="ref-r08"></a>
**[R08] Kauf, C., et al. (2024).** *Lexical semantic content, not syntactic structure, is the main contributor to ANN-brain similarity of fMRI responses in the language network.* Neurobiology of Language. DOI: `10.1162/nol_a_00116`. Bibliographic record: `https://pubmed.ncbi.nlm.nih.gov/37205405/`. The record also reflects an earlier preprint; distinguish the published paper from the preprint when assembling the final bibliography.

<a id="ref-r09"></a>
**[R09] Frankland, S. M., and Greene, J. D. (2015).** *An architecture for encoding sentence meaning in left mid-superior temporal cortex.* Proceedings of the National Academy of Sciences, 112(37), 11732–11737. DOI: `10.1073/pnas.1421236112`. Location: `https://www.pnas.org/doi/10.1073/pnas.1421236112`.

<a id="ref-r10"></a>
**[R10] Chen, C., Gong, L., Deniz, F., Klein, D., and Gallant, J. (2024).** *Representations of Semantic Relations in the Human Brain During Active Relation Processing.* Conference on Cognitive Computational Neuroscience report. Location: `https://2024.ccneuro.org/pdf/448_Paper_authored_ccn_2024.pdf`. Accessible report inspected; usable trial-level data access was not established for this proposal.

<a id="ref-r11"></a>
**[R11] Chen, C., et al. (2026).** *Representations of semantic relations in the human cerebral cortex.* bioRxiv preprint. DOI: `10.64898/2026.02.19.706815`. Location: `https://www.biorxiv.org/content/10.64898/2026.02.19.706815v1`. Existence and title verified; attempts to retrieve the full manuscript failed. This entry is an explicit novelty-review dependency, rather than a claim that the expanded experiments have been comprehensively reviewed.

### Neural datasets and their documented reuse

<a id="ref-r12"></a>
**[R12] Deniz, F., Nunez-Elizalde, A. O., Huth, A. G., and Gallant, J. L. (2019).** *The representation of semantic information across human cerebral cortex during listening versus reading is invariant to stimulus modality.* Journal of Neuroscience, 39, 7722–7736. Authors’ publication page: `https://gallantlab.org/publications/2019-representation-semantic-information-human/`. The proposed use is the reading condition of a documented release, whose payload remains to be verified.

<a id="ref-r13"></a>
**[R13] Oota, S. R., Çelik, E., Deniz, F., and Toneva, M. (2024).** *Speech language models lack important brain-relevant semantics.* ACL 2024. Accessible manuscript: `https://arxiv.org/html/2311.04664v2`. This reuse documents a reading/listening subset and examines sensory-feature contributions to model–brain correspondence.

<a id="ref-r14"></a>
**[R14] Authors’ code and subset documentation accompanying R13.** Location: `https://github.com/subbareddy248/speech-llm-brain`. Inspected README documents six participants, eleven stories, a ten-story/one-story split, and TR 2.0045 seconds. It links `https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri`. The README was verified; the linked neural dataset payload was not successfully inspected. Retrieved README blob SHA: `5c56264f0d6edeb20ab14e3f38a4482f36eebc9a`.

<a id="ref-r15"></a>
**[R15] Chen, C., et al. (2024), and associated authors’ code.** *The cortical representation of language timescales is shared between reading and listening.* Communications Biology. Repository: `https://github.com/denizenslab/timescales_filtering`. Inspected README links the BOLD data at `https://berkeley.app.box.com/v/Deniz-et-al-2019`. The code documentation was inspected; the Box payload was not successfully inspected. Retrieved README blob SHA: `4802cd43a156a1054c9020c065e0317a87401fab`.

<a id="ref-r16"></a>
**[R16] Wehbe, L., et al. (2014).** *Simultaneously uncovering the patterns of brain regions involved in different story reading subprocesses.* PLOS ONE. Authors’ data/method page: `https://www.cs.cmu.edu/~fmri/plosone/`. The source documents narrative reading, the study’s participants and presentation procedure, and alternative data derivatives. Exact downloaded derivatives and transforms must be recorded.

<a id="ref-r17"></a>
**[R17] Pereira, F., et al. (2018).** *Toward a universal decoder of linguistic meaning from brain activation.* Nature Communications, 9, 963. DOI: `10.1038/s41467-018-03068-4`. Location: `https://www.nature.com/articles/s41467-018-03068-4`. Relevant sentence experiments and repetition details are also documented in R01. Verify the exact accessible trial/beta representation before selecting reliability analyses.

<a id="ref-r18"></a>
**[R18] LeBel, A., Wagner, L., Jain, S., et al. (2023).** *A natural language fMRI dataset for voxelwise encoding models.* Scientific Data, 10, 555. DOI: `10.1038/s41597-023-02437-z`. Article: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10447563/`; dataset: `https://openneuro.org/datasets/ds003020`; accompanying code: `https://github.com/HuthLab/deep-fMRI-dataset`. The eight-participant, 27-narrative resource is the documented listening alternative. Do not infer the full contents of an evolving release from the paper alone.

<a id="ref-r19"></a>
**[R19] Nastase, S. A., et al. (2021).** *The “Narratives” fMRI dataset for evaluating models of naturalistic language comprehension.* Scientific Data. Authors’ publication page: `https://hassonlab.princeton.edu/publications/%E2%80%9Cnarratives%E2%80%9D-fmri-dataset-evaluating-models-naturalistic-language-comprehension`. Published totals describe a heterogeneous collection; they do not imply every participant heard all stories.

### Semantic formalisms and analysis methods

<a id="ref-r20"></a>
**[R20] Banarescu, L., et al. (2013).** *Abstract Meaning Representation for Sembanking.* Linguistic Annotation Workshop. Location: `https://aclanthology.org/W13-2322/`. Methodological precedent for the semantic graph scaffold.

<a id="ref-r21"></a>
**[R21] Uniform Meaning Representation resources.** Van Gysel, J. E. L., et al. (2021), *Designing a Uniform Meaning Representation for Natural Language Processing*, KI, location: `https://link.springer.com/article/10.1007/s13218-021-00722-w`; Chun, J., and Xue, N. (2024), *Uniform Meaning Representation Parsing as a Pipelined Approach*, TextGraphs, location: `https://aclanthology.org/2024.textgraphs-1.3/`; project: `https://umr4nlp.github.io/web/`. The proposed annotation schema borrows relevant distinctions and preserves its own neural-alignment provenance fields.

<a id="ref-r22"></a>
**[R22] Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A. O., and Gallant, J. L. (2022).** *Feature-space selection with banded ridge regression.* NeuroImage, 264, 119728. DOI: `10.1016/j.neuroimage.2022.119728`. Article: `https://pmc.ncbi.nlm.nih.gov/articles/PMC9807218/`; implementation documentation: `https://gallantlab.org/himalaya/`; modeling tutorials: `https://gallantlab.org/voxelwise_tutorials/pages/voxelwise_modeling.html`. Exact compatible software versions remain to be validated and locked.

<a id="ref-r23"></a>
**[R23] Haufe, S., et al. (2014).** *On the interpretation of weight vectors of linear models in multivariate neuroimaging.* NeuroImage, 87, 96–110. DOI: `10.1016/j.neuroimage.2013.10.067`. Location: `https://pubmed.ncbi.nlm.nih.gov/24239590/`. Relevant to the distinction between discriminative readout weights and activation-pattern interpretations.

<a id="ref-r24"></a>
**[R24] RSA toolbox documentation, “Estimating dissimilarities.”** Location: `https://rsatoolbox.readthedocs.io/en/stable/distances.html`. A public implementation reference for distance estimation, including noise-aware choices. This proposal adds dataset-specific eligibility, independence, and inference requirements; importing an estimator does not establish those assumptions.

<a id="ref-r25"></a>
**[R25] Hewitt, J., and Liang, P. (2019).** *Designing and Interpreting Probes with Control Tasks.* EMNLP-IJCNLP. Location: `https://aclanthology.org/D19-1275/`; preprint: `https://arxiv.org/abs/1909.03368`. Relevant to control tasks and interpretation of information recovered by trained probes.

<a id="ref-r26"></a>
**[R26] Hernandez, E., et al. (2024).** *Linearity of Relation Decoding in Transformer Language Models.* ICLR 2024. Locations: `https://openreview.net/forum?id=w7LU2s14kE`; `https://arxiv.org/abs/2308.09124`; authors’ project: `https://lre.baulab.info/`. A relational-model-analysis precedent; it does not itself provide the proposed matched brain–LLM experiment.

### Venue information

<a id="ref-r27"></a>
**[R27] COLING 2027 official main-conference call.** Location: `https://2027.coling-iccl.org/calls/main_conference_papers/`. Accessed for this proposal on September 30, 2026. Submission dates and formatting requirements are administrative facts that should be rechecked at submission.

### Official checkpoint registry

The following official model cards were inspected for candidate existence and identity. Exact revisions, weights, licenses/access acceptance, compatible implementations, and local execution must still be recorded before experiments. Names containing “Instruct,” “it,” or post-training designations identify the selected checkpoint; they do not guarantee a particular result in this study.

| ID | Checkpoint | Official card location | Project use |
|---|---|---|---|
| M01 | `openai-community/gpt2-xl` | `https://huggingface.co/openai-community/gpt2-xl` | Evaluated representation. |
| M02 | `google-bert/bert-base-uncased` | `https://huggingface.co/google-bert/bert-base-uncased` | Evaluated prefix-constrained representation. |
| M03 | `Qwen/Qwen2.5-1.5B` | `https://huggingface.co/Qwen/Qwen2.5-1.5B` | Evaluated representation. |
| M04 | `Qwen/Qwen2.5-7B` | `https://huggingface.co/Qwen/Qwen2.5-7B` | Evaluated representation. |
| M05 | `Qwen/Qwen2.5-7B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct` | Evaluated representation. |
| M06 | `Qwen/Qwen3-8B-Base` | `https://huggingface.co/Qwen/Qwen3-8B-Base` | Evaluated representation. |
| M07 | `Qwen/Qwen3-8B` | `https://huggingface.co/Qwen/Qwen3-8B` | Evaluated raw-stimulus representation, without generated reasoning. |
| M08 | `meta-llama/Llama-3.1-8B` | `https://huggingface.co/meta-llama/Llama-3.1-8B` | Evaluated representation. |
| M09 | `meta-llama/Llama-3.1-8B-Instruct` | `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct` | Evaluated representation. |
| M10 | `google/gemma-2-2b` | `https://huggingface.co/google/gemma-2-2b` | Evaluated representation. |
| M11 | `google/gemma-2-9b` | `https://huggingface.co/google/gemma-2-9b` | Evaluated representation. |
| M12 | `google/gemma-2-9b-it` | `https://huggingface.co/google/gemma-2-9b-it` | Evaluated representation. |
| M13 | `Qwen/Qwen2.5-32B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-32B-Instruct` | Proposed annotation production. |
| M14 | `meta-llama/Llama-3.3-70B-Instruct` | `https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct` | Proposed independent annotation audit. |

**End of proposal.**
