# Project handoff: scientific scope and design decisions

**Recorded:** 2026-10-04  
**Purpose:** Primary high-level reference for future research and coding threads.  
**Status:** Decision record and scientific framing; not a report of experimental findings or a binding implementation manual.

Read this file before proposing a new scope, interpreting the older proposals, or splitting implementation among agents. It preserves the conceptual discussion, the researcher's corrections to earlier advice, and the resulting direction. Most of this document concerns decisions made before implementation. Section 13 separately records operational decisions that followed, so they are not mistaken for original scientific commitments.

**Accepted-review integration, 2026-10-05:** The researcher closed annotation
review and authorized downstream implementation against immutable build
`5cd83e0779bb5440da47`. This supersedes earlier instructions to wait for annotation
or human adjudication before integration. Preserve the accepted archive and its
uncertainties; acceptance does not constitute an independent accuracy estimate.
The active downstream contract is [reviewed_downstream.md](reviewed_downstream.md).
Cluster verification is paused; local compilation and real-corpus verification
precede researcher-operated cluster execution. The full scientific scope remains.

## 1. Source authority and how to use this record

The researcher originally supplied two proposals and a longer discussion that generated the plan:

- [Original shared research conversation](https://chatgpt.com/share/6ac18a1b-4f5c-83e9-b542-b85d56aa0512), explicitly designated by the researcher as central evidence above the Markdown plans.
- [First-principles / human-reference proposal](docs/Structured_Meaning_Brain_LLM_Human_Reference.md).
- [Detailed technical proposal](docs/Structured_Meaning_Brain_LLM_Research_Proposal.md).

The researcher was satisfied with the broad design and brainstorming but considered the technical proposal excessively prescriptive. The request was to understand and evaluate the research, not mechanically implement every instruction in that document.

**Authority:** Later explicit researcher instructions govern. This handoff is the primary entry point to the agreed direction; the underlying researcher discussion takes precedence if this summary misrepresents it. The first-principles document explains the ideas. The technical proposal supplies methodological detail, candidate designs, and a reference registry. Statements in the older proposal calling itself authoritative do not override the researcher's subsequent corrections.

**Source-coverage note:** This is a consolidated decision record, not a verbatim conversation export. It was assembled from the researcher messages available in this thread and the two repository proposals. The shared link could not be reopened when writing this handoff; unavailable dialogue has not been reconstructed or attributed as a quotation. Scientific explanations inherited from the proposals are distinguished below from explicit researcher preferences and later implementation choices. This handoff is not a new systematic literature review.

Future threads should preserve three distinctions:

1. **Agreed direction:** the scientific purpose and explicit researcher decisions recorded here.
2. **Implementation choice:** a revisable method for realizing that direction, whose consequences should be understood.
3. **Open empirical question:** something the data must answer, not an assumption that an agent should turn into a result.

## 2. The project in one coherent statement

Study how **concepts, role-bound events, reference, and discourse relationships** are represented in human brain responses and contemporary frozen language models processing corresponding text. Construct an explicit semantic description of the recorded stimuli, then use neural encoding, semantic decoding, and representational geometry to examine what information is available, how it is organized, and where humans and models agree or differ.

The project connects three objects describing the same language:

- **Observed human responses:** existing stimulus-matched fMRI recordings.
- **Model representations:** internal activity extracted from frozen language models given the corresponding text.
- **Structured semantics:** evidence-linked concepts, entities, events, roles, references, and relationships extracted from that text.

The graph is a shared analytical description and source of semantic supervision. It is not an assertion that the brain literally stores a graph or executes the chosen symbolic program.

The aim is broader than maximizing a brain-prediction score or producing an attractive concept map. It is to discover relationships among **predictive information, recoverable meaning, and representational organization**. Agreement, partial agreement, context dependence, and reliable dissociation are all meaningful outcomes.

## 3. What prompted the evaluation, and the decision to proceed

The researcher's initial concerns were substantive:

- Is the complete idea scientifically sensible and potentially suitable for a strong venue such as COLING?
- Can it be implemented locally, and which parts require the available A100/H100/L40S-class cluster resources?
- How do we construct and label a useful semantic graph from an actual paired text–brain dataset?
- Which dataset best supports the intended questions?
- How can model–human prediction and concept grounding be evaluated without anatomical concept ground truth?
- What has to be trained, and what can remain frozen?
- How much do encoding, decoding, and concept-distance analyses depend on annotation quality?
- Why retain old model checkpoints simply because earlier papers used them?

**Resulting decision: proceed with the integrated research program.** The combination of existing neural recordings, frozen-model feature extraction, structured annotations, and trained readouts is implementable with the available local and cluster resources. Full language-model pretraining or fine-tuning is not a prerequisite for the central study.

This is a feasibility judgment and a research commitment, not a promise of acceptance or a prediction of positive results. The paper's strength will come from a clearly posed semantic question, a credible measurement program, and substantive findings about human–model agreement or divergence. The graph resource and reusable software are valuable contributions, but the scientific interpretation remains central.

COLING is a possible destination. It is not a deadline that defines the allowable scientific scope. If its submission cycle does not fit, pursue another suitable venue without automatically compressing the project.

## 4. The researcher's corrections that future threads must retain

### 4.1 Implementation pace and scope

The researcher explicitly rejected a calendar-driven, repeatedly narrowed "pilot-test-smoke-repeat" workflow and asked for a **"one-shot approach workflow for implementation."** The intended approach is a comprehensive, integrated codebase designed around the complete research questions and real data dependencies.

Consequences:

- Do not repeatedly replace the full program with a tiny demonstration, then ask whether to continue.
- Do not make a favorable initial effect a prerequisite for implementing the other branches.
- Organize implementation by dependencies and interfaces, not by a sequence of reduced scientific ambitions.
- Use actual data and meaningful correctness checks. Do not manufacture placeholder datasets, synthetic neural responses, dummy graphs, or smoke outputs to represent progress.
- When an inaccessible input or a long execution genuinely blocks the next step, give a concrete handoff to the researcher instead of filling the gap with scaffolding.

One-shot implementation still includes ordinary debugging, real-data validation, and correction of actual errors. Those activities establish that the intended measurements are implemented correctly; they are not a rationale for repeatedly shrinking the scientific design.

### 4.2 Novelty assessment

The researcher rejected novelty judgments based on overlapping keywords or superficial descriptions. Prior work involving graphs, semantic features, concept decoding, or brain–model similarity does not by itself establish that this project's question, method, comparison, or interpretation has already been done.

Compare related work concretely along these dimensions:

| Dimension | What a useful comparison establishes |
|---|---|
| Scientific question | The actual phenomenon investigated and inference sought. |
| Stimuli and modality | Images/video, isolated sentences, or connected language; reading versus listening. |
| Semantic object | Concept occurrence, argument binding, reference, event relations, or their interaction. |
| Measurement direction | Encoding, decoding, grounding, native activity geometry, or a joint analysis. |
| Supervision and access | What is trained, what labels exist, and what information enters each model. |
| Generalization | New contexts, stories, participants, combinations, model families, or conditions. |
| Interpretation | Predictive association, accessible semantic information, learned support, or claims about underlying organization. |

Acknowledge direct precedents and use them as foundations or informative baselines. Do not retreat to a narrowly defined novelty claim merely because the ingredients have names in common with earlier work. Equally, do not claim that familiar individual techniques were invented here. The proposed contribution is the integrated question and the empirical account it enables, with any methodological advances stated specifically.

Earlier advice described in the researcher's correction used terms such as "carefully controlled," "targeted small batch," and "bounded" to justify restriction. Those terms do not establish a scientific reason to reduce scope. A necessary control must address a real alternative explanation; it should not become an unlimited checklist that displaces the actual study.

### 4.3 Scientific rigor without unnecessary machinery

The researcher later asked whether all proposed downloads and components were essential. Preserve the complete scientific objectives while selecting components for their contribution to those objectives. Neither every dataset in the proposal nor every possible control must become an unconditional dependency.

Conversely, essential distinctions—actual recordings versus predictions, semantic labels versus anatomical labels, and held-out evaluation versus training fit—cannot be discarded to make execution easier. These define what the measurements mean.

## 5. The conceptual levels the project must preserve

The semantic program spans more than bags of concepts:

| Level | What is represented | What would otherwise be missed |
|---|---|---|
| Concept/type | A person, book, lending, returning, intention. | Basic semantic content. |
| Entity and event instances | A particular participant, object, or occurrence in a story. | Which occurrence or referent a statement concerns. |
| Role binding | Who did what to whom, with what object, in which role. | Sentences with the same words but reversed participants. |
| Reference | Which mentions refer to the same entity or event. | Meaning maintained or revised across sentences. |
| Event/discourse relations | Temporal order, supported cause or motivation, conditions, and other links. | Relationships beyond isolated events. |
| Scope and status | Negation, reported information, hypothetical events, intentions, and uncertainty. | The difference between an event being discussed and being asserted to occur. |

Three meanings of hierarchy remain distinct: **taxonomic** organization of concept types, **compositional** organization of entities/roles/events/relations, and **temporal** accumulation over a passage. A longer context window is not automatically a more compositional representation, and none of these hierarchies presupposes a matching cortical hierarchy.

Concept-level comparison remains an explicit objective. It should not silently be replaced by only stimulus-level RSA because the latter is easier to implement. Stimulus-level comparisons are complementary and help connect aggregate concept estimates to actual observations.

## 6. What graph construction and labeling mean here

Start from the original linguistic stimuli paired with neural recordings. Preserve their identity, wording, and timing. Produce a typed semantic graph whose assertions have supporting text spans and known information availability.

An AMR/UMR-informed representation is a useful starting point, not a requirement to reproduce an entire formalism or its parser. The scientific needs determine the schema. Nodes and edges should distinguish concept types, story-local entities, mentions, events, role arguments, reference links, discourse relations, scope, uncertainty, and changes in described state.

Important semantic distinctions include:

- A name or local entity ID is not a transferable semantic concept. Cross-story concept geometry requires an explicit type, sense, role, or event-schema definition.
- A promise, desire, hypothetical action, and completed action are different claims.
- Surface event order does not establish causation. Record supported relations and retain uncertainty when the text does not resolve them.
- A graph need not be a tree. Shared arguments and reference create legitimate reentrancy; only relations whose meanings require acyclicity should receive that constraint.
- Missing annotation is not automatically a false proposition. Negatives and alternatives must be justified within the relevant textual scope.

For analyses intended to reflect information available as a passage unfolds, annotations must use only the relevant text prefix. Later revelations cannot silently rewrite earlier labels. A persistent entity/event ledger can maintain reference across prefixes; its contents must also come from earlier available text. Retrospective full-story analysis is a different information condition and should be identified as such.

This temporal availability rule concerns the research measurement. It is separate from the researcher's rejection of incremental software delivery. A comprehensive implementation can process a story sequentially without becoming a pilot-driven research workflow.

The annotation pipeline yields two principal downstream products:

1. Numerical semantic features for encoding and representation analysis.
2. Semantic queries and answers for decoding, including concept occurrence, role assignment, binding verification, reference, and supported event relations.

The full graph supplies labels and feature construction. It must not also be handed to a decoder as an input that reveals the answer.

### Annotation model and quality decisions

The researcher explicitly allowed a **closed model** for annotation. There is no requirement to prefer a small open model when a closed model provides better labels within the available budget. The original technical proposal's open-weight annotator pair is therefore not binding.

The researcher proposed using the authenticated **Codex CLI with ChatGPT usage** instead of paying for separate API calls. The later operational preference was the cheapest adequate model and no high reasoning. These are resource decisions; they do not establish annotation quality in advance.

Labels need evidence, consistent semantics, automated structural checks, and human examination of actual outputs. Review should cover the phenomena that drive the paper, including difficult role, reference, scope, and discourse cases. Report what was checked and the resulting uncertainty. A valid JSON object, a model's confidence score, or agreement between two runs of the same model is not a human accuracy estimate.

An independent second annotator can supply useful evidence, but a second full same-model pass is not intrinsically required for rigor. Exact human-review budgets and coverage thresholds in the technical document are planning defaults, not immutable research laws. Final choices should reflect observed semantic coverage and the claims being tested, without selecting labels according to favorable brain/model results.

## 7. Frozen models, trained mappings, and the ground-truth question

The initial discussion asked whether the project must train something instead of evaluating a frozen system "as NEURONA does." The important clarification is **which component is frozen** and **which ground truth is absent**.

The working account of NEURONA in both proposals is answer-supervised neural decoding: stimulus-derived graphs generate queries and semantic answers; a trained decoder learns to answer from fMRI. Intermediate concept-to-region assignments do not have anatomical supervision. Thus, absence of concept-localization labels does not mean that its entire decoder is untrained or that final-answer evaluation has no targets. See the technical proposal, §2, for its source-backed account and references.

For this project:

| Component | Treatment | Available target or evidence |
|---|---|---|
| Pretrained language model | Keep weights frozen for representation comparisons. | Native states extracted from the recorded text. |
| Annotation model | Use to construct semantic records; no project-specific fine-tuning is required by the central design. | Source-supported labels and human review. |
| Neural encoding map | Fit on training data. | Actual measured fMRI. |
| Brain semantic decoder | Fit on training examples. | Semantic query answers supported by the stimuli. |
| Model-state semantic decoder | Fit separately on frozen model states. | The corresponding semantic query answers. |
| Intermediate grounding maps | Learned or estimated according to the chosen method. | Indirect task, dependence, reliability, and replication evidence; no assumed anatomical answer key. |
| Native/measured geometry | Estimate from observed representations and semantic instance definitions. | Repeatability and cross-system relationships, with uncertainty. |

There is therefore no need to invent a ground-truth cortical location for every concept. Encoding is scored against recordings; decoding is scored against semantic targets. Grounding is assessed through its usefulness, dependence on the correct neural input, stability, and correspondence with other measurements.

The semantic answers concern the text. They do not certify a participant's conscious interpretation or prove that the participant performed the generated question-answering task.

## 8. Encoding, decoding, and geometry are peer scientific components

### Encoding: what predicts the measured response?

Fit mappings from text-derived representations to recorded fMRI and evaluate on held-out material. Compare semantic content, structured semantic information, frozen-model states, relevant presentation/context variables, and combinations that test additional predictive information.

The key question is not just which model has the highest score. It is whether role, reference, or discourse structure explains measurable variation beyond simpler descriptions, and how that relationship varies across models, layers, contexts, and brain regions.

Ridge or related regularized estimators are sensible methods, not the high-level objective. Fitted coefficient maps or predicted responses are model-derived quantities and must remain distinguishable from measured activity.

### Decoding: what meaning is recoverable?

Fit a mapping from measured fMRI and a semantic query to an answer. Fit a separate mapping from cached frozen-model states to the same kind of answer. Use comparable semantic targets and clearly defined access to queries and candidates.

During evaluation, the brain decoder does not receive the original passage, answer graph, answer-revealing evidence spans, or correct answer. The model-state decoder uses representations extracted before the query is introduced; asking an LLM to reread the passage and answer the question would measure a different capability.

Compare simple readouts with a structured decoder that uses concept and relation evidence. The full design retains neuro-symbolic decoding rather than quietly reducing it to a collection of independent classification heads. Exact module architecture, loss, and optimization details remain implementation choices to be justified against appropriate baselines.

A query-only baseline with the same legal candidate information tests whether questions or answer choices make neural input unnecessary. Correct answers indicate accessibility through a specified decoder, not a literal reconstruction of biological computation.

### Geometry: how is semantic information organized?

Estimate concept-, role-, or event-associated signatures across appropriate occurrences. Compute relationships within the brain-derived representation and within model representations, then compare those relationship structures over the same semantic items. Their vector dimensions need not match; no one-to-one voxel–model-unit correspondence is assumed.

Retain the four signature families developed in the conceptual design:

| Family | What is compared | Interpretation |
|---|---|---|
| Grounding profiles | Distributions of concept-related readout evidence over brain regions or declared model support units. | Where a particular decoder obtains evidence. |
| Measured/native patterns | Concept-conditioned brain responses or frozen-model activity. | Organization estimated from observed representations. |
| Learned latent signatures | Intermediate representations learned by readouts. | Geometry of a specified trained representation. |
| Encoding-implied signatures | Response patterns or contrasts predicted by an encoding model. | Organization implied by the fitted predictive model. |

These are different objects, not interchangeable versions of a single ground-truth concept map. Preserve them as possible sources of findings instead of imposing one universal map in advance.

Cosine similarity or an RDM describes relationships between specified vectors. It does not establish anatomical proximity, causal influence, or a unique semantic metric. Roles and directed event relations need explicit treatment; a symmetric similarity matrix cannot encode all of them.

A learned latent can change coordinates while preserving task predictions, so its metric is partly method-dependent. Likewise, composing an event vector from argument vectors can create similarity by construction. Compare primitive and composed evidence and use native/measured patterns where informative. If a model is explicitly trained to align brain and model geometry, that resulting alignment is not independent evidence of naturally matching organization.

### How the branches relate

Encoding, decoding, and geometry have equal scientific standing. Encoding and decoding use separately fitted mappings and objectives. Neither must achieve a positive result before the other can be implemented or reported.

Measured/native geometry can operate without a trained semantic decoder. Grounding and learned-latent geometry depend on decoding; encoding-implied geometry depends on encoding. These are execution dependencies, not a ranking of research importance.

The branches still share stimuli, labels, and recordings. Separate fitting does not make their evidence statistically independent. A difference between them can be the finding: for example, relational information might be recoverable while adding little aggregate encoding accuracy.

## 9. Dataset and model-selection direction

### Paired neural data determine the usable stimulus corpus

Begin with actual recorded stimuli and their neural measurements. A large unrelated text corpus cannot increase the number of neural observations. Evaluate candidate datasets for timing quality, access, repeated measurements, anatomical information, story/context diversity, and coverage of the semantic distinctions the project needs.

Reading is the preferred setting because the presentation modality matches the models' text input more directly. The planning documents identify:

- **Deniz reading:** the central multi-story resource, subsequently selected for implementation.
- **Wehbe reading:** a complementary narrative resource, with the limitations of a single chapter.
- **Pereira2018:** complementary sentence/passage-scale repeated measurements.
- **LeBel / story-listening resources:** an alternative or extension where broader coverage justifies the different modality and relevant acoustic controls.

These resources are not an instruction to download everything. Additional corpora should support a named corroboration or scale comparison. Listening does not automatically require adding an entire speech-model project.

Dataset descriptions in the proposals were provisional. For example, their Deniz participant count and access caveats must yield to the inspected release. At the implementation boundary, the downloaded Deniz reading material was verified to contain **nine participants and eleven stories**, rather than treating the earlier six-participant planning statement as authoritative. Current manifests and inspection artifacts establish exact coverage and array structure.

Natural narratives do not guarantee factorial coverage of every concept–role–relation combination. Measure actual support before interpreting compositional generalization or concept-level distances. Insufficient coverage limits a particular inference; it does not by itself justify abandoning the entire integrated project.

### Contemporary frozen models are central

The researcher explicitly objected to old checkpoints retained solely because earlier literature used them. Select contemporary, inspectable models that address meaningful contrasts such as family, scale, layer, context, and base versus post-trained representations where available.

A legacy checkpoint can remain when it answers a specific continuity or baseline question. It must earn that role. The technical proposal's fixed twelve-model list is not an enduring requirement, and model recency alone is not evidence of better brain agreement.

Pin the actual chosen revisions and extraction semantics for reproducibility. Exact rosters, numerical precision, layer selection, context lengths, and cache layouts belong in maintained configurations and model manifests. They should not be frozen indefinitely by this high-level handoff.

The annotation model and the evaluated representation models have different jobs. Their identities and possible shared biases should be recorded; a label generator is not an oracle defining which evaluated model understands meaning correctly.

## 10. The essential evaluation logic

The following principles protect the meaning of the experiment. They are not a demand to accumulate every conceivable analysis before making progress.

- **Test beyond fitting examples.** Group related questions and observations by their real stimulus support. Use story/context divisions and appropriate separation for overlapping text and fMRI windows. Fit learned preprocessing and tuning choices using the allowed training partitions.
- **Separate content from structure.** Evaluate role assignments, reference links, and event relationships in ways that cannot be explained solely by which concepts appear. Preserve valid semantic scope when generating alternatives.
- **Demonstrate dependence on the representation.** Use informative simple/query-only baselines and disrupted neural pairings or related controls to test whether the correct fMRI or model state matters. A trained model evaluated on mismatched inputs and a model retrained under mismatching answer different questions.
- **Respect temporal measurement.** Prefix-limited text access does not eliminate fMRI delay or mixing with nearby material. State the supported temporal interpretation and examine relevant alignment/lag assumptions.
- **Measure reliability and coverage.** Aggregate concept estimates need independent occurrences. Interpret weak agreement alongside the reliability of both sides. Many questions from one passage are not many independent neural samples; many RDM cells also share concepts.
- **Use existing recordings honestly.** Editing text or a graph does not create the brain response to that edited stimulus. Predicted fMRI and computational counterfactuals remain model outputs. Existing data can test representational associations and accessibility without establishing biological necessity.
- **Keep evaluation definitions independent of desired outcomes.** Annotation adjudication and planned comparisons should not be revised merely to favor a model or improve an attractive map. Document actual corrections and exploratory analyses.

Human review and annotation uncertainty are part of this logic because the graph defines many downstream targets. Numerical labels and metrics do not remove semantic ambiguity.

## 11. Contribution framing and informative outcomes

The working contribution is an integrated investigation of structured meaning across neural and model representations, supported by reusable semantic annotations and analysis software. The motivating precedents are integrative model-to-brain prediction and NEURONA-style answer-supervised grounding; interpretable semantic encoding, role-sensitive neuroscience, graph semantics, and representation analysis provide additional foundations.

The project should ask questions that an aggregate alignment leaderboard does not answer: Which semantic distinctions are recoverable? Which improve prediction? Which have comparable geometry? How do those relationships change with context, model depth, and the level of composition?

Possible informative outcomes include:

- Agreement for broad concepts but reliable differences in roles, reference, or discourse.
- Strong semantic decoding with little additional encoding gain from explicit graph features.
- Predictive structural features whose learned grounding maps vary with decoder assumptions.
- Context-dependent organization that is reproducible across examples or participants.
- Evidence that apparently meaningful geometry is primarily supplied by annotation or architecture.
- A distinction left unresolved because the available measurements or semantic coverage are insufficient.

No branch is required to win a leaderboard or show a positive effect. A universal concept atlas, identical human–model organization, and unique cortical locations are not prerequisites for a successful study.

A publication argument should compare actual questions and methods with current literature, then state the specific findings supported by the complete evidence. The proposals' reference registry is a starting point, not a claim that all later work has been exhaustively excluded. Literature gaps should be investigated concretely rather than used as vague reasons to narrow ambition.

## 12. Integrated implementation strategy

The intended dependency structure is:

```text
Paired recordings and text ──> verified stimulus/timing interface
                                      │
                       ┌──────────────┴──────────────┐
                       │                             │
               Semantic annotation          Frozen-model extraction
                       │                             │
             Graph features + queries + shared evaluation divisions
                       │                             │
                       └──────────────┬──────────────┘
                                      │
                     ┌────────────────┼─────────────────┐
                     │                │                 │
                  Encoding         Decoding      Measured/native geometry
                     │                │                 │
              Encoding-implied    Grounding and         │
                  geometry       latent geometry        │
                     └────────────────┼─────────────────┘
                                      │
                         Controls, inference, interpretation
```

The diagram describes data dependencies, not a requirement to wait before writing each dependent module. Once the interfaces are defined, graph compilation, model extraction, encoding/decoding, and geometry/statistics code can be developed concurrently in one coherent repository. Actual final labels are required to execute label-dependent analyses and establish coverage, not to begin all downstream coding.

Use one integration owner and clear module/file ownership when dividing work across agents. Keep running annotation inputs and contracts stable so unrelated code development does not invalidate resumption. Parallel agents can shorten elapsed coding time, but do not inherently reduce usage cost.

Local work includes repository implementation, schema and query logic, real text/timing preparation, annotation orchestration, and inspection of returned artifacts. Cluster work includes substantial downloads, model-state extraction, and large fits or resampling jobs. Whether a particular readout needs a GPU depends on the chosen estimator and scale; cluster access does not imply every operation belongs on a GPU.

The complete codebase should express the scientific design directly. It should not contain dummy implementations presented as completed branches. An unavailable dependency should produce a clear, actionable stop rather than invented outputs.

## 13. Later operational decisions retained for continuity

These decisions followed the conceptual discussion. They are included to prevent future threads from restarting settled workflow debates, not to convert this document into a live operations log.

| Decision | Rationale and consequence |
|---|---|
| Use Deniz reading as the current core dataset. | Actual paired text, timing, responses, and supporting files were acquired and inspected. Exact release facts belong to manifests/contracts. |
| Download large data directly on the cluster. | The researcher's local network should not govern bulk acquisition speed. Code development remains local. |
| The researcher operates the persistent SSH session. | Login requires interactive MFA. Do not connect to or change the cluster directly unless explicitly asked. |
| Inspect storage, workspace, allocation, and environment before setup. | This inspection was completed for the current project; do not restart a speculative separate-quota investigation without new evidence. |
| Build and run only on allocated compute nodes. | Environments, model caches, dependency installation, downloads, and compute jobs belong under the project workspace. Nothing should be built on the login node or in home storage. Lightweight login-node inspection/submission is distinct. |
| Hand long runs and inaccessible sites to the researcher. | Provide exact steps or copy-pastable commands, then wait for the actual required outputs. Do not spend the session polling long jobs or fabricate replacements. |
| Use CMD for Windows instructions. | Absolutely no PowerShell in user-run copy-pastable commands. Commands for the existing Linux SSH session must be clearly identified as remote shell commands. |
| Use direct instruction and authoring for production annotation. | The researcher discarded the ineffective CLI pipeline and explicitly superseded its Luna/none preferences. The earlier CLI attempt remains development evidence. See [the production annotation record](direct_annotation.md). |
| Use progressive isolated story contexts. | The researcher explicitly authorized isolated annotation agents. Actual incremental authoring receives one unit at a time without later passages or completed target-story graphs. Save evidence and available model/settings provenance; do not infer unavailable settings from historical configuration. |
| Keep this annotation task separate from downstream analysis changes. | The researcher will address downstream compatibility in a separate thread. This task produces annotations and shared deterministic, lossless annotation exports, without fitting features or changing encoding, decoding or geometry modules. |
| Use the versioned independent review for the current annotation input. | Resolve `data/annotations/deniz-independent-review-v1/latest.json`, then pin its immutable `recommended_input` and build hash. The [scientific review handoff](independent_review_handoff.md) defines surgical corrections, preserved alternatives, compiler semantics and pending human adjudication. Original direct-authoring exports remain intact. |

The cluster project workspace established in this thread is `/weka/projects/tshu2/zzhan330/neurosymbolic_coognitive_linguistics`; the login is `zzhan330@login.arch.jhu.edu`. This records the agreed destination, not permission for an agent to operate it independently.

The local workspace is `C:\Users\pigby\neurosymbolic_coognitive_linguistics`.

Current operational details should be read from the actual repository, including [annotation configuration](configs/annotation.json), [dataset manifest](manifests/deniz-reading.json), and [frozen-model registry](manifests/frozen-models.lock.json). Files, jobs, or artifacts appearing after this handoff may reflect further work. Do not infer current completion, annotation quality, or scientific results from this conceptual record.

## 14. What remains open without changing the agreed scope

The following are implementation or research choices to resolve against real evidence; they are not reasons to restart feasibility discussion from scratch:

- Final semantic normalization, relation inventory, treatment of difficult senses, and handling of ambiguous readings.
- The human-review protocol and any additional independent annotation needed for the reported claims.
- Which concepts/compositions have enough actual support for particular estimates, without inheriting arbitrary numerical thresholds from the proposal.
- Exact query/candidate construction and grouping rules for within-story, cross-story, and compositional tests.
- Brain regions/atlas, alignment details, temporal windows, and appropriate reliability estimates supported by the released data.
- Contemporary model panel, precise revisions, extraction conditions, and resource allocations. These may already have been resolved by a later implementation branch; inspect its records first.
- Exact encoding estimator and structured-decoder architecture, together with the baselines needed to interpret them.
- Operational definitions and estimation procedures for the different geometry families.
- Which complementary dataset genuinely adds useful evidence, and what source/stimulus overlap it has with the core data.
- Planned statistical contrasts and the computational budget for inference and null fits.
- The final empirical emphasis of the paper and its venue, after the integrated evidence is available.

Routine engineering choices can be made within the authorized scope. A change that redefines a semantic target, replaces a dataset, removes a scientific branch, or changes what a measurement means should be made explicit. Do not turn every ordinary implementation choice into a new approval checkpoint.

## 15. Compact decision ledger for future threads

| Topic | Decision to retain | Earlier assumption or shortcut it supersedes |
|---|---|---|
| Whether to pursue | Proceed with the integrated program using real paired data and available compute. | Indefinite feasibility discussion or a demonstration-only substitute. |
| Source authority | Researcher discussion and later corrections govern; this file is the high-level entry point. | Treating the technical proposal as an executable manual. |
| Timeline | Venue choice follows scientific readiness. | Deadline-driven reduction of the central questions. |
| Workflow | Comprehensive implementation against shared contracts, with meaningful real-data verification. | Repeatedly shrinking to pilots, placeholders, or synthetic smoke outputs. |
| Novelty | Compare actual questions, methods, evidence, and interpretation. | Declaring the project already done from overlapping terminology. |
| Scientific scope | Preserve concepts, roles/events, reference/discourse, encoding, decoding, and geometry. | Demoting a branch because another is easier or more familiar. |
| Ground truth | Use observed fMRI and semantic answer targets; anatomical groundings lack direct labels. | Calling the entire task ground-truth-free or assuming nothing is trained. |
| Frozen versus trained | Freeze representation models; train the necessary mappings/readouts separately. | Fine-tuning all models or treating trained probes as incompatible with frozen evaluation. |
| Graph labels | Evidence-linked, time-aware semantics with uncertainty and human review. | Equating parser validity or same-model agreement with semantic truth. |
| Annotator | Direct instruction and progressive authoring are the primary production attempt; preserve available provenance. | Mandatory older open-weight annotators, the discarded CLI runner, and earlier Luna/none preferences. |
| Model panel | Contemporary models selected for useful comparisons, with justified legacy baselines only. | Permanently retaining the old fixed roster because earlier papers used it. |
| Datasets | Verified core reading data plus scientifically motivated complements. | Downloading every listed resource or counting unrelated text as neural evidence. |
| Geometry | Preserve concept comparisons and distinguish measured, grounded, latent, and predicted signatures. | Treating every embedding or map as the same biological object. |
| Outcomes | Agreement and reliable dissociation both count; report unresolved measurements honestly. | Requiring a positive effect, a universal map, or a leaderboard win. |
| Collaboration | Agents implement integrated modules; the researcher handles access barriers and long cluster runs. | Direct cluster operation without authorization or blocking all coding behind annotation. |

## 16. Maintaining this handoff

Use this file to orient future threads and to resolve conflicts with old proposal defaults. Read the technical proposal for relevant methodological detail, then inspect the current code and manifests before deciding something is missing or fixed.

When the researcher makes a new high-level decision, update the relevant section and record what it supersedes. Keep changing run counts, job IDs, environment versions, and troubleshooting history in operational records. Do not allow an implementation workaround to silently become a new scientific objective.

**Continuing brief:** Build the full study of structured linguistic meaning in human neural responses and modern frozen language-model representations. Preserve the complementary encoding, decoding, and geometry questions; use the actual recorded stimuli and justified semantic labels; keep claims matched to the evidence; and execute through the agreed local-development and researcher-operated cluster workflow.
