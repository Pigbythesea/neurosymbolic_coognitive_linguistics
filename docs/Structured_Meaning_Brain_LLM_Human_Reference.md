# How Brains and Language Models Represent Meaning
## Research proposal — human reference

> **Read first:** [Project handoff: scientific scope and design decisions](../PROJECT_HANDOFF.md) records the researcher's later decisions and is the primary high-level reference. This proposal remains supporting material; conflicting model lists, dataset assumptions, and implementation defaults are superseded by that handoff and subsequent explicit decisions.

*Companion to **Structured Meaning in Human Brains and Language Models**, version 1.0. This is a proposed study; no experiments or results are reported here. Detailed software specifications remain in the original document.*

## 1. Start with the basic problem

Consider this passage:

> Leah lent Noah a book. Noah promised to return it. Because Leah needed the book back, she reminded him. Noah returned it.

Understanding this passage involves several kinds of information:

| Kind of information | What the reader needs to understand |
|---|---|
| Individual concepts | The passage involves people, a book, lending, promising, reminding, and returning. |
| Roles within an event | Leah is the lender. Noah is the recipient. The book is the item being lent. |
| References across sentences | “She” refers to Leah; “him” refers to Noah; “it” refers to the book. |
| Relationships between events | Leah’s stated need explains the reminder. A promise to return something is different from actually returning it. |

Changing “Leah lent Noah a book” to “Noah lent Leah a book” preserves the people, object, and action. It changes who did what. A representation that records only which concepts appear could miss this difference.

The project asks:

> **Which of these kinds of meaning can we recover from human brain recordings and language-model activity? How are they organized within each system, and where do the systems agree or differ?**

Here, a **representation** means a pattern of activity that can carry information. In an fMRI recording, it is a pattern across measured brain locations. In a language model, it is a pattern of numerical values produced while processing text. We investigate what those patterns tell us about meaning.

We will examine individual concepts, events with their participant roles, and relationships across sentences. **Discourse** means connected language extending across sentences. We will also examine how the findings change with the amount of preceding text, the brain region, and the stage of processing inside a model.

The outcome could be a reproducible concept map, a difference between two measurements, or a finding that agreement holds for some kinds of meaning and weakens for others. We leave those possibilities open.

*The passage above is an invented explanation. It has no associated brain recording.*

## 2. What we would actually have in our hands

Every usable example starts with **text that someone actually encountered during an fMRI experiment**, together with the corresponding recording. A large text collection without paired recordings cannot supply additional neural evidence.

We then assemble three descriptions of the same material.

**The brain recording.** fMRI measures changes related to blood oxygenation. It provides an indirect, delayed measurement of neural activity across many small brain volumes, called **voxels**. Several voxels can be grouped into a larger analysis region, called a **parcel**. Nearby events in time can contribute to overlapping fMRI responses.

**The language-model activity.** We give a language model the corresponding text and save its internal numerical values, often called **hidden states**. These values form a vector: simply a list of numbers. We keep the model’s original parameters unchanged. This is what **frozen** means in the proposal.

**A structured description of the meaning.** We record the concepts, event roles, references, and event relationships supported by the text. For the example, one record would say that Leah lends, Noah receives, and the book is transferred. These connected records form a **semantic graph**: entries for things and events, with labeled connections between them.

“Symbolic” refers to these explicit labels and connections. “Neuro-symbolic” describes a method combining them with learned numerical models. Choosing this description gives us a clear set of questions to test; it does not establish that the brain literally stores a graph.

### What is known, and what is inferred?

The scanner supplies the measured response. The passage supplies evidence for semantic answers, such as who returned the book. Neither supplies a correct anatomical label saying where “returning” must be represented.

This distinction makes the project possible. We can evaluate predictions of recorded activity and predictions of semantic answers. Intermediate concept maps receive indirect evaluation through those tasks and through separate checks of their stability and dependence on neural data.

The semantic labels describe the text. They do not certify that a participant consciously noticed or reasoned about every annotated relationship.

## 3. How the motivating papers lead to this study

**Schrimpf and colleagues** gave models the language stimuli used in human experiments. They trained mappings from model activity to neural recordings and evaluated predictions on held-out stimuli. “Held out” means excluded from fitting the mapping. This supplies the foundation for measuring how well a representation predicts brain responses. [Schrimpf et al., 2021; technical proposal §2]

**NEURONA, from Wang, Hsu, Adeli, and Wu**, learns to answer questions about images or videos from fMRI. Its questions have explicit concept and relationship structure. Training supplies the correct final answer, while the system learns intermediate concept-to-region assignments without anatomical labels for those assignments. It reports answer accuracy, compositional generalization, and grounding consistency. The original proposal also records verification that its released BOLD5000 annotation script uses GPT-4o. [Wang et al., 2026; technical proposal §2]

Our project integrates these directions for language. It adds a systematic comparison of concepts, event roles, and discourse relationships across brain responses and frozen model states, using several separately defined ways of measuring their organization.

The individual ingredients already have precedents. QA-Emb uses answers to semantic questions as interpretable features for predicting fMRI. Other work studies composed meaning, semantic relations, and who-did-what-to-whom information. NEURONA itself includes embedding retrieval and correlations between grounding profiles. [Technical proposal §3]

**The proposed contribution is the joint experiment and the findings it enables:** determining whether the same semantic distinctions are predictive, recoverable, and similarly organized across systems. Simply adding a graph or a cosine-similarity calculation would provide a limited extension.

## 4. Two equally important measurements

We run neural encoding and semantic decoding in parallel. Each has its own fitted mapping, objective, and evaluation. Neither must succeed before we run or report the other.

### Neural encoding: can a description predict the recorded response?

The direction is:

**Text-derived features → fitted prediction model → predicted fMRI.**

A **feature** is a numerical description of some property of the text. One feature might indicate that a lending event occurs. Another might specify which participant fills which role. A language model’s hidden-state vector is another possible feature representation.

We fit the prediction model using some stories, then evaluate it on recordings from other, held-out material. The basic model learns weighted combinations of features. A constraint discourages excessively large weights, helping limit overfitting. This is the idea behind **ridge regression**.

The important comparisons are between descriptions: words and presentation information; concepts alone; concepts with event roles; concepts with reference and discourse relationships; and frozen language-model states. We also combine structured features with model states to test whether they contribute additional predictive information.

For example: **Does knowing who lent the book improve prediction beyond knowing that people, a book, and lending appear?** Does that information add anything when an LLM’s internal state is already available?

We measure how closely predicted responses match actual held-out recordings. The technical plan retains both correlation and prediction-error-based scores. Correlation measures whether predicted and observed responses vary together; the error-based score also tests how close the predicted values are.

A gain means that the added description helps predict neural responses under this model. It leaves open how the brain computes or represents that information internally.

### Semantic decoding: can the recorded pattern support the correct answer?

The direction is:

**Measured fMRI + a question → fitted decoder → predicted semantic answer.**

A **decoder** is the trained mapping that produces that answer. For a recorded passage like our example, questions could ask who lent the book, what “it” refers to, or whether the text states why the reminder occurred.

During training, the decoder sees recordings, questions, and correct answers. During testing, it receives the recording and question, together with any permitted answer choices. The source passage, correct graph, and answer are withheld from its input.

We train a separate decoder from frozen language-model states to the same semantic answers. The states are saved before introducing the question. The LLM therefore cannot reread the passage in response to each test question during this representation-accessibility experiment.

We compare simple readouts, small flexible neural readouts, and a structured decoder that combines concept and relation evidence according to the question. Comparable training effort and model capacity help reveal what the structural design contributes.

Scores distinguish identifying concepts from assigning roles, resolving references, and connecting events. Correct answers show that information is **accessible through the tested decoder**. They do not establish that the person performed the same question-answering procedure.

### Why keep both directions?

Encoding tests predictive association with measured activity. Decoding tests recovery of specified semantic information. A useful feature can help one measurement more than the other.

The mappings are fitted separately. They still use shared stories, annotations, and recordings, so agreement between them contains shared evidence. We account for that dependence when interpreting the results.

## 5. How we would build and compare concept maps

The additional geometry analysis asks which concepts or event configurations have similar representations within each system.

For “lending,” we estimate a numerical **signature** across many occurrences. We do the same for “returning,” “giving,” and other sufficiently represented concepts. We can also estimate signatures for a concept in a particular role or for a complete event configuration.

A signature is an estimate from multiple examples. Its meaning depends on how we construct it. The proposal retains four possibilities:

| Signature | What its numbers describe |
|---|---|
| **Grounding profile** | How much concept-related evidence the decoder assigns to each brain parcel or specified group of model features. |
| **Activity-pattern signature** | The measured brain pattern, or native model activity, associated with a concept after accounting for relevant context. |
| **Learned latent signature** | An internal numerical representation constructed by a trained readout. “Latent” means it is an intermediate representation inside that model. |
| **Encoding-implied signature** | A response pattern predicted by the encoding model for a specified concept or configuration. |

Grouping model features is an analysis choice; those groups have no assumed anatomical counterpart.

These are different research objects. Similar grounding profiles indicate similarly distributed decoding evidence. Similar activity patterns concern the measured or native representations themselves. An encoding-implied map remains a model prediction. We label each explicitly and keep the options available.

### What cosine similarity tells us

Cosine similarity compares the direction of two vectors. After accounting for vector length, it asks whether their relative patterns are similar. Repeating this comparison for every concept pair produces a square table, called a **similarity matrix**. A distance or dissimilarity matrix expresses the corresponding differences.

We build one table within the brain-derived representation and another within each model representation. The same concepts label their rows and columns. The vectors behind the tables can have different numbers of entries.

This lets us ask whether pairs that are relatively similar in the brain are also relatively similar in a model. We do not directly match a voxel number to a model-unit number.

We also retain comparisons between actual passages or events. These **stimulus-level** comparisons complement the concept maps and help show how the broader organization relates to particular observed examples.

### What “close” needs to mean

Lending and giving may resemble each other. Giving and receiving are closely related through their participant roles. One event causing another is a directed relationship. These are different relationships, and one symmetric distance table cannot fully describe all of them.

We therefore preserve roles and ordered event links. “Leah lends to Noah” and “Noah lends to Leah” must remain distinguishable. Likewise, keeping causal direction requires more than recording that two event vectors are similar.

Similarity in a vector space also differs from physical proximity in the brain. Similar signatures can be distributed across distant locations.

### Why the geometry needs independent checks

A learned readout can change its internal coordinates while preserving its answers. Some such changes alter cosine distances. Its geometry therefore depends partly on the measurement method.

Shared training can also create apparent agreement. We will fit brain and LLM readouts separately for the independent comparison, without a loss that explicitly forces their geometries to match.

The analysis will compare concept evidence before and after combination, alongside independently estimated activity patterns. For example, adding an argument vector into an event vector automatically contributes to their similarity. We must separate that built-in relationship from a relationship recovered from neural data.

## 6. The experiments that make the interpretation credible

### Does the representation preserve relationships?

We compare structured descriptions with controls that keep the same concepts while changing roles, reference links, or connections between events. A **control** is a comparison designed to test an alternative explanation.

We also hold out combinations: the training material contains the constituent concepts and relation types, while the tested combination is new to the fitted readout. This tests **compositional generalization**—using familiar parts in an unfamiliar arrangement.

Changing a question about a recorded passage is a valid experiment when the passage establishes its answer. Changing the passage itself creates a different stimulus. Without a recording for that edited stimulus, its human response remains unknown.

### Does the fMRI contribute information?

A decoder might guess from question wording or ordinary associations. We therefore compare it with a question-only system receiving the same legal answer choices.

We also disrupt the pairing between recordings and their associated text, while respecting the recordings’ temporal structure. Applying an already trained decoder to mismatched recordings tests its dependence on the correct input. Retraining with disrupted pairings tests how much semantic-looking organization the decoder design and labels can generate without the true neural correspondence.

A fixed concept-name vector could be perfectly consistent and agree with an LLM while ignoring fMRI entirely. Stability and semantic plausibility alone would miss that failure.

### Does the finding recur in different examples and people?

We estimate concept relationships using separate contexts, stories, participants, and trained fits. We check whether an apparent concept relationship follows shared vocabulary, topic, or recurring characters instead.

The target is appropriate stability: meaning-preserving variation should preserve relevant information, while a role or reference change should affect the corresponding distinction. We do not demand an identical map for every use of a concept.

### Are the annotations trustworthy?

An annotation model proposes structured records; independent annotation and human review check them. Ambiguous relationships remain marked as uncertain. A promise is kept separate from a completed action, and event order alone cannot establish causation.

For analysis at a particular point in the story, the annotation must use only information available by that point. A later revelation cannot silently determine the earlier label. The same input restriction applies when extracting model states.

Delayed fMRI windows can nevertheless include responses to later text. The analysis explicitly examines this temporal overlap; restricting the annotation’s context does not eliminate it.

### Are the tests genuinely held out and reliable?

Questions and repeated measurements from the same passage remain together when dividing training and evaluation data. Overlapping text or recording windows must not leak across those divisions.

We report uncertainty using participants, stories, and independent contexts. Generating hundreds of questions from one passage does not create hundreds of independent neural observations. Similarly, many cells in a concept-distance table share the same concepts.

Weak brain–model agreement needs to be interpreted alongside measurement reliability. Noisy or sparsely sampled signatures cannot justify a strong claim that the systems represent meaning differently.

## 7. The data and models in the proposed study

### Reading is the preferred setting

The technical proposal identifies three complementary reading resources:

| Resource | Role in the study |
|---|---|
| **Deniz reading condition** | Multi-story estimation. The documented subset contains six participants and eleven stories. Exact downloadable recordings and metadata still require verification. |
| **Wehbe reading dataset** | Independent narrative corroboration: eight participants reading one Harry Potter chapter. It supports within-chapter tests, with limited story diversity. |
| **Pereira2018** | Repeated sentence measurements across 627 sentences in short passages. Useful for sentence-scale comparisons; available individual repetitions must be checked. |

These descriptions and access caveats are carried from the technical proposal, §5. None of the datasets has been downloaded or analyzed for this project.

Reading aligns the presentation setting with the model’s text input. Visual presentation, word length, timing, and fMRI delay still require attention.

The larger LeBel story-listening resource remains a specified alternative when reading-data access or semantic coverage prevents important tests. Models would still receive transcripts; additional acoustic controls would account for aspects of the sound presentation. Adding a speech model is unnecessary for that alternative.

Dataset selection depends on access, reliability, and enough independent examples of the required semantic relationships. It should be settled without selecting for favorable scientific results.

### The model panel stays broad

The original proposal specifies twelve checkpoints, spanning GPT-2 XL, BERT, and Qwen, Llama, and Gemma families. It includes size comparisons and related checkpoints before and after additional training for instruction-following or other post-training objectives. Exact checkpoint names remain in the technical registry. [Technical proposal §9]

We examine several processing depths, called **layers**, and several amounts of preceding text, called **context windows**. These comparisons ask where information becomes accessible and how additional context changes it. They do not assign a human memory span or a cortical location to each model layer.

We retain the fixed panel and planned comparisons. Recent release date or strong language-task performance does not predetermine brain agreement.

## 8. What different outcomes would mean

| Possible finding | Interpretation supported by the study |
|---|---|
| Event roles improve both neural prediction and semantic decoding, with reproducible geometry. | Several measurement routes support sensitivity to role information. |
| Semantic decoding succeeds, while structured features add little beyond an LLM in neural prediction. | The LLM may already capture that predictive information; explicit labels help identify what is recoverable. |
| Broad concepts align across systems, while role or discourse relationships diverge reliably. | Agreement depends on the kind of meaning being compared. |
| A structured decoder performs well, while its maps change substantially across decoder designs or disrupted-data fits. | The method may be useful for decoding, with limited support for a stable neural-map interpretation. |
| A semantic distinction cannot be measured reliably. | The available experiment leaves that distinction unresolved. |

**The study can succeed through a well-supported agreement or a well-supported dissociation.** A dissociation is a difference between measurements that might otherwise have been expected to move together.

Neither a leaderboard win nor a single universal semantic map is required. We report encoding, decoding, geometry, and generalization together, including conflicting results.

The strongest existing-data conclusions concern recoverable information, predictive associations, and reproducible organization under specified measurements. Unique anatomical locations, biological necessity, and literal execution of our symbolic program require further evidence.

## 9. How the research would proceed

**Establish the usable data.** Obtain exact releases, verify presentation and scan timing, inspect anatomical information, and check overlaps between datasets. Record which stories, participants, and repeated measurements are actually available.

**Define and audit the meaning descriptions.** Specify concepts, roles, references, event relationships, and uncertainty. Annotate the stimuli, review the difficult cases, and measure coverage. Fix the semantic questions before examining their neural results.

**Prepare the model representations and evaluation divisions.** Resolve the full checkpoint panel, save stimulus-only model states, and divide stories or passages into fitting and evaluation sets. Keep future text, related questions, and overlapping recordings from creating unintended information access.

**Run encoding and decoding in parallel.** Fit their separate models and execute the planned comparisons. Retain every eligible condition and its failures. Activity-based geometry can proceed from the prepared recordings and states; grounding and encoding-implied geometry follow their respective fitted models.

**Compare organization and test alternatives.** Estimate the several signature types, compare their relationships within and across systems, and run structural, neural-dependence, annotation, and replication controls.

**Write the evidence-based account and release the work.** Report which semantic distinctions are supported, where results diverge, and what remains unresolved. Release permitted annotations, evaluation splits, configurations, code, and results so the comparisons can be repeated.

This is a dependency-based roadmap. The full experiment is planned in advance; implementation checks establish that the measurements work as specified.

## 10. Feasibility and the intended contribution

The study uses existing recordings, frozen language models, and smaller trained readouts. Its demanding parts are trustworthy discourse annotation, correct temporal alignment, sufficient semantic coverage, and separation of neural information from assumptions introduced by the method. Model-state extraction and the complete comparison panel also require substantial computation.

Coding agents can implement data handling, extraction, fitting, and reporting. Researchers must decide ambiguous semantic labels, approve the comparisons, and judge what the evidence supports.

The exact reading-data files, preprocessing details, checkpoint versions, and annotation coverage remain implementation decisions. The original proposal also records incomplete retrieval of two close follow-up papers; their full novelty comparison remains unresolved. Those limitations carry into this version.

For the intended NLP submission, the contribution would be a reusable way to compare **which meaning distinctions are available and how they are organized**, together with a reproducible finding that aggregate brain-prediction scores alone do not describe.

> **The project follows the same language through an explicit description of its meaning, human neural responses, and language-model activity. It uses prediction, semantic recovery, and representational relationships together to identify what the systems share and where they differ.**

---

## Source and technical-reference notes

This version condenses the existing proposal and preserves its unresolved items. It introduces no new literature search, dataset verification, or experimental results.

**Full reference:** [Structured Meaning in Human Brains and Language Models — technical proposal](Structured_Meaning_Brain_LLM_Research_Proposal.md).

For details, use the original’s **§§1–4** for questions and related work; **§§5–9** for datasets, semantic labels, and models; **§§10–12** for encoding, decoding, and geometry; **§§13–16** for controls and interpretation; and **§§20–23** for execution, feasibility, and outstanding decisions. Its reference registry supplies the sources behind the dataset descriptions and prior-work comparisons.

**Motivating papers:** Schrimpf et al. (2021), *The neural architecture of language: Integrative modeling converges on predictive processing*, especially pp. 2–3 and 8–9; Wang, Hsu, Adeli, and Wu (2026), *Neuro-Symbolic Decoding of Neural Activity*, especially pp. 4, 6–10, 18, and 20. These are R01 and R02 in the technical proposal. QA-Emb is R04; the related composition and role studies are R07–R11.
