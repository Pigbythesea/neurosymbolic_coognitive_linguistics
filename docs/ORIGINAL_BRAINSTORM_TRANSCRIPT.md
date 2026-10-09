# Original research brainstorm transcript

**Retrieved:** 2026-10-08. **Source title:** Evaluate Neurosymbolic fMRI Mapping.

**Source:** https://chatgpt.com/share/6ac18a1b-4f5c-83e9-b542-b85d56aa0512

This is a local transcript of the shared conversation, recovered by a direct HTTPS request after the web reader could not fetch it. It contains user messages and assistant final replies in their original order. Tool calls, tool results, and assistant analysis were excluded. Message numbers are positions in the shared page's linear conversation, so gaps are expected.

The message bodies below are preserved as supplied by the page. Original citation markers and sandbox download links may not resolve outside the source conversation. Statements about papers, deadlines, and resources are historical dialogue, not newly verified claims. The unavailable-custom-instructions placeholder is retained as a source artifact.

Source HTML SHA-256: 47bc6a047e1a8113bbee35de8d1141fb5f7cdc3046b6fb5f2dd603440e36500d

**Read the researcher's messages first:** [initial idea](#user-message-1), [concept maps and interpretive goals](#user-message-47), [parallel measurements and integration](#user-message-100), [plain-language request](#user-message-182), [annotations, datasets, and modern models](#user-message-209), [request for explanation](#user-message-220).

---

## User message 1

i have another research brainstorm idea after reading these papers, and im thinking to use this prepare for coling submission for 2027. the idea is this: the jiajun's paper builds upon the linear hypothesis of latent mapping from model to fmri signal, but the novelty is in 1 first defining symbolic extraction of meaning in the form of text from image (i think it uses gpt4o but verify), 2 train an embedding mapping fmri signal of each image and the symbolic text from each model, 3. use this idea as a new evaluation paradigm for how symbolic meaning from an image can be encoded/represented differently in brain. a key thing is this needs no actual fmri ground truth for each"concept"-they can just claim that the concepts may be encoded in brain in certain ways if the prediction is consistent or compositionally sound-because this is a new proposed evaluation paradigm, not accurate prediction of where each concept shows up in fmri. the shcrimpf paper gives a very good foundation on how early models like bert and gpt2 can predict lanaugage cortex activity based on mass dataset; therefore my idea is to fuse these two papers together: do a neurosymbolic extraction from some stimuli, preferrably some large text dataset, even hierarchically-for long text what concept is linked, and how these concepts are lthen linked in a higher level, etc. and train an embedding mapping text (or other stimuli, i thought of text only because i had this idea brainstorming for coling, if there's other more effective input absolutely suggest it) that maps the concepts to fmri so that we can test how these symbolic grounding maps language area (if text, maybe visual area for image/video, auditory area for sounds, or just full brain) and we can do latent evaluation of both open weighted llm and brain to see how certain concepts may be more similar (cos distance?) or closely located in the latent of human and model to add another layer of proof in terms of compositionality and consistency above jiajun's work. push back, evaluate and reason whether this idea is feasible, how can it be improved and is it implmenetationally doable, but above all whether it makes sense and worth trying or not.

## User message 5

Original custom instructions no longer available

## Assistant message 46

**Yes—there is a worthwhile project here, and it is implementationally feasible with existing fMRI datasets. But the strongest version is not simply “NEURONA applied to text, plus cosine similarity.”** It is a study of **which levels of structured meaning—concepts, role-bound events, and relationships across events—are recoverable from brain and language-model representations, and whether those representations organize meaning similarly beyond shared vocabulary and topic.**

Your intuition about not needing an anatomical ground-truth label for every concept is substantially right. The important distinction is between **validating a useful, structurally constrained decoding framework** and **establishing how the brain itself implements composition**. The first is achievable with existing data; the second requires stronger evidence than consistent decoder groundings.

One practical issue before the scientific assessment: **COLING 2027’s latest eligible ARR deadline is October 12, 2026**, with commitment on December 23. From today, September 30, that is 12 days—not several months because the conference is in 2027. I would separate this project’s scientific value from whether a defensible full study can be completed for that particular deadline. citeturn759634view0

## 1. What the two papers actually establish

### NEURONA: your interpretation is partly right, but the mapping direction matters

**The GPT-4o detail is verified:** the released BOLD5000 scene-graph generation script explicitly calls `model="gpt-4o"`. The paper itself describes a pretrained vision-language model without naming it in the dataset-construction section. Thus, the released code supports your recollection for BOLD5000; I would not infer that every video annotation used exactly the same configuration. citeturn142658view2 fileciteturn0file1L352-L365

However, its main pipeline is not a linear mapping from a model’s text embeddings to fMRI. It has two distinct paths:

**Dataset construction:**  
Image/video → scene graph → symbolic queries and answers.

**Neural decoding:**  
Observed fMRI + symbolic query → parcel embeddings → concept-grounding modules → compositional execution → predicted answer.

Figure 1 on page 4 makes this distinction particularly clear. The fMRI encoder produces candidate neural-region representations; learned modules score concepts over those representations; the executor combines the scores according to the query. Training uses the final QA answer, not intermediate region labels. Although the concept classifiers include linear layers, the complete system includes a convolutional encoder and compositional operations, so it is not equivalent to Schrimpf-style linear encoding. fileciteturn0file1L190-L211 fileciteturn0file1L343-L351 fileciteturn0file1L1264-L1285

There is also an important novelty detail in the appendix: **NEURONA already includes symbolic-query–fMRI embedding alignment and retrieval**, using compositional concept embeddings, an fMRI encoder, and a contrastive loss. Therefore, adding a shared embedding space would not, by itself, extend beyond that paper. fileciteturn0file1L927-L936

You are right that it does not require ground truth saying “holding is represented in these brain regions.” But the authors explicitly limit the interpretation: their groundings are **model-dependent decoding patterns**, not direct estimates of encoding representations, and they say the work does **not establish representational compositionality in neural activity**. That limitation is not an external objection I am imposing—it is their own distinction between what the experiments demonstrate and what remains hypothetical. fileciteturn0file1L519-L543

### Schrimpf: a foundation for prediction, not a complete account of meaning

Schrimpf et al. evaluate 43 models against three neural datasets, fitting model activations to actual, stimulus-matched neural recordings and testing predictions on held-out stimuli. Their contribution is integrative benchmarking across models, neural measurements, behavior, and language-task performance—not a new massive text–fMRI corpus. fileciteturn0file0L183-L194 fileciteturn0file0L233-L278

Two details matter for your proposal:

First, the near-ceiling results are dataset-specific and reliability-normalized. GPT-2 XL approaches the ceiling on the sentence datasets but achieves much lower normalized predictivity on the naturalistic story dataset. Second, **representational dissimilarity analysis without fitting is already included**. Consequently, “add latent similarity analysis” is useful methodology, but not a new evaluation paradigm on its own. fileciteturn0file0L279-L295

Interestingly, the discussion explicitly proposes moving toward semantic parsing and conceptual representations supporting grounded understanding and reasoning. Your direction therefore addresses a real limitation identified by that paper—but the intervening literature matters considerably. fileciteturn0file0L1070-L1083

## 2. The closest prior work changes where the novelty should be

The largest overlap is **not** simply prior work showing that language models predict fMRI. It is work already making semantic features interpretable or separating compositional information from individual-word information.

| Prior work | What was actually tested | What your project would need to add |
|---|---|---|
| **QA-Emb, Benara et al., 2024** | LLM answers to semantic yes/no questions become interpretable features. These features predict narrative fMRI through ridge regression, using three participants with extensive story-listening data. Its principal setup uses local preceding-word contexts. | Explicit bindings and relationships between events, rather than a collection of sentence/window properties. This is the closest precedent to “extract symbolic meaning, then map it to fMRI.” citeturn267195view1 |
| **Toneva, Mitchell & Wehbe, 2022** | Constructs representations of composed, “supra-word” meaning beyond individual-word meaning and examines their neural correspondence using fMRI and MEG. | Identify *which* semantic structure contributes: agent–patient bindings, reference across sentences, event dependencies—not merely contextual information beyond isolated words. citeturn759634search3 |
| **Kauf et al., 2024** | Uses the 627-sentence Pereira dataset and perturbs the inputs used to extract ANN representations. Lexical-semantic content contributes much more to ANN–brain similarity than the tested syntactic manipulations. | A relation-sensitive evaluation that cannot be satisfied by retaining the same content words or topic. Their result is not proof that the brain ignores structure. citeturn114080search0turn114080search1 |
| **Frankland & Greene, 2015** | Controlled sentence experiments distinguish who did what to whom and examine agent/patient information in temporal cortex. | Generalization to natural discourse, richer event structures, and matched comparisons with LLM representations. Basic role-sensitive neural coding is not itself a new finding. citeturn949421search0 |
| **Chen et al., CCN 2024** | Six participants complete over 1,000 relation-verification trials involving six semantic relation types, using the same 60 objects across relations. Encoding models test held-out relation instances while accounting for lexical and other features. | Context-dependent event composition and discourse structure, rather than mainly semantic relations between object concepts. This is also a useful example of a more controlled validation dataset. citeturn114080view3turn759634view2 |
| **Antonello et al., generative causal testing** | Generates verbal hypotheses from language encoding models and tests them through follow-up experiments with generated stimuli. | A different scientific target—structured semantic organization across brains and models—not merely automatically naming what a brain region might represent. citeturn907291view1 |

The QA-encoding line has also advanced beyond the original QA-Emb paper: the authors’ project description reports a 35-question model evaluated on both fMRI and ECoG. I verified that description, but could not retrieve the associated 2025 preprint’s complete text. Likewise, I located Chen et al.’s expanded 2026 semantic-relations preprint, but the detailed comparison above rests on their accessible 2024 report, not an assumed reading of the newer manuscript. citeturn114080view0turn114080view2

**My novelty judgment:** the general components are established, but the specific intersection you are aiming toward remains meaningful:

> **A structured, multilevel evaluation of semantic agreement between brains and LLMs, distinguishing concept overlap from role binding and cross-event organization.**

That is substantially stronger than claiming a new way to correlate semantic embeddings with fMRI. It is also more specific than saying “hierarchical meaning has not been studied,” which would be difficult to defend.

## 3. The version of your idea I would pursue

A suitable central question would be:

> **Do brains and language models preserve the same relational organization of meaning as comprehension progresses from individual concepts to events and connected discourse—and can conventional brain-predictivity scores distinguish models that do and do not preserve that organization?**

This preserves your original ambition. It does not reduce the project to finding one small condition under which a decoder works.

### Define “hierarchy” in terms of composition, not just abstraction

Three different ideas can otherwise get mixed together:

**Taxonomic hierarchy:** book → physical object → entity.

**Compositional hierarchy:** people and objects → events with assigned roles → relationships between events.

**Temporal hierarchy:** words → sentences → extended discourse.

They are related but not interchangeable. My recommendation is to make **compositional hierarchy the main object**, and test how it changes with temporal context.

For example:

> Leah lent Noah a book. Because Leah needed it back, she reminded him. He returned it.

A structured annotation could distinguish:

| Level | Information represented |
|---|---|
| Concepts | Leah, Noah, book, lending, reminding, returning. |
| Within-event structure | Leah is the lender; Noah is the recipient; the book is the transferred object. |
| Cross-sentence reference | “He” and “him” refer to Noah; “it” refers to the book. |
| Cross-event structure | The stated need motivates the reminder; the same book participates in the lending and returning events. |

The crucial comparison is not whether a representation recognizes that this passage concerns books and people. It is whether it preserves **which person occupies which role, which mentions refer to the same entity, and which events are connected**.

This also gives “compositional consistency” a concrete meaning. A representation should preserve relevant information under paraphrase while changing appropriately when roles or event links change. Simply placing semantically related words near one another does not test that.

### Why include both brains and LLMs?

The scientific payoff is a possible dissociation between:

**Overall predictive similarity:** a model’s features predict substantial neural variation.

**Structure-specific agreement:** the model and brain distinguish the same role assignments, referents, and event relationships.

An informative outcome could be that two models have comparable overall fMRI prediction but differ substantially in discourse-level agreement. Another could be that an explicit event representation explains brain–model correspondence primarily at one representational depth, while individual concepts explain correspondence elsewhere.

These are hypotheses, not results I would assume. But they offer a clearer contribution than another global ranking of “brain-like” models.

## 4. Which stimuli and datasets make this practical?

**I would choose language first—but likely spoken narratives with text transcripts, rather than restricting yourself to visually presented text.** The models can process the transcripts of the same stories heard by participants. You should keep the presentation modality explicit and include acoustic controls, rather than treating story-listening fMRI as pure text processing.

The essential constraint is:

> **Choose the neural dataset first, then annotate its stimuli. A large unpaired text corpus does not provide additional neural supervision.**

An external text corpus can help train or validate the semantic parser. It cannot substitute for measured responses to the stimuli used in the brain comparison.

### Recommended dataset roles

| Dataset | Verified resource | Best role in this project |
|---|---|---|
| **LeBel et al., 2023** | Eight participants, each listening to 27 natural narratives, approximately six hours; raw and preprocessed data, cortical surfaces, and encoding-model code. | **Primary dataset:** substantial within-person data for fitting structured encoding and decoding models. citeturn949421view3turn759634view1 |
| **Narratives, Nastase et al., 2021** | The published release contains 345 participants, 891 scans, and 27 stories totaling approximately 4.6 hours of unique material, with timed transcripts and preprocessed data. The collection is heterogeneous—not every participant heard every story. | **Independent replication and broader participant coverage**, using a prespecified subset with suitable overlap and data quality. citeturn952621search1turn275260view2 |
| **Pereira2018** | Two sentence experiments totaling 627 sentences, with repeated presentations; short passages rather than long narratives. | **Sentence-level complementary evidence**, particularly for reliable representational comparisons. Not sufficient as the sole dataset for your discourse-hierarchy claim. fileciteturn0file0L1060-L1063 fileciteturn0file0L1122-L1125 |

My preferred package is **one deep narrative dataset, one independent narrative replication, and a sentence-level complementary analysis where it genuinely answers a different question**. These are complementary experiments, not a sequence of tiny pilots.

For a more controlled relational test, the Chen relation-verification paradigm is attractive. However, I have not verified an accessible release of the necessary trial-level data, so I would not make your project dependent on obtaining it.

### Would images or video be better?

Video is attractive for relationships that are directly observable: actions, spatial relations, and event sequences. But you would then need to separate language, visual content, motion, and other correlated information. NEURONA already uses Friends-derived video–fMRI data, so moving into video alone would not create a strong distinction. fileciteturn0file1L372-L378

For the question you are currently articulating, **narratives are the better primary choice**: reference, role binding, and links between propositions can be specified directly from the linguistic input. I would add multimodality only when it tests whether a particular semantic structure generalizes across modalities—not merely to broaden the paper.

## 5. A concrete implementation that keeps the evidence independent

I would organize the project around **three linked analyses**, with the same semantic annotations and stimulus splits but separately trained mappings.

### A. Construct a shared, explicit semantic reference

For every stimulus segment, create a graph containing entities, events, argument roles, reference links, and selected discourse relationships.

Use an LLM to produce annotations, but make the output constrained and inspectable: typed nodes, typed edges, supporting text spans, and an uncertainty indicator. Separate explicitly stated relationships from inferred ones. For example, “after” must not silently become “caused by.”

For time-resolved narrative analysis, annotations should reflect **the prefix available at that moment**. A parser given the entire story could resolve a pronoun or identify a motive using information the listener had not yet received. That would confound a claim about online semantic construction.

I would use a **common reference annotation across all evaluated models**. Otherwise, differences between models could arise because each model extracts a different graph, rather than because its internal representations differ. Model-generated graph quality can be a separate behavioral outcome.

Human checking should focus on the annotations carrying the scientific claim: role assignments, reference resolution, and event links. You do not need manually annotated anatomical concept maps, but you do need evidence that your purported semantic distinctions are actually present in the stimuli.

Finally, preserve binding in the numerical representation. An unordered sum of “Leah,” “Noah,” and “lend” cannot distinguish who lent to whom. Use role-aware features or an edge-aware graph encoder, with explicit controls that remove roles or scramble links while preserving the concepts.

### B. Test what structured meaning explains in actual fMRI

This is the direct extension of the encoding approach you already know.

Construct feature groups for:

- Low-level and linguistic controls: acoustic features for listening, word rate, lexical properties, and relevant temporal/context controls.
- Flat semantic content: concepts without their bindings.
- Within-event structure: predicates and assigned argument roles.
- Discourse structure: reference and relationships across events.
- Frozen LLM representations, analyzed separately by model and depth.

Fit subject-specific encoding models with appropriate hemodynamic delays. **Banded ridge regression** is a suitable implementation because it allows different feature groups to receive different regularization strengths; the associated Himalaya implementation was developed for precisely this kind of multifeature voxelwise modeling. citeturn759634search2

A central comparison would be:

\[
\Delta R^2_{\text{structure}}
=
R^2(\text{controls + concepts + structure})
-
R^2(\text{controls + concepts}).
\]

Then examine what happens when frozen LLM features are included.

This answers two distinct questions:

**Does explicit structure explain held-out neural responses beyond concept content?**

**Is that explanatory information already captured by a particular LLM representation, or does the structured representation add complementary information?**

Do not make “structure must improve over every LLM” a requirement. The LLM might already encode the relevant information. In that case, the important result could be identifying the structured information shared by its representation and the brain.

Conversely, a positive improvement is not automatically evidence for a special symbolic neural mechanism. Better annotations, a useful inductive bias, or better regularization could explain the gain. This is why structure-destroying controls and the next analyses matter.

### C. Evaluate brain and LLM representations on the same structured queries

This is where NEURONA’s contribution becomes useful rather than merely something to imitate.

Train separate readouts from:

**Observed fMRI → structured semantic answers**

and

**Frozen LLM activations → the same structured semantic answers.**

Use comparable readout capacity, transparent input adapters, and the same train/test partitions. Compare an explicit compositional executor against a noncompositional decoder with comparable access to the inputs.

The questions should cover a hierarchy: entity presence, role assignment, cross-sentence reference, and selected event relationships. Report performance separately at each level rather than letting numerous easy entity-presence questions dominate the overall score.

A particularly important restriction: **extract the LLM activations from the stimulus before presenting the evaluation query**. Allowing the LLM to reread the passage and reason in response to each query measures prompted task solving, not information accessible from its original stimulus representation. Prompted QA can still be a separate behavioral comparison.

This produces a useful comparison:

> Does an explicit structural readout help recover similar kinds of information from brain and model representations, or does it compensate for different representational deficiencies in each?

NEURONA already tests argument swapping, predicate transfer, and role systematicity. Your advance would therefore need to include the **matched brain–LLM comparison and genuinely discourse-level structure**, not just reuse those names for equivalent tests. fileciteturn0file1L966-L1002

## 6. How to make your latent-geometry idea informative

Your geometry proposal is valuable—but **the spaces should not first be trained to agree and then have that agreement presented as independent confirmation**.

### Compare relationships between stimuli, not raw coordinates across systems

For the same held-out passages or event windows, construct:

\[
D_{\text{brain}},\qquad
D_{\text{LLM}},\qquad
D_{\text{semantic structure}},
\]

where each matrix describes the pairwise dissimilarity between the same stimuli.

You can compare these matrices even when the brain representation has thousands of voxels and the model representation has a different number of dimensions. This is the basic rationale behind representational similarity analysis; it is already present in Schrimpf’s evaluation. fileciteturn0file0L279-L284

The interesting analysis is then not just:

> Are the brain and model distance matrices correlated?

It is:

> **Does their agreement track relational structure after accounting for concept overlap, lexical similarity, topic, and context?**

For example, do patterns group events according to who acted on whom, or mainly according to the nouns present? Do cross-sentence representations preserve an entity’s identity when its surface expression changes from a name to a pronoun?

For repeated neural measurements, noise-aware, cross-validated distances can strengthen the analysis. Without suitable independent repetitions, use an appropriate alternative and report reliability limits rather than treating every estimated distance as equally trustworthy. citeturn431684search1

### Three distinctions are essential

**Semantic proximity is not anatomical proximity.** Two concepts may evoke similar distributed patterns without occupying neighboring cortical locations. Your geometry analysis addresses the former; spatial maps address a different question.

**Readout geometry is not native geometry.** A learned linear transformation can stretch or compress distances while retaining predictive information. An aligned latent space can therefore be useful for retrieval without proving that the original brain and model spaces had the same organization.

**Similarity is not compositionality.** A representation can place “doctor,” “nurse,” and “hospital” near one another while failing to distinguish “the doctor thanked the nurse” from its role reversal. The test must evaluate structure-sensitive distinctions and appropriate invariances.

These distinctions do not weaken your idea. They specify what its additional evidence should actually establish.

### You do not need a brain embedding for every isolated concept

A useful consequence is that you can avoid an unnecessarily difficult intermediate problem: estimating a unique neural vector for “lending,” “belief,” or “causation.”

Instead, compare **measured neural patterns for actual stimulus instances** and ask which semantic descriptions explain their relationships. Concept-specific maps can be secondary, model-dependent summaries.

That keeps the analysis anchored to observed neural data while retaining your goal of studying abstract organization.

## 7. What would make the conclusions convincing?

The most consequential issues are not small implementation sensitivities. They are whether the experiments distinguish your intended explanation from easier alternatives.

### Compositional generalization must separate structure from familiarity

A held-out query is not necessarily a held-out semantic situation. A decoder can benefit from already seeing the same stimulus, overlapping windows, or near-identical combinations.

Split by story or passage first. Keep every query and every repetition associated with a stimulus on the same side of the split. Then define composition-specific holdouts within that framework: familiar constituents appearing in new combinations, new reference configurations, or new combinations of event links.

The availability of such cases must be measured. A corpus with many words is not automatically a corpus with enough repeated relational contrasts.

### The fMRI must contribute information beyond the query

For the neural QA task, compare against query-only prediction and appropriately shuffled or mismatched fMRI. Use difficult negatives that preserve the same entities while changing a role, referent, or relation—not only negatives involving an unrelated concept.

Otherwise, “person–bicycle → riding” can be answered from ordinary semantic expectations, without identifying what the participant actually encountered.

### Consistency needs specificity and neural dependence

NEURONA’s consistency score rewards repeated use of similar region sets for a concept. But a model that always assigns every instance of a concept to the same region can be perfectly consistent even when that assignment is uninformative. This follows directly from the metric’s definition. fileciteturn0file1L473-L518

For your study, useful consistency would mean that groundings are stable **where the semantic hypothesis predicts stability**, sensitive **where the binding changes**, and supported by held-out neural prediction. Region-occupancy-matched nulls are more informative than only uniformly random assignments.

Likewise, decoder coefficients should not be equated with where a concept is encoded. The interpretation of multivariate decoding weights is a longstanding issue; predictive weights and neural activation patterns need not coincide. citeturn431684search0

### Synthetic semantic edits are useful, but do not create neural observations

You can change a graph or sentence computationally and test whether the changed representation predicts the **original** recorded response less well. That tests whether the original semantic structure was useful to the model.

It does not reveal how the person’s brain would have responded to the edited sentence. Without recordings of the edited stimulus, that counterfactual neural response remains a prediction.

This is fully compatible with a strong existing-data paper. It simply prevents model-generated predictions from becoming their own validation.

### Keep uncertainty at the right level

Thousands of generated questions about a few stories are not thousands of independent neural experiments. Statistical inference should respect participants, stories, repeated stimuli, and temporal dependence.

Likewise, many layers from the same LLM are not independent models. Model-family and layer comparisons should not artificially inflate the evidence for a universal relationship.

## 8. Is it worth trying, and what would the paper contribute?

### Scientifically: yes, with the revised target

I would pursue the following package:

**A reusable semantic resource:** stimulus-aligned, multilevel event and discourse annotations, with audited structure and explicit generalization splits.

**A matched evaluation framework:** the same semantic distinctions tested in observed neural responses and frozen LLM representations, separating decoding accessibility, encoding predictivity, and representational geometry.

**A substantive finding:** evidence about where brain–model agreement preserves relational organization, where it reduces to concept content, and whether those conclusions change across representational depth and discourse scale.

The paper should not need to claim that brains literally execute your symbolic graph. A defensible conclusion could be:

> “Role-bound and cross-event semantic structure explains reproducible aspects of neural and model representations that are obscured by aggregate brain-predictivity scores.”

That is a meaningful scientific and NLP contribution—provided the results distinguish it from flat semantic features and generic contextual embeddings.

### Implementationally: well within reach, but annotation and identifiability are the hard parts

Given your experience with frozen-representation fMRI readouts and representational comparisons, much of the numerical infrastructure is familiar. Public narrative data and established multifeature encoding implementations exist. citeturn759634view1turn759634search2

I would budget for a prespecified panel of roughly 8–12 frozen models, spanning established reference models and several open-weight model variants, rather than training a new large language model. The graph machinery and readouts can remain modest.

The genuinely difficult work is producing trustworthy discourse annotations, aligning them to the evolving stimulus, establishing sufficient compositional coverage, and separating relational information from correlated lexical and contextual information. Those are central research tasks, not problems an extremely capable coding agent can eliminate by generating more code.

### For COLING 2027 specifically: the topic fits; the current deadline is the problem

COLING’s call includes computational cognitive modeling, discourse, interpretation, grounding, and NLP-based scientific study of language, so the substantive fit is good. But the full project described here is not a credible from-scratch 12-day commitment unless substantial data preparation, annotations, and results already exist. citeturn157735view4turn759634view0

**My recommendation is to pursue the research idea, but not its literal initial formulation.** “Extract semantic text, align it to fMRI, and inspect cosine distances” overlaps heavily with existing work. **Testing the multilevel relational organization shared—and not shared—by human neural responses and LLM representations is the version worth investing in.**

Your strongest intuition is that symbolic structure can make brain–model comparisons more scientifically interpretable without requiring an anatomical label for every concept. The improvement is to turn that structure into **specific, independently testable distinctions**, rather than treating consistency or a learned shared embedding as sufficient evidence on its own.

## User message 47

your revised version of idea and central question sound very promising and i would approve your novelty analysis. but im not sure about predicativity scores; from my understanding, in the jiajun's work, they do not necessarily have a prediction accuracy, but use that framework to establish how a symbolic concept can be encoded in the brain, and show this prediction not by examining with a ground truth (thus the score part is absent) but with the internal consistency and compositionality to show that it makes sense. im not sure should we do this this way; the cosine similarity, or distance in latent space idea i said, is after we have this prediction (so that for each concept we can build a map on how they are close in terms of latent to each other concept, build this pairwise relationship within brain, and we can have the same for openweighted llm which are new after the schimpf paper where the models weren't that developed at that time) so that compared to jiajun's work we can have more confidence saying this framwork works (a criticism for that work is that it's like drawing the target after shooting the arrow-propose a framework without ground truth, and since question is predifined it's interpretaion is basically what they claim to be). however, in my ideation, the criticism is not necessarily a criticism-it can be a shortcut for our work since we are not beating a leaderboard on accuracy, and such leaderboard don't even exist yet so we are freer to make interpretative claims.  i would agree on using language as the input; for your concrete implementation plan, i think that makes very much sense, but im not sure if introducing acoustic element for dataset would add another layer of complication or should we use text only. explain to me further and push back. for the things i did not comment on, that means i approve of them; this is the final round of back and forth before i ask you to generate a full implementation level plan. report to me first

## Assistant message 99

**I agree with putting learned concept maps and their pairwise organization at the center of this project. My previous answer gave conventional fMRI predictivity too much prominence for the paper you want to write.** We can make the principal contribution an interpretable framework for studying semantic organization, without making superior voxelwise prediction the main claim.

There are three things to settle: what NEURONA actually validates, what your concept–concept geometry would add, and whether reading-only data can support the project. **My recommendation is a reading-first study, with concept grounding and relational geometry as the main outputs, held-out semantic decoding as a validation requirement, and conventional neural predictivity as supporting evidence.**

## 1. NEURONA does have prediction accuracy; the missing ground truth concerns its intermediate maps

Your understanding is correct about **the absence of anatomical ground truth for individual concepts**. However, the conclusion that its prediction-accuracy component is absent needs correcting.

NEURONA has three different objects that should be kept separate:

| Object | What is available? | What can be evaluated directly? |
|---|---|---|
| The participant’s fMRI response to a stimulus | An actual measured recording. | Predictions of that recording could be evaluated through an encoding model, although this is not NEURONA’s main task. |
| Semantic answers about the stimulus | Answers derived from the stimulus’s automatically generated scene graph. | Whether a decoder given fMRI and a question produces the annotated answer. |
| The brain regions assigned to a concept | No independently labeled correct regional assignment. | Grounding consistency, generalization, sensitivity to neural data, and other indirect evidence. |

Section 3.4, on page 6, explicitly says that training uses ground-truth answers and a cross-entropy loss, while intermediate concept groundings receive no supervision. Those answer labels are generated from stimulus annotations; they are not reports of what the participant consciously represented. fileciteturn5file1L343-L365

**Table 1 on page 7 reports overall QA accuracies of 70.41% on BOLD5000-QA and 70.46% on CNeuroMod-QA.** Table 2 reports accuracy on unseen compositions. The paper therefore combines an objectively scored decoding task with more interpretive analyses of its intermediate representations. fileciteturn6file0L14-L38

For a language example, suppose the stimulus says:

> Leah lent Noah a book. He returned it the next day.

We know from the text that the answer to “Who returned the book?” is Noah. A decoder can be tested on that answer using the corresponding fMRI. We do not need to know which voxels encode Noah, returning, or the book.

During training, the decoder might learn an intermediate assignment suggesting that particular distributed neural signals help identify the returning event. **That intermediate assignment remains a hypothesis about the information used by the decoder.** Its usefulness for answering held-out questions provides one constraint; its repeatability provides another.

This is also how the authors qualify their findings. They describe their maps as model-dependent decoding patterns and explicitly state that NEURONA does not establish representational compositionality in neural activity. fileciteturn5file1L519-L543

### What this changes for our project

We should distinguish two meanings of “predictivity”:

**Semantic decoding performance:** Can measured brain activity support correct answers about concepts and their relationships?

**Neural encoding performance:** Can semantic features or LLM activations predict measured fMRI responses?

Schrimpf’s principal neural score is the second: held-out correlation between predicted and measured neural responses. fileciteturn5file0L256-L275 NEURONA’s principal accuracy is the first.

**We should retain semantic decoding performance. We can give conventional neural encoding performance a secondary role.** That resolves much of the apparent disagreement.

There is no need to require that your symbolic framework outperform every LLM at predicting the entire fMRI signal. A framework can provide useful, reproducible distinctions between semantic structures even when a less interpretable representation achieves higher aggregate prediction.

## 2. The absence of concept-localization labels offers a real efficiency advantage—with a specific trade-off

I agree with the practical opportunity you identify. We can use existing recordings, annotate their stimuli, and investigate candidate semantic organizations without collecting a separate experiment to establish the correct cortical location of every concept.

That is a substantial advantage. It allows the project to address a broad semantic space and to develop a useful scientific instrument.

Where I would push back is the idea that the absence of a leaderboard makes interpretive claims easier to justify. **It gives us freedom to define the evaluation target; it does not supply evidence for a particular interpretation.**

There are two defensible contributions here.

**A framework contribution:** “This procedure learns reproducible, concept-conditioned groundings that support structured semantic decoding.”

**An empirical representational contribution:** “These groundings reveal particular relationships among concepts that recur across independent neural data and agree, or disagree, with corresponding relationships in LLM representations.”

Both can be valuable without recovering the brain’s unique, complete semantic code. The second requires evidence beyond the framework producing internally coherent outputs.

### Why internal coherence alone cannot carry the interpretation

Consider a deliberately defective system. For every concept, it returns a fixed vector derived from the concept’s name. It ignores the fMRI.

Such a system could produce stable maps, sensible clusters of related concepts, and high agreement with an LLM’s semantic geometry. Its concept representations might remain perfectly consistent across every stimulus. None of those properties would establish that the maps contain information extracted from the neural recordings.

This is an illustrative failure case, rather than an accusation about NEURONA. It explains why **neural dependence must be demonstrated alongside semantic plausibility**.

The scientifically useful shortcut is therefore:

> **Use weak supervision to infer candidate concept representations, then test whether their organization depends reproducibly on the measured neural data.**

That is still much easier than obtaining anatomical ground truth for every concept. It also leaves ample room for an interpretation-focused paper.

## 3. Your concept–concept geometry is a legitimate central analysis

I now understand your intended sequence more precisely:

> Learn a neural signature for each concept → calculate relationships among those signatures → independently obtain corresponding signatures from LLMs → compare the relationships within the two systems.

**Yes, we should preserve that sequence.** My previous emphasis on stimulus–stimulus similarity did not fully capture your intended concept-level analysis.

For example, after fitting the framework, we might obtain a brain-derived signature \(u_c^{B}\) and an LLM-derived signature \(u_c^{M,\ell}\) for concept \(c\), with \(\ell\) denoting the model layer. Within each system, we could calculate cosine dissimilarities:

\[
D_B(c_i,c_j)
=
1-\cos(u_{c_i}^{B},u_{c_j}^{B}),
\]

\[
D_{M,\ell}(c_i,c_j)
=
1-\cos(u_{c_i}^{M,\ell},u_{c_j}^{M,\ell}).
\]

Both matrices have concepts on their rows and columns. Their underlying vectors can have different dimensions. We compare the matrices’ relationships, rather than directly taking a cosine between a brain vector and an LLM vector.

This would let us ask whether, for example, the relative organization of **giving, receiving, lending, returning, requesting, and reminding** differs between brain-derived representations and different model layers.

The crucial design decision is **what counts as a concept signature**.

### A. A grounding profile answers a question about decoding support

A NEURONA-like signature could contain the evidence assigned to each parcel when decoding a concept:

\[
u_c^{B}
=
[\text{evidence in parcel 1},\ldots,\text{evidence in parcel }P].
\]

Similarity between these vectors tells us that two concepts receive similar distributions of decoding evidence across parcels. NEURONA’s unary grounding scores have this general form. fileciteturn5file1L213-L227

That is a useful object. We can visualize it anatomically and study its organization.

Its interpretation should remain precise: two similar grounding profiles suggest similar **regional support under the readout**. They do not automatically establish similar fine-grained neural activity patterns. Several concepts could draw information from the same broad network while being distinguished by different patterns within it.

### B. A concept-conditioned activity pattern answers a question about representational geometry

A complementary signature can be estimated from the actual activity associated with a concept across held-out occurrences, using matched comparisons or statistical controls to reduce correlated context.

For the brain, the entries would be measured neural features. For an LLM, they would be native hidden-state features elicited by the corresponding stimulus occurrences.

This makes the two sides more directly comparable: both summarize concept-conditioned activity. The estimates still depend on sampling and analysis choices, and naturalistic data will not perfectly isolate a concept from everything accompanying it.

**I would keep your learned grounding maps as the principal framework output and add this activity-based view as corroborating evidence.** The two views serve different purposes:

| View | Main question |
|---|---|
| Learned grounding geometry | Which concepts rely on similar distributions of evidence under the framework? |
| Concept-conditioned activity geometry | Which concepts are associated with similar patterns in the measured brain responses and native model activations? |

Agreement across these views would increase confidence. Disagreement would identify where the readout changes or compresses the apparent semantic organization.

### C. An unconstrained learned latent requires particular caution

Suppose a decoder computes scores from an internal representation \(h\) through a linear readout \(W\). For an invertible transformation \(A\), we can write:

\[
h'=Ah,\qquad W'=WA^{-1}.
\]

The output scores remain identical:

\[
W'h'=Wh.
\]

An arbitrary \(A\) can nevertheless change cosine similarities between latent vectors.

This mathematical example shows that **decoding performance alone does not uniquely determine a latent geometry**. Architecture and regularization choose among possible geometries.

We can still analyze a learned latent. We should define it as a representation produced by a specified measurement procedure and test whether the conclusions survive reasonable changes to that procedure. Fixed-coordinate grounding profiles and independently measured activity patterns provide useful anchors.

One further restriction follows directly: **the “human” distance matrix should not consist solely of fMRI patterns predicted by the same LLM being evaluated.** That would mainly characterize the fitted prediction model. Measured neural responses need to remain part of the evidence.

## 4. What would make the added geometry genuinely stronger evidence?

### Separate relationships supplied by the architecture from relationships recovered from data

There is a particularly relevant detail in NEURONA.

Appendix D.4 already calculates correlations between subject, object, and predicate grounding vectors. It reports greater correlation between guided predicates and their arguments. fileciteturn6file1L91-L105 Meanwhile, Appendix G’s full-guidance formulation explicitly constructs the guided predicate score by adding predicate, subject, and object grounding vectors. fileciteturn5file1L1343-L1348

This matters for interpreting the correlation. In a simplified example,

\[
G_{\mathrm{guided}}=G_{\mathrm{predicate}}+G_{\mathrm{subject}}+G_{\mathrm{object}}.
\]

Even independent, zero-mean components with equal variance produce positive correlation between the sum and each component. The construction itself contributes to the relationship.

**That does not erase the paper’s decoding results. It limits how independently the correlation supports the grounding interpretation.** For our study, the added geometry should examine relationships whose outcome is not already substantially specified by the composition rule.

For example, the framework can define which entity occupies the agent role. It should leave open whether **agent-bound instances of lending** have similar neural geometry to **agent-bound instances of giving**, and whether that relationship changes across model layers.

This is also a small qualification to the novelty assessment we already agreed on: computing correlations among learned groundings has a precedent within NEURONA itself. Our meaningful extension would be the **systematic, multilevel concept geometry, its comparison with LLM representations, and its independent neural validation**.

### Treat LLM agreement as an empirical comparison, rather than a correctness oracle

A contemporary open-weight LLM gives us another representational system to examine. Its agreement with a neural map can be informative.

However, agreement can arise from the shared linguistic environment, the same stimulus distribution, the same annotation scheme, or the same readout constraints. It does not uniquely establish that the neural interpretation is correct.

I would therefore avoid making “modern models agree with our maps” the framework’s success criterion. A sound framework could reveal that some models reproduce concept-level relationships well while diverging at event or discourse levels. Those divergences could become the paper’s most interesting findings.

The comparison becomes more informative when the brain and LLM mappings are fitted separately, the principal comparison has no brain–LLM geometry-matching loss, and the conclusions are evaluated on held-out material. Shared concept labels remain useful anchors; controls establish how much of the agreement comes from those anchors alone.

### Retain context and direction when constructing concept representations

A single average vector for each concept is a useful starting summary. Your approved hierarchical question requires additional structure.

Consider:

> Leah lent Noah the book.

> Noah lent Leah the book.

An inventory containing Leah, Noah, book, and lending is identical for both sentences. Averaging their constituent vectors without retaining roles will obscure the distinction we want to study.

Likewise, “event A caused event B” contains a directional relationship. A symmetric cosine matrix cannot express that direction by itself.

I would therefore retain concept signatures at several levels: concept types, role-conditioned concepts, event configurations, and relationships between events. The original flat concept map can remain a summary view.

This also clarifies what “close” means. Semantic similarity, association, shared argument roles, and causal relatedness are different relations. “Giving” and “receiving” might be strongly associated while involving complementary participant roles. We should test these relationships separately, rather than expecting one universal distance matrix to capture them all.

### The essential validation package can remain compact

For the eventual plan, I would organize validation around three substantive questions:

**Does the neural recording matter?** Compare against query-only or annotation-prior baselines, and repeat the analysis with appropriately disrupted stimulus–brain pairing while preserving the relevant temporal structure.

**Does the concept organization recur?** Estimate it across independent stories, participants, and training fits. Check that agreement survives changes in wording, co-occurring entities, and topic.

**Does the relational structure explain anything beyond the constituent concepts?** Compare role-aware and discourse-aware representations with controls that preserve concepts while removing or changing their relationships.

These tests do not turn the project into a leaderboard competition. They establish whether its central object—the learned semantic organization—is meaningfully constrained by neural observations.

We should also quantify uncertainty in the distances. A large matrix contains many entries that share the same concepts and observations; those entries should not be treated as independent experimental replications.

## 5. Reading-only is viable, and I would revise the dataset preference accordingly

Your concern about auditory complications is reasonable. There are two separate choices:

**What the LLM receives:** This can remain text throughout the project.

**What the participant received during scanning:** Written text or spoken language.

Using story-listening fMRI does not require a speech model. A text model can process the transcript, while a small set of acoustic and timing features is included as a statistical control. The LeBel resource supplies aligned transcripts and established feature spaces for this purpose, including word rate, phoneme rate, and articulatory features. citeturn616676view5

Nevertheless, reading-only gives the cleanest match to your preferred input setting, provided the dataset contains enough varied discourse.

### A stronger reading-only option was missing from my previous answer

**The Deniz et al. reading/listening resource deserves serious consideration as the primary dataset’s source, using its reading condition alone.**

A published analysis of a public subset documents **six participants and eleven narrative stories**, with every participant reading and listening to the same material. The reading condition presents words visually, one at a time, with presentation durations matched to the spoken stories. This is connected narrative reading, rather than a collection of isolated sentences. citeturn730667view1

The authors’ related code repository also explicitly links the BOLD resource. I verified that documentation, although the underlying Box and GIN data hosts did not render successfully here. Consequently, **the exact downloadable files, participant coverage, and metadata completeness still need file-level verification before we hard-code the dataset in the implementation plan**. fileciteturn7file0L2-L5

That is an access-verification limitation, rather than evidence that the reading data are unavailable.

Here is how I would now evaluate the main options:

| Dataset option | What it offers | Role I would give it |
|---|---|---|
| **Deniz reading condition** | Multiple connected narratives; a published reuse documents a six-participant, eleven-story subset. | **Best primary reading-only candidate**, pending verification of the actual release and semantic coverage. citeturn730667view1 |
| **Wehbe Harry Potter reading dataset** | Eight participants reading one chapter, with words displayed for 0.5 seconds each; the authors provide data and story-feature download routes. | Independent reading corroboration, especially for recurring characters and reference. One chapter provides limited story-level diversity. citeturn654312view3 |
| **Pereira2018** | 627 sentences organized into short passages, with repeated sentence presentations. | Reliable sentence-level complementary analyses; limited coverage of extended discourse. fileciteturn5file0L1060-L1063 fileciteturn5file0L1122-L1125 |
| **LeBel story listening** | Eight participants, 27 narratives, approximately six hours per participant, with raw and preprocessed recordings. | A strong option when broader within-person semantic coverage is necessary. citeturn897817view1 |

**My revised preference is reading-first, with Deniz’s reading condition as the primary candidate and Harry Potter as independent reading corroboration.** Pereira can support particular sentence-scale analyses where repetition improves their reliability.

### What reading-only simplifies—and what remains

Reading-only removes the need to account for the physical sound signal when interpreting the main dataset. It also makes the shared input easier to describe: humans read the text; models process the corresponding text.

It still requires accounting for visual presentation, word length, exposure duration, and timing. Empirical work with the reading/listening resource shows that correlated low-level features can contribute substantially to apparent brain–model alignment, including alignment with sensory regions. citeturn616676view0

It also retains the temporal limitations of fMRI. Displaying one word at a time does not yield an isolated, instantaneous fMRI measurement for that word. Our event and discourse annotations must still be aligned to delayed, temporally overlapping responses. The existing naturalistic encoding methodology explicitly accounts for this through downsampling and hemodynamic delays. citeturn616676view4

These are manageable requirements. Neither reading nor listening gives us a measurement of semantics with every other influence removed.

### My pushback on making reading an absolute constraint

The decisive issue is whether the data cover enough examples of the semantic relationships we want to study.

A small reading corpus could contain thousands of words yet very few independent instances of particular reference configurations or event relationships. Generating more questions about those same occurrences would not create additional neural evidence.

Therefore, I would use reading-only **as a preferred design choice rather than an inviolable requirement**. If the verified reading resources provide sufficient concept and relational coverage, there is no reason to add an auditory component to the principal study. If their coverage is inadequate, the larger listening resource could be more valuable than the modest simplification obtained by excluding sound.

The existing evidence for substantial shared semantic selectivity between listening and reading supports using either modality for language-semantic research. It does not settle whether every fine-grained event or discourse relationship will behave identically across them. citeturn637729search3

## 6. The decisions I would carry into the full plan

**The main object should be a multilevel map of concept organization.** It should retain individual concepts, role-bound events, and discourse relationships, with explicit definitions of the signatures whose similarities we compare.

**Semantic decoding should validate the framework.** We need evidence that its readouts recover stimulus-grounded information from held-out neural recordings. We do not need a new state-of-the-art result on aggregate voxelwise prediction.

**Grounding geometry and activity geometry should provide complementary evidence.** Learned maps are legitimate framework outputs. Independent measured-response analyses help establish how much of their organization is supported by the recordings.

**Brain–LLM agreement should remain an outcome that can vary.** Agreement at one semantic level and divergence at another would both be informative. The analysis should accommodate either result without changing its success criteria afterward.

**The main dataset should be reading-first.** Deniz’s reading condition is the strongest candidate identified in this round; its exact release and usable semantic coverage remain to be verified. An acoustic model is unnecessary for the proposed core study.

I would now formulate the central question as:

> **What multilevel organization of concepts, role-bound events, and discourse relationships can be recovered reproducibly from human neural responses, and which aspects of that organization are shared with—or differ from—language-model representations?**

That keeps your interpretive ambition intact. The lack of anatomical concept labels becomes a reason to use weak supervision and converging evidence. **The study’s strength would come from establishing a reproducible semantic organization with clear empirical constraints, while allowing the detailed brain–model relationships to emerge from the results.**

## User message 100

yes-that's what im also thinking about. we can have semantic decoding performance and nueral encoding performance as 2 independent measurement, do in parallel instead of presetting a "primary/secondary" tier; your feedback on my additional ideas are very explanatory. your already proposed project shape itself is also very established so consider integrating instead of replacing, or predefining a "principle/main" expected object shape. but other than that, the points i did not comment on are approved. now based on your reasoning up to now, formulate a complete md file proposal for this project, from higher level questions to implementationa roadmap so that this can serve as the primary reference document for researcher and coding agent to be handoff and cary out. you should also contain the details in your verbal replies in this for reference.

## Assistant message 181

[**Download the complete research proposal and implementation handoff (.md)**](sandbox:/mnt/data/Structured_Meaning_Brain_LLM_Research_Proposal.md)

The document contains **24 sections plus four implementation appendices**, integrating the full discussion into one standalone reference.

**Neural encoding and semantic decoding are coequal, independently fitted branches.** Grounding, concept- and stimulus-level geometry, compositional generalization, and consistency remain integrated components, with no predetermined requirement that one map or measurement become the principal result. The foundations preserve the distinction between Schrimpf’s held-out neural prediction and NEURONA’s answer-supervised decoding with unlabeled intermediate groundings. fileciteturn8file0L256-L275 fileciteturn8file1L343-L351

The proposal includes the research questions and novelty analysis; reading-first dataset strategy; semantic ontology and annotation protocol; a 12-model comparison panel; encoding and decoding architectures; geometry estimators; leakage controls and statistical procedures; and a dependency-based execution roadmap. The handoff sections specify the repository structure, data contracts, configuration examples, proposed command-line interfaces, resource estimates, **40 engineering and scientific-integrity tests**, and researcher/agent responsibilities.

Dataset-access uncertainties, checkpoint revisions requiring verification, and incomplete literature retrievals are explicitly recorded. The software interfaces and resource budgets are labeled as proposed specifications, separate from verified source facts and completed work.

## User message 182

good, but this is way to long and not human readble. generate another version based on this one, starting from the first principles explanation so that the proposal can be clearly and straightforwardly understood by human; must be extremely straightforward, explain complicating terms and interpretations, for human reference

## Assistant message 208

[**Download the human-reference proposal (.md)**](sandbox:/mnt/data/Structured_Meaning_Brain_LLM_Human_Reference.md)

Approximately **3,700 words**, compared with roughly 25,000 in the technical handoff. It begins with the lending-a-book example and explains what the brain recordings, model states, and semantic annotations provide; how encoding and decoding work; how concept maps are constructed; and what different results would justify.

Encoding and decoding retain equal standing, alongside grounding, geometry, and generalization. The proposal leaves the eventual findings and form of the representation open. fileciteturn13file0L68-L85

The original technical document is unchanged.

## User message 209

the main thing im concerning are the actual neurosymbolic graph construction and labelling from the dataset, as well as the best dataset for fit; and also, the evaluation pipeline for model-to-human prediction without groundtruth (like neurona): to do this, we have to train something instead of evaluating a frozen model as neurona does? and the other pipelines-concept level distance in human and model, encoding and decoding, etc since these are relying upon the quality of the data symbolic labelling; also, the forzen models are way to out of date and are retaining legacy only because some past literature uses it. lastly, because the entire file is way too long and the first principle version is way stripped down, i actually not sure how exactly the brainstorm fro the project/s shape and framing is right now. explain and push back my concerns clearl

## User message 220

do not edit file-i want you to explain to me

## Assistant message 272

**Your concerns identify the decisions that actually determine whether this project works: what the semantic labels mean, whether the recorded stimuli contain enough examples of those meanings, and what information the trained mappings genuinely recover.** The earlier technical proposal made several implementation choices look more settled than they were. The shorter version then removed too much of the reasoning needed to evaluate those choices.

The scientific direction remains coherent. The graph construction, dataset selection, and model panel need sharper decisions. Also, **NEURONA trains its neural decoding machinery**; keeping the language models we examine frozen is compatible with training those mappings.

## 1. What the project currently is

The central question I would retain is:

> **Which relationships among concepts, event participants, and connected propositions are preserved in human neural responses and contemporary language-model representations—and how do semantic recovery, neural prediction, and representational organization agree or differ?**

We would take the **same recorded language stimuli** and obtain three things: their measured fMRI responses, the internal states of current language models processing that text, and an audited description of the meaning expressed by the text.

Encoding and decoding then provide two equally important measurements. Encoding asks which descriptions predict the measured brain response. Decoding asks which semantic distinctions can be recovered from a brain or model representation. Geometry asks how concepts and configurations relate within each system.

An important source of confusion is that we have been discussing three different graph-like objects:

| Object | How it is obtained | What it describes |
|---|---|---|
| **Stimulus semantic graph** | Extracted from the text and checked against it. | What the passage says: entities, events, roles, references, and relationships between propositions. |
| **Neural grounding map** | Learned from measured fMRI using semantic supervision. | Where the trained decoder obtains evidence relevant to a concept or relationship. |
| **Concept relationship matrix** | Calculated from concept-associated numerical signatures. | Which estimated representations are similar within the brain or a language model. |

**The semantic graph supplies questions and labels. Its edges do not automatically become connections in the brain.** Likewise, the graph saying that two concepts are related does not establish that their neural signatures should have a particular cosine similarity.

This remains **one integrated research project**. The different measurements are ways of examining the same scientific question. The eventual findings could emphasize context-dependent concept organization, a difference between encoding and decoding, or agreement at the concept level alongside divergence at the relational level. We should leave those outcomes open.

## 2. What we train—and what “without ground truth” actually means

### NEURONA’s grounding system is trained

NEURONA’s Figure 1, on page 4, shows measured fMRI being converted into parcel representations, processed by learned concept modules, and combined according to a symbolic query. Section 3.4 explicitly states that it trains on correct final answers using cross-entropy, while intermediate concept groundings have no direct supervision. Its implementation includes learned regional projections and a small convolutional encoder. fileciteturn16file1L190-L211 fileciteturn16file1L343-L351 fileciteturn16file1L1264-L1285

For an illustrative recorded passage, a training example would look like:

> **Input:** its measured fMRI and “Did Leah lend the book to Noah?”  
> **Training target:** yes.

The decoder initially assigns imperfect evidence to concepts and relationships. Its answer error changes the parameters of the neural encoder and grounding modules. Across many examples, the system learns intermediate computations that help produce the right answers.

There is no additional training target saying:

> “The correct grounding of *lending* is parcel 17.”

That is the sense in which grounding is **weakly supervised**: we supervise a downstream answer, and the intermediate map is inferred through learning.

Schrimpf’s study also trains something. It keeps the language-model representations fixed and fits a linear mapping from those representations to measured neural responses. fileciteturn16file0L256-L275

### Our proposed training arrangement

| Component | Proposed treatment | What supervises it? |
|---|---|---|
| LLM used to annotate the text | Use a pretrained model with a fixed annotation procedure; audit its outputs. | The text and annotation instructions; human checks provide quality control. |
| LLM whose representations we study | Keep its foundation-model parameters frozen. | No project-specific training of the foundation model. |
| Neural encoding mapping | Train a separate regression model. | Actual fMRI responses. |
| Brain semantic decoder | Train the neural adapter and semantic readout, including structured modules. | Answers supported by the annotated text. |
| LLM semantic decoder | Train a separate readout on cached LLM states. | The same semantic answer definitions. |
| Concept geometry | Estimate signatures and distances through explicitly identified procedures. | Depends on the signature: measured activity, learned groundings, or encoding predictions. |

**“Frozen” describes which parameters remain unchanged during our experiment. It says nothing about the model’s release date.**

For the main comparison, freezing the LLM preserves the object we are measuring. Fine-tuning it on our neural data would create a different question: how well an adapted model can fit this dataset.

### There are two kinds of known targets

We have **measured fMRI**, which supports numerical neural-prediction evaluation, and **text-supported semantic answers**, which support decoding evaluation.

We lack anatomical labels for each intermediate concept map. Accordingly, a phrase such as “model-to-human prediction without ground truth” needs to be unpacked. Predicting recorded fMRI has a measured target. Inferring where a concept is supported has indirect validation.

**We can investigate concept grounding without anatomical labels. We cannot validate a brain-grounding interpretation using semantic plausibility alone.** The interpretation needs evidence that the recording contributes information and that the estimated organization recurs outside the examples used to fit it.

## 3. How I would actually construct and label the semantic graphs

This is the part that deserves the most design attention.

### Start with an explicit meaning description

Consider this constructed example:

> Leah lent Noah a book. Because she needed it back, she reminded him. Noah returned it.

A suitable annotation contains:

| Record | Example |
|---|---|
| Entity instances | Leah, Noah, this particular book. |
| Event instances | Lending, needing the book back, reminding, returning. |
| Roles | Leah is the lender; Noah is the recipient; the book is the transferred item. |
| Reference links | “She” refers to Leah; “him” refers to Noah; “it” refers to the book. |
| Proposition/event links | The stated need motivates the reminder. |
| Evidence and timing | The words supporting each record and when those words became available. |

The annotation should leave “the reminder caused the return” unresolved unless the text supports it. Sequence alone does not establish that causal relationship.

For the formal structure, **I would use a controlled, UMR-informed representation**. Uniform Meaning Representation extends sentence-level semantic graphs to relationships across sentences, including reference, temporal relations, and distinctions about whether events are actual, possible, or reported. It provides an existing starting point for the schema. citeturn606504view3

We do not need to implement every feature of a full semantic formalism. We need a consistent set of distinctions that the corpus supports and that our measurements can test.

### The graph must preserve identity, roles, and uncertainty

Several details determine whether the graph is scientifically useful.

**Concept types and specific instances must remain separate.** “Book” is a reusable concept. The book in this story is a particular entity. Two mentions can refer to that same entity even when different words are used.

**Roles must remain ordered.** Leah lending to Noah differs from Noah lending to Leah. Replacing both names with `person` too early destroys the distinction. Conversely, arbitrary story-specific names cannot simply become universal concept classes expected to transfer across unrelated stories.

**Predicate normalization must preserve meaning.** “Lending” and “borrowing” describe related perspectives on a transfer. Merging them into a single label without adjusting their roles would corrupt the analysis.

**Statements, intentions, and completed events must remain distinguishable.** “Noah promised to return it” does not say that the return already occurred. Negation and reported beliefs need the same care.

These decisions belong in the annotation scheme before we ask what brain or model representations preserve.

### The production pipeline should have five concrete stages

**First, recover the exact presented text and its timing.** The feature-extraction transcript I inspected for *How to Draw* is a lowercase, word-per-line export containing speech markers. This is usable material, although sentence boundaries, markers, and correspondence to the originally presented sequence need explicit handling. We should preserve word identities and offsets rather than rewrite the narrative into cleaner prose. fileciteturn20file0L2-L5

**Second, extract local propositions and event roles.** A capable annotation model proposes structured records with supporting word spans. Each event receives a predicate, its participants, their roles, and factual status.

**Third, connect the records across the available context.** Resolve references and add supported relationships between events or propositions. For analyses at a particular point in the story, the annotator sees only the prefix available by then. A later revelation must not silently determine an earlier label.

**Fourth, compile questions from the validated records.** This should be largely deterministic. A role record can produce a role-identification question or binding-verification question. A reference link can produce a referent-choice question. Generating the graph, question, and answer through three unconstrained LLM calls would introduce unnecessary opportunities for disagreement.

**Fifth, independently check the labels against the original text.** Human reviewers should inspect complete contexts and answer the relevant semantic questions. Merely checking that an LLM’s answer agrees with its own graph would leave shared errors undetected.

I would aim for a large automatically annotated training collection, a carefully adjudicated evaluation collection, and a representative audit of the training labels. The audit should cover both ordinary examples and difficult reference, role, negation, and discourse cases.

### What annotation quality should mean

A high score for identifying nouns is insufficient for this project. We need to know whether the annotation correctly captures:

- complete event bindings, including who occupies each role;
- reference links and the identity of the linked entity;
- the direction and type of relationships between propositions;
- the answer to the complete evaluation query.

A graph can contain every relevant word and still reverse the meaning of the sentence.

Negative examples also require checking. Failure to find an event in an incomplete annotation does not prove that the text contradicts it. Depending on the task, labels should distinguish **supported, contradicted, and unresolved**.

**All evaluated models should face the same audited semantic reference.** Each model can separately be tested on its ability to generate graphs, although allowing every model to define its own evaluation labels would undermine the representation comparison.

### Your “higher-level linked concepts” idea remains included

There are several different relationships we can represent: a concept belonging to a broader category, entities participating in an event, and propositions supporting or explaining other propositions.

The last category is especially relevant to your original long-text idea. A scientific explanation might connect pressure, buoyancy, displacement, and floating through explanatory propositions. A narrative might connect a character’s need, request, and subsequent action.

We should encode the relationship explicitly and retain its textual support. **The annotation scheme specifies what we ask about; it does not prescribe the shape of the resulting neural organization.**

### A real difficulty in extending NEURONA to language

Language introduces recurring entities, repeated events, and relationships spanning different times. NEURONA’s candidate neural entities are predefined parcels; those parcels have no guaranteed one-to-one correspondence with a person or event. The paper itself identifies the absence of canonical neural “objects” as a central challenge. fileciteturn16file1L141-L159

Consequently, copying an image-style “person–holding–object” module and adding more labels will be insufficient for unrestricted discourse decoding. Our readout needs temporal context and a way to preserve ordered participants.

This also means distinguishing **within-story identity tracking** from **cross-story generalization of semantic roles and event structures**. They are both useful experiments, with different difficulty and evidence requirements.

## 4. Which dataset best fits this project?

**For the broad reading-based project, Deniz’s multi-story reading condition remains my strongest candidate. Its status should remain “candidate pending file and semantic-coverage verification.”** I should not have allowed the earlier documents to make that qualification feel like a settled implementation detail.

There are two important dataset properties: the diversity of semantic situations and the amount of reliable recording for each person. More participants reading a tiny set of passages does not create more distinct semantic combinations.

| Resource | Fit to this project | Main constraint |
|---|---|---|
| **Deniz multi-story reading subset** | Six participants read eleven narratives; suitable for recurring concepts, event roles, and discourse across multiple stories. | The exact neural files and usable metadata still need verification; semantic coverage has not been measured. citeturn606504view1 |
| **Reading Brain Project, adult English subset** | The release documents 52 participants with reading fMRI and eye tracking. Its five scientific texts are attractive for relationships among explanatory concepts. | The original stimulus set contains only about 1,500 words, limiting semantic diversity despite the larger participant count. fileciteturn21file0L2-L5 citeturn606504view0 |
| **Wehbe narrative reading** | Eight participants, a coherent story, and existing character/action-related feature resources. Useful for recurring entities and reference. | One chapter provides limited independent story diversity. citeturn396765view3 |
| **Pereira sentence/passages** | Repeated recordings for 627 sentences in short passages support sentence-scale comparisons. | Limited extended discourse. fileciteturn16file0L1060-L1063 fileciteturn16file0L1122-L1125 |
| **LeBel narrative listening** | Eight participants with 27 narratives each; the release also describes an extended collection for three participants. Stronger within-person training coverage. | Auditory presentation requires additional controls and appropriate wording of the interpretation. citeturn606504view2 |

### A dataset the earlier shortlist underemphasized

The **Reading Brain Project** is particularly relevant to your idea of concepts linked into larger explanatory structures. Its scientific passages are manageable enough for detailed human annotation.

However, it already has prior work representing textual knowledge as concept networks. Those networks used selected keywords and proximity-based connections; they are useful precedents or baselines, rather than ready-made labels for the exact semantic relationships we want. Our contribution would need to concern richer relational annotations and what their neural/model measurements reveal. citeturn606504view0

I would use this resource for a well-defined explanatory-structure analysis. I would hesitate to make five short texts the sole training foundation for a broad, vocabulary-rich neuro-symbolic decoder.

### My actual selection recommendation

Use **one multi-story discovery corpus and one complementary corpus chosen for a specific scientific contrast**. We do not need every dataset in the earlier document.

For a reading-first study, that means Deniz as the broad candidate, with either scientific reading for explanatory concept relationships or Wehbe for recurring narrative identity. Pereira is useful when a particular analysis benefits from repeated sentence measurements.

If the Deniz files or semantic coverage prove unsuitable, **the extended LeBel resource is a serious alternative**. Its documentation explicitly identifies additional within-person data intended to support demanding language encoding and decoding analyses. citeturn606504view2

I would accept acoustic controls to obtain substantially better semantic coverage. The LLM can still receive the transcript. Word/phoneme timing and acoustic information enter as comparison features; this does not require turning the project into speech-model development.

**The unresolved quantity is how many independently occurring, correctly labelable relationships each corpus supplies.** A participant count, scanning-hour total, or number of generated questions cannot answer that.

## 5. How the evaluation pipelines actually depend on the labels

Your concern about shared dependence is correct for several branches. It does not apply equally to all of them.

| Measurement | Needs symbolic labels? | What it tests |
|---|---|---|
| **LLM states → measured fMRI** | No. | Whether the model representation predicts the recorded response. |
| **Semantic graph features → measured fMRI** | Yes. | Whether the annotated semantic distinctions predict neural responses. |
| **fMRI + query → semantic answer** | Yes. | Whether those distinctions are recoverable from neural recordings. |
| **LLM states + query → semantic answer** | Yes. | Whether the same distinctions are accessible in the model representation. |
| **Concept-conditioned geometry** | Yes, to identify and organize occurrences. | Relationships among estimated concept or configuration signatures. |
| **Stimulus-level brain–model geometry** | No semantic graph is required for the basic comparison. | Whether actual passages/events have similar representational relationships across systems. |

The first and last measurements provide label-independent comparisons. Schrimpf includes both trained model-to-brain prediction and representational comparisons without a fitted alignment. fileciteturn16file0L256-L284

However, good overall LLM-to-fMRI prediction cannot certify that our label for a particular causal relation is correct. The graph-dependent conclusions still require annotation validation.

### How the graph becomes useful to encoding

Encoding requires a numerical representation of the graph. We can construct features that distinguish concept occurrence, predicate–role–participant combinations, reference links, and relationships between propositions.

A simple unordered average of all concept embeddings would erase important distinctions. “Leah lends to Noah” and “Noah lends to Leah” could receive the same vector. A role-aware representation gives different positions or transformations to the lender and recipient.

We then train regression models to predict the actual fMRI, comparing flat concepts, structured features, LLM states, and combinations of them. This tests whether explicitly retaining relationships contributes predictive information.

A large graph neural network is optional. The scientific requirement is that the representation preserves the distinctions being tested and that its comparison controls are clear.

### How the graph becomes useful to decoding

The graph generates the target questions and answers. At test time, the brain decoder receives the recording and a query, plus any legal answer candidates. It does **not** receive the correct source graph.

A structured decoder estimates concept and relation evidence, then combines it according to the query. A simpler decoder provides a comparison without explicit composition. Both require comparable training and evaluation.

For the LLM-side analysis, the representation is saved **before the question is introduced**. We train a separate readout from that saved state. Allowing the LLM to reread the passage and generate an answer after seeing the query would measure prompted problem solving, which can be reported separately.

### How concept distances are obtained

For “lending,” we gather many held-out occurrences and estimate a signature. We repeat this for other concepts and for role-conditioned configurations where coverage permits.

That signature could describe the decoder’s regional evidence, a measured activity contrast associated with the concept, or a response predicted by the encoding model. These have different interpretations and should retain separate labels.

We then compare concept pairs within each system:

> How similar are the estimated signatures of lending and giving?  
> Does that relationship recur across different contexts?  
> Does a model’s corresponding relationship agree with the brain-derived relationship?

The brain and model vectors can have different dimensions because we compare their **within-system relationship matrices**, with the same concepts indexing both.

Two qualifications matter. First, a freely learned latent can change its geometry while preserving its answers, so successful decoding does not uniquely establish the latent distances. Second, similarity of distributed patterns says nothing by itself about physical proximity in cortex.

### What would prevent the whole analysis from reproducing its own labels?

The strongest checks are concrete:

**Use a question-only baseline.** It receives identical queries and answer candidates, allowing us to measure what can be guessed without the recording.

**Train an otherwise identical system with disrupted stimulus–fMRI pairing.** A semantically attractive map that also emerges without the true pairing has weak support as a neural interpretation.

**Estimate the relationships again in independent contexts and participants.** Shared vocabulary, recurring characters, or one story should not alone explain a purported general concept organization.

These controls preserve your interpretation-focused goal. They establish which parts of the result are constrained by measured neural information.

## 6. You are right about the old model panel

**I would withdraw the previous twelve-checkpoint list as the default panel.** Historical comparability is insufficient justification for making GPT-2, BERT, or older model families dominate a study aimed at contemporary language representations.

Current official releases provide substantially newer candidates. Examples verified for this response include:

| Candidate | Why it merits consideration |
|---|---|
| **Qwen3.8-27B** | A current post-trained open-weight checkpoint with accessible model artifacts. citeturn396765view0 |
| **Gemma 4 models, including 12B and 31B variants** | Another contemporary family with different sizes and inspectable representations. citeturn396765view1 |
| **Mistral Small 4** | A 2026 open-weight model with configurable reasoning and a mixture-of-experts architecture; its total weight size makes it a more demanding reference. citeturn396765view2 |

These are candidates rather than a newly locked registry. I would build a contemporary panel around **scientifically useful contrasts**: multiple families, some size variation, and matched training-stage variants where such pairs actually exist. A historical model could serve a narrowly identified replication check; it need not occupy the core analysis.

I would also remove the requirement that the full study’s context lengths accommodate BERT. Context should be chosen around the semantic dependencies we intend to measure and the accessible capabilities of the selected models.

My pushback concerns what “newer” guarantees. Release date does not establish better neural correspondence, better annotation of our particular distinctions, or greater interpretability. These are empirical questions.

There is another important distinction for reasoning models. Stimulus-only hidden states measure information accessible after processing the passage. Generated reasoning and query-conditioned answers measure additional computation performed when asked. Failure of a simple probe on the former cannot establish inability on the latter. We can retain both as separately defined analyses without mixing their scores.

## 7. Where the brainstorm stands now

**The project has a clear scientific shape, with unresolved data and measurement choices.** Its intended contribution is a shared, audited semantic description of recorded language, separately trained encoding and decoding models, and an empirical account of how semantic organization is shared or differs between human responses and current LLMs.

I would keep the integrated scope. I would discard the appearance that a particular model count, latent dimension, corpus combination, or final concept atlas has already been justified.

The most consequential unresolved decision is the semantic representation: which distinctions we can annotate consistently, how those distinctions remain available over time, and how many independent examples the actual recordings contain. Once that is established, the trained readouts and geometry analyses have well-defined objects to measure.

**My judgment remains that this is worth pursuing.** The immediate scientific work is to establish a trustworthy stimulus-to-semantics layer and its coverage in an appropriate corpus. That layer supports the ambitious comparison you originally proposed; it also determines where our eventual interpretations will be strong, qualified, or unsupported.
