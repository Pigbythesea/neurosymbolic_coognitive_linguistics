**Codebase plan V2**

**Scientific redesign, 8 October 2026. Implementation has not started.**

The aim is to understand **how concepts and relationships expressed in language are organized in human brain activity and in LLM activity, where their brain associations lie, and whether a learned transformation connects the two systems.**

This plan replaces the previous requirement to build three equally elaborate encoding, question-answering, and geometry pipelines. It keeps their useful scientific questions. It also keeps the existing recordings, reviewed annotations, and frozen LLMs. The change is in how we turn those materials into an experiment.

Use this document as the starting point for the redesign. The older [scientific handoff](SCIENTIFIC_STATUS_HANDOFF.md) records V1 decisions and the existing implementation. The [original handoff](PROJECT_HANDOFF.md) remains useful for motivation. The [engineering handoff](ENGINEERING_HANDOFF.md) remains the source for transfer and execution history. V2 does not claim that new fits, maps, or results already exist.

**The three questions remain the purpose of the project.**

1. **Organization:** Which meanings have similar or different activity patterns? Does that organization change when a concept plays a different role or appears in a different context?
2. **Localization:** Which measured brain locations have responses associated with particular concepts and relationships?
3. **Correspondence:** Can a transformation learned from paired LLM and brain activity predict their relationship during an unseen story? Which meanings does that correspondence capture?

For example, “Leah gave Noah a book,” “Noah gave Leah a book,” and “Leah imagined giving Noah a book” share much of their content but express different relationships. The experiment should make these differences intelligible. These sentences illustrate the question; they are not proposed artificial fMRI observations or a replacement dataset.

**The central experiment consists of three learned mappings.**

| Input | What we learn to predict | What it tells us |
|---|---|---|
| Meanings described by the reviewed graphs | Recorded activity at brain locations | Spatial associations between meaning and brain responses |
| The same descriptions of meaning | Recorded internal activity of a frozen LLM | Which semantic distinctions organize that model's activity |
| LLM activity during the actual stories | Brain activity during the same stories | A direct correspondence between the two systems |

“Learn” means fitting a relatively small statistical mapping. The LLMs stay frozen. The third mapping learns from paired recordings without semantic labels directing the alignment. This matters: teaching two systems the same labels is not, on its own, evidence that their activity corresponds.

These mappings share one preparation pipeline and one story-based evaluation procedure. Maps, concept comparisons, and explanations are outputs of these fits, rather than separate collections of unrelated experiments.

**1. Keep the existing scientific materials and separate them from V1 analysis decisions.**

Use the Deniz reading recordings for all nine participants. Keep the ten development stories and the reserved final story. Keep the accepted independently reviewed graph export identified in [the semantic configuration](../configs/semantics.json).

Reuse the frozen-model identities and revisions already recorded in [the model lock](../manifests/frozen-models.lock.json). Reuse cached word-level states where their text, layer, and extraction identities match. There is no scientific reason to repeat expensive model extraction merely because the downstream experiment changes.

Keep the complete graphs as the semantic archive. Their people, events, argument links, reference links, scope, discourse relations, identity, and state changes remain available. V2 does not reduce the research to a bag of nouns. It does remove the assumption that every annotation field must become an independent prediction task.

Keep V1 outputs as historical results under their existing identities. New feature definitions and fits get separate V2 paths and identities. In this repository, “execution protocol 2” already names a V1 optimization; it must not be confused with this scientific redesign.

**2. Build a readable record of meaning occurrences before building numerical features.**

An occurrence is a place in the text where a concept appears or a relationship is expressed. It is not automatically the whole passage supplied to the annotator.

The compiler should produce records a person can read: the source text, the concept or proposition, its participants and their roles, relevant scope or reference information, and the supporting text spans. Each record links back to the accepted graph.

For the lending example, a record might say that the event is giving, Leah is the giver, Noah is the recipient, and the book is the transferred object. A separate field describes whether the event is asserted, imagined, negated, or inside another explicitly annotated context. Operator order remains available when it changes the meaning.

Two people remain two different referents even when both have the category “person.” A reference such as “it” keeps its link to the particular book. Story-specific identifiers establish these links; their arbitrary ID strings are not semantic features.

Compile occurrences at supported mentions and event or relation expressions, including later references. A concept should not disappear from the time series merely because its graph node was first introduced earlier in the story.

The existing review remains accepted. New semantic judgments are needed only where the new representation exposes a concrete missing or ambiguous fact. A second wholesale annotation exercise is not part of this plan.

**3. Put those meanings on the actual reading timeline.**

The basic neural observation is a row of the recorded fMRI time series. We explain that time series using the meanings encountered as the story unfolds.

Use the released word timing and the text spans supporting each occurrence. For a relationship that becomes specified across a clause, use the completion of the evidence needed for that relationship. Include the relevant qualifier or scope evidence when it changes what is being represented. A later clarification is an update at its own location, not information silently inserted into an earlier moment.

The whole annotation-unit endpoint is no longer the universal timestamp. An annotator's work batch can cover many events and is not a biological trial.

Some existing records may not establish a precise earlier availability time. Preserve that uncertainty rather than inventing a timestamp. Such a record can retain its documented broader timing or remain available for untimed interpretation. This affects the particular observation, not the validity of every annotation.

Build a continuous feature time series and allow its effects to appear over subsequent fMRI samples. Reuse the existing within-story delayed-feature machinery: it lets the model learn how recent semantic input contributes to a later blood-oxygenation response. The current one-to-four-TR delays are a practical initial setting, not a claim that every concept has an identical biological delay.

The semantic description and LLM activity must be sampled against the same presentation record. The brain mapping uses delayed inputs. The description-to-LLM mapping uses the corresponding undelayed text timeline. For direct LLM-to-brain prediction, apply the brain-delay construction to the LLM inputs.

Aggregation within a scan interval represents several words or events contributing to one slow measurement. This is different from averaging all passages containing a concept and declaring that average its representation. Preserve the individual occurrence records even when several contribute to the same scan interval.

This remains an offline association study. It does not assume that we observe the exact millisecond at which a participant understands a proposition.

**4. Use a numerical description that shares information across related meanings and preserves their structure.**

The representation should allow experience with giving, receiving, people, and objects to inform a new combination. V1's separate feature for each complete predicate–role–filler combination often could not do that.

Use separate, understandable feature blocks for content and for its organization. Content describes the concepts and predicates present. Organization describes their role assignments, reference links, scope, and discourse or state relationships.

A practical default is one shared frozen text encoder for short semantic descriptions, with the graph supplying explicit component and role structure. Concept descriptions, participant descriptions, and descriptions of their bindings share the same numerical vocabulary. The encoder is a reusable way to represent related words and phrases, not another brain model to train.

One concrete implementation default is [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), a compact sentence/paragraph encoder. Pin its revision when preparing V2. Its role is numerical representation of the annotations; the contemporary frozen LLM panel remains the object of comparison. This default is not a claim that its embedding geometry is human semantic truth.

Preserve role assignments explicitly in separate blocks or role-conditioned features. Include the supported identifying descriptions of participants, rather than replacing every participant by its broad category. Ordered scope and reference links need corresponding structured features. Do not encode one enormous serialized graph and assume that every relevant distinction survived.

For implementation, the readable record and the numerical representation should demonstrate the same distinctions on real corpus examples: changing the giver and recipient changes the role features; resolving a pronoun to a different referent changes the reference features; changing the applicable scope changes the scope features. These checks establish what the representation measures. They are not synthetic scientific results.

The exact vector widths and storage layout are engineering choices. Record them in one configuration. The scientific commitments are that related meanings can share information, roles and referents remain distinguishable, and the representation can be explained through its source records.

Keep an accompanying content-only description built from the same occurrences, participants, predicates, and timestamps. Removing structure should remove the relationships, not also remove most of the source content or change when it appears. This supplies a direct comparison when asking whether structured meaning contributes information.

The descriptor encoder introduces a linguistic prior. Results learned using these descriptors are therefore model-based semantic descriptions. The direct brain–LLM correspondence experiment does not depend on that encoder, and the organization report also displays similarity in the input descriptions themselves. Agreement already present in those inputs must not be presented as a new discovery about the brain.

**5. Learn spatial semantic maps by predicting measured brain responses.**

For each participant, fit the description of meaning to the recorded response at each available brain voxel. A voxel is a small measured volume of brain tissue. Start with regularized linear regression: a weighted combination of semantic features, with a penalty that discourages unstable large weights.

This is the main anatomical experiment. It uses all usable native response coordinates. Atlas parcels are named groups of locations used for summaries and display; they are not the places where we presume concepts must live. The existing parcel assignment has incomplete coverage, so it must not silently determine which native voxels are fitted.

Reuse the verified numerical solvers where suitable, but allow the amount of regularization to vary across output locations using training data. Do not require one cortex-wide tuning choice to serve every location. A large search over random feature-group mixtures is not the scientific objective.

First report how well the complete semantic model predicts an unseen story at each location. Then show the difference from the same-content description without structure where the claim concerns roles, reference, scope, or discourse. Keep basic presentation predictors such as reading rate in both sides of a comparison.

To visualize a concept, evaluate its description through the fitted semantic mapping. To visualize a relationship, examine the fitted response associated with that relationship in its recorded context. Where helpful, compare two clearly described model inputs, such as the recorded role assignment and a swapped assignment.

These are fitted response profiles and model contrasts. We did not record the participant reading a counterfactual swapped story. Save the temporal response profile as well as any spatial summary, so a display does not conceal which delays it combines.

A meaningful map should be accompanied by prediction performance and examples of the supporting passages. Similar meanings that repeatedly co-occur may remain difficult to distinguish. Describe the supported association at that level rather than assigning each word an exclusive brain address.

This approach follows the established logic of spatial semantic encoding in [Huth et al.](https://www.nature.com/articles/nature17637) and the [Deniz reading/listening study](https://pubmed.ncbi.nlm.nih.gov/31427396/). The proposed extension concerns structured meaning and its comparison with LLM representations.

**6. Learn how the same meanings organize LLM activity.**

Fit the same semantic descriptions to each frozen model's recorded internal activity during the stories. Use the existing word-level states and preserve their layer identity.

This gives each meaning a fitted activity profile within the LLM, just as the brain fit gives it a spatial response profile. LLM coordinates are not anatomical regions. Do not divide them into arbitrary groups and interpret those groups as model versions of brain parcels.

Use these profiles to examine which concepts and relationships are distinguished similarly within each system. For example, are giving and receiving associated with related profiles? Does imagined giving differ from asserted giving? Does changing participant roles change the profile?

A pairwise similarity matrix is a useful display: each cell compares two learned profiles. A comparison between the brain and LLM matrices asks whether the same pairs are relatively similar or different. Explain that meaning beside the plot. A correlation between matrices is neither a percentage of shared concepts nor a percentage of brain activity explained.

Keep context-specific predictions available. A single overall concept profile may be shown as a summary of a clearly described set of contexts, but it is no longer the required starting object.

Only meanings with actual observational support can receive an empirical interpretation. An encoder can generate a vector for an unseen concept, and a fitted model can project it into a brain map, but that projection is an extrapolation rather than measured evidence for that concept.

The shared semantic description makes these maps comparable; it also influences their organization. Report that influence openly. These comparisons characterize organization under the fitted semantic description, not a uniquely determined geometry of the untouched biological and artificial systems.

**7. Test brain–LLM correspondence directly, without using the annotations to force it.**

Pair the LLM activity and fMRI response through the story timeline. Fit a regularized linear transformation from the delayed LLM features to brain responses. Evaluate its predictions on an unseen story.

Also provide a compact form of this same mapping: reduced-rank regression. In ordinary language, it asks whether a smaller collection of combinations of LLM activity can account for a corresponding collection of combinations of brain activity. Choose the amount of compression using training-story validation.

Save both sides of this learned space. On an unseen story, project the actual LLM activity and actual brain activity through their respective training-fitted transformations, and examine whether their trajectories correspond. Comparing a model prediction with itself is not a correspondence test.

The number of retained dimensions is the size of a useful predictive representation under this model. It is not an estimate of the total number of concepts or mental dimensions in the brain. A successful linear map establishes a linear predictive relationship under the measured conditions, not identical neural mechanisms.

After learning this correspondence, use the semantic records to interpret it. Fit descriptions of the shared variation on training stories and examine those descriptions on unseen stories. Show which concepts, roles, scope distinctions, or discourse changes accompany the shared variation, and where that variation projects onto the brain.

This separates two claims that V1 risked blending: there is shared predictable variation, and some of that variation is associated with particular meanings. Shared variation could also reflect presentation or other nonsemantic properties; semantic interpretation has to earn its name through the observations.

Retain a full linear mapping as the comparison for the compressed mapping. This is one correspondence analysis with a question about dimensionality, not a new tournament of unrelated decoder architectures.

Coordinates learned in separate fits can rotate or change sign. Compare their predictions and supported subspaces; do not average “dimension 3” from different fitted models as if it necessarily names the same thing. The general choice of mapping should follow the scientific claim, as discussed by [Ivanova et al.](https://arxiv.org/abs/2208.10668).

**8. Retire the current QA machinery from the default V2 experiment.**

The core questions above do not require all-prefix answer catalogues, query-only neural networks, three answer-scoring architectures, or a 200-by-200 array of regional routing scores.

Keep the old implementation and its outputs for reference, but do not make V2 preparation, fitting, or reporting depend on them. Direct semantic decoding can remain an extension when the research question specifically concerns recovery of an answer from a recording.

If that extension is used, train on explicit semantic distinctions drawn from the same meaning records. Questions should express those distinctions, with appropriate answer alternatives. A comparison that withholds the brain observation is useful there because it tests whether the answer can be inferred from the question alone.

NEURONA remains an inspiration for learning concept associations without concept-to-region labels. Its particular question-answering objective is not necessary for the spatial encoding route chosen here. This is a deliberate methodological change, not a claim to have reproduced NEURONA's decoder. [NEURONA](https://arxiv.org/html/2603.03343v1).

**9. Use ordinary unseen-story evaluation and keep the results understandable.**

Keep the ten development stories. Train on nine and predict the remaining story, repeating so each story is evaluated while excluded from fitting. Choose regularization and compression using validation stories inside the training set. The existing three inner story folds can be reused as an execution default.

Keep story 11 reserved for the final evaluation after V2 development decisions are fixed. It remains one additional story, not a large independent population of narratives.

Remove the special 01–05 versus 06–10 partitions from the default analysis. They were consequences of the previous averaging design. V2 can study variation across stories using the ordinary story evaluations.

Keep all nine participants and the existing five frozen-model conditions. Use the existing quarter-depth and final-layer positions for the depth profile, sharing preparation and computations where possible. Label final-layer comparisons directly; any “best layer” selection comes from training stories.

Report results by participant, story, model, layer, and brain location before summarizing. Repeated scans help characterize measurement consistency where available. They are not additional people or stories, and a reliability estimate from one repeated story is not automatically a ceiling for every other result.

Keep a small number of comparisons with clear purposes: presentation predictors for basic stimulus effects, the matched content-only description for structural claims, and a timing-disrupted pairing for checking that direct correspondence depends on matching observations. Timing disruption should preserve the time-series structure, for example through documented within-story shifts that exceed the modeled response window. It is a diagnostic, not a complete account of every possible confound.

Use participants and stories when describing variability. Do not treat thousands of neighboring voxels or questions as thousands of independent replications. Maps making formal statistical claims need an appropriate map-level error procedure; descriptive maps should be identified as descriptive.

There is no universal correlation threshold that certifies the project. Judge what is predicted, where, how consistently, and how much measurement variation is available. The final report should explain each number in a sentence and show representative actual and predicted responses.

**10. Build one integrated V2 implementation using the existing foundations.**

The following module names are proposed destinations, not files or commands that already exist.

| Proposed component | Its responsibility | Existing work to reuse or replace |
|---|---|---|
| neurosym/v2/meaning.py | Read accepted graphs and emit readable occurrence records with evidence and identities | Reuse reviewed_archive.py and reviewed_graph.py; replace the measurement choices in reviewed_compile.py |
| neurosym/v2/features.py | Turn the records into shared content and structured descriptions and numerical features | Replace exact-combination lookup as the principal representation in semantic_features.py |
| neurosym/v2/timeline.py | Join meanings, word-level LLM states, and fMRI rows through actual timing | Reuse corpus.py, deniz.py, temporal.py, and model_features.py; replace universal unit-end timing |
| neurosym/v2/mapping.py | Fit and evaluate semantic-to-brain, semantic-to-LLM, and direct LLM-to-brain mappings | Reuse suitable encoding.py and compute.py numerical routines; add reduced-rank fitting |
| neurosym/v2/maps.py | Produce spatial semantic maps, context-specific profiles, and correspondence maps | Reuse spatial.py and the released surface mappers; replace decoder-routing localization as the default |
| neurosym/v2/organization.py | Compare learned concept profiles and interpret shared variation | Replace native passage-average concept signatures as the central geometry |
| neurosym/v2/report.py | Produce figures, readable examples, results, and a clear account of what each result measures | Reuse useful report/export utilities, without inheriting the old experiment matrix |
| configs/science_v2.json and scripts/run_v2.py | Define and execute the coherent V2 study | Reuse job, checkpoint, bundle, and identity infrastructure after removing V1 analysis dependencies |

Use separate prepared-data and result directories, such as data/processed/meaning-v2 and data/analysis-v2. Input identities include the accepted annotation export, corpus/timing identity, frozen-model locks, and descriptor-encoder revision.

The implementation sequence is straightforward: define the occurrence record; compile real annotated stories into it; prepare aligned features; implement all three mappings; implement the spatial and organization reports; then connect the complete workflow to cluster execution. These are dependencies within one implementation, not successive miniature scientific projects.

The meaning/timing work and numerical mapping/report work can proceed concurrently after their shared input format is agreed. Integration uses the real corpus and available real evidence. Do not build a dummy pipeline to stand in for inaccessible recordings.

**The finished code should deliver a small set of substantive research outputs.**

- A readable atlas of concept and relationship associations, with examples and prediction evidence.
- A comparison of semantic organization in human recordings and each LLM, including context and layer differences.
- A direct brain–LLM correspondence result, with the shared variation interpreted through the annotated meanings.
- A record of the underlying predictions, fitted maps, source occurrences, and uncertainty, so a reader can trace an interpretation back to observations.

One result need not be positive for the others to be useful. A useful correspondence with weak semantic attribution, for example, is a different conclusion from a strong concept map with weak correspondence to a particular LLM.

**Local work and cluster work remain separate for practical reasons.**

Locally, implement the integrated code, compile the existing text and annotations, inspect real timing and semantic examples, and verify the interfaces against the saved dataset inspection and any available real arrays. Reuse existing model-state metadata. A new descriptor model can be prepared wherever access and resources permit.

On allocated cluster compute, use the complete recordings and model-state caches to prepare full numerical inputs, fit the study, and export evidence and maps. The first full-data execution must verify actual dimensions, time indices, identities, and anatomical mappings before producing scientific fits. Local compatibility checks alone cannot certify arrays that are only on the cluster.

The researcher operates SSH and authorizes remote actions. Install environments, populate caches, and run jobs only on allocated compute, never on the login node. If a required resource is inaccessible or a terminal operation will be long, report the exact requirement and give the researcher the necessary steps. Local user-run commands must be CMD commands, not PowerShell.

Reuse the existing concurrency limit and scheduler mechanisms rather than copying the old scientific job inventory. Linear algebra can use GPUs efficiently and descriptor preparation is comparatively small, but the V1 GPU-hour estimate does not predict V2 runtime. Cost the new manifest from its actual dimensions and operations before launch; no particular wall time is promised here.

**The implementation handoff is to replace the measurement design while preserving the research question.**

Implement this plan as an integrated V2 path. Preserve the reviewed semantic archive, verified input readers, frozen model caches, and useful execution infrastructure. Make readable meaning occurrences, spatial semantic maps, and direct brain–LLM correspondence the organizing objects. Keep V1 results identifiable as V1. Explain any substantive change to these scientific meanings in plain language before silently substituting a different experiment. Routine implementation choices do not require new approval checkpoints.
