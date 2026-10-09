**Codebase plan V2**

**Revised scientific and implementation direction, 8 October 2026. V2 implementation has not started.**

**Architecture clarification, 9 October 2026:** The encoding and decoding choices below now identify their literature basis and the adaptations needed for this study. The scientific scope remains the agreed V2 scope.

The aim is to understand **how concepts and relationships expressed in language are organized in human brain activity and in LLM activity, where their brain associations lie, and how the two systems correspond.**

Understandable semantic descriptions connect those questions. Encoding, semantic decoding, representational similarity analysis (RSA), and a learned linear transformation provide complementary measurements. None is appointed the required successful outcome or a secondary research question. They share data preparation, timing, evaluation splits, and reporting.

This revision incorporates the researcher's latest decisions: simpler measured concepts; encoding and decoding retained together; concept-associated brain maps from both directions; RSA explicitly retained alongside linear correspondence; and no mandatory external text encoder defining the semantic space. It supersedes the earlier V2 draft's encoding-first emphasis and optional semantic decoding.

Read this file first for the current redesign. The [original brainstorm transcript](ORIGINAL_BRAINSTORM_TRANSCRIPT.md) preserves the motivation and the researcher's corrections. The [first-principles reference](Structured_Meaning_Brain_LLM_Human_Reference.md), [technical proposal](Structured_Meaning_Brain_LLM_Research_Proposal.md), and [historical handoff](PROJECT_HANDOFF.md) provide background, not additional implementation requirements. The [scientific status handoff](SCIENTIFIC_STATUS_HANDOFF.md) records V1, and the [engineering handoff](ENGINEERING_HANDOFF.md) owns execution and cluster status. This plan does not claim that V2 fits or results already exist.

**1. Define concepts simply, and let their relationships do useful work.**

A measured concept is a reusable meaning that we can name and identify in the text. Examples include person, book, holding, giving, remembering, and fear. Concepts can concern physical things, actions, properties, or abstract meanings. They are not restricted to nouns or identical to literal word strings: different expressions can convey the same concept.

Combine those concepts into understandable expressions, such as holding(person, book). A particular occurrence retains its participants and roles. Giving a book to someone uses reusable meanings and role links; the entire event does not become a new isolated class.

The default target is **meaning expressed or referred to in the text**. A passage about not giving still brings up giving. Whether a giving event is asserted is a further distinction supported by its annotation. Preserve the status needed to avoid confusing these interpretations, but do not require every nested scope or discourse detail to become a prediction target.

The reviewed graphs remain the semantic archive. Richer reference, scope, event, and discourse information remains available when it specifies a meaning being investigated or helps explain a finding. The core measurement uses clear concepts and explicit relationships. No exhaustive linguistic ontology or fixed inventory of hierarchy experiments is required.

Relationships must affect the numerical representation or learned readout. Storing a graph while discarding all its edges would not test the proposed neurosymbolic account. Conversely, showing that two compiler outputs differ is not evidence that the brain distinguishes their meanings. The experiment asks whether those distinctions are supported by activity.

**2. Reuse the real materials already prepared.**

Keep the Deniz reading recordings for all nine participants, the ten development stories, and the reserved final story. Use the independently reviewed graph export identified in [the semantic configuration](../configs/semantics.json).

Reuse the contemporary frozen-model identities and revisions in [the model lock](../manifests/frozen-models.lock.json). Keep the existing five model conditions and quarter-depth and final-layer measurements. Reuse word-level state caches when text, layer, and extraction identities match. The LLMs remain frozen; statistical mappings and semantic readouts are trained.

Preserve V1 results as historical evidence. V2 gets distinct prepared-data and result identities. "Execution protocol 2" already refers to a V1 optimization and must not be mistaken for this scientific redesign.

The existing annotation review remains accepted. Compile from it rather than commission another wholesale annotation exercise. If a required fact is missing or ambiguous, report that concrete issue and retain its uncertainty.

**3. Build one readable representation of meaning occurrences.**

An occurrence is a place where a concept or relationship is expressed. Record its story, supporting text spans, reusable concept labels, participants and role links, relevant status, and link back to the reviewed graph. Keep the source text readable beside the compiled record.

An occurrence is not automatically the entire passage supplied to the annotator. Repeated mentions and later references can produce new occurrences. A concept does not disappear from the timeline because its graph node was first introduced earlier.

Preserve individual referents: two people are not the same person simply because both have type person. Arbitrary graph IDs are bookkeeping, not new semantic concepts. Each prediction target states whether it concerns participant types, particular participants, or a relationship. Do not claim identity recovery from features that only preserve types.

Start with numerical features for reusable concepts and their role-linked combinations. Components share parameters or features across events, so a new combination can draw on familiar meanings. Avoid V1's separate, unshareable category for every complete predicate-role-filler tuple.

No particular external descriptor encoder is mandatory. MPNet is no longer a default scientific commitment. A frozen text encoder may help represent related descriptions if that sharing is needed, but record its contribution and keep symbolic labels and relationships explicit. Its embedding distances are not an answer about human conceptual organization.

Use real annotated examples to explain what the features retain and what the targets mean. Keep a content-only version of the same occurrences for the comparison that asks what relationships contribute. This version removes relationships while preserving the corresponding content and timing.

**4. Put meanings, model states, and recordings on one documented timeline.**

Use the released word timings and the evidence spans supporting each occurrence. A relationship becomes available when its supporting text has been read; a later clarification is an update at its own location. Do not insert knowledge from later text into an earlier observation.

The annotation-unit endpoint is no longer the universal timestamp. Where the graph only supports a broader interval, preserve that limitation instead of inventing exact timing.

The brain response is slow and overlapping. Build a continuous semantic time series and let recent meanings contribute to subsequent fMRI samples. Reuse delayed-feature machinery; the existing one-to-four-TR delays are an initial implementation setting, not a biological claim about every concept. A TR is the interval between successive fMRI volumes.

Semantic-to-LLM encoding uses the corresponding text timeline. Semantic-to-brain and direct LLM-to-brain encoding use delayed inputs. Semantic decoding uses the corresponding delayed response windows to recover the meanings being tested. Record this alignment explicitly: a text event does not have its own clean, independent brain scan.

RSA uses observations from the same timing preparation, with its delay or temporal filtering declared and chosen without optimizing the test result. Preserve the identities of the observations compared. Temporal alignment must not silently become a fitted spatial brain-model alignment before RSA.

Aggregation within a scan interval can be necessary because several words contribute to one measurement. This differs from averaging all passages containing a concept and declaring that average its representation. Keep occurrences and context available throughout. This is an offline association study, not a claim to observe the exact instant of comprehension.

**5. Learn encoding and direct semantic decoding in parallel.**

Encoding asks: **given the expressed meanings, what activity can we predict?** Fit semantic descriptions to each participant's brain responses and, separately, to each frozen model's internal activity. Use the established voxelwise encoding framework: delayed features form a finite impulse response (FIR) model, and regularized regression estimates their association with each brain location. Use ridge for one feature space and banded ridge for jointly fitted feature families needing separate regularization, such as semantic and presentation features. This is a selected scientific measurement, not a preliminary model chosen merely for simplicity. [Voxelwise encoding framework, 2025](https://doi.org/10.1162/imag_a_00575).

For brain encoding, fit usable native response coordinates and select regularization using training data, allowing different output locations to need different amounts. Reuse verified numerical solvers. Save unseen-story predictions and the corresponding response profiles. For LLM encoding, retain model and layer identity.

Include a defined nonlinear brain-encoding comparison in the integrated experiment: a single-hidden-layer multilayer perceptron (MLP) predicts a compressed brain representation learned by principal component analysis (PCA), and its predictions are projected back into voxel space. Use the architecture in [Han et al., 2025 preprint](https://arxiv.org/html/2502.12771v1) as the reference, including its normalization and dropout; fit PCA and select capacity and training settings within the training stories. The paper supports this candidate, not an established advantage on our reading data or graph-derived features. Record those adaptations. This comparison is planned from the outset, not added only if linear results disappoint.

Decoding asks: **given the activity, which expressed meanings can we identify?** Train direct semantic readouts for brain and LLM activity using the same concept and relation definitions. Targets come from reviewed text annotations, not presumed brain locations. Brain decoding scores measure recovery from the recordings; they are not the participants' behavioural answer accuracy.

Adapt the complete [NEURONA architecture](https://arxiv.org/html/2603.03343v1): spatially identified regional inputs, learned regional feature extraction, concept and region-pair relation scores, and argument-guided symbolic composition trained from semantic answers. Its [released implementation](https://github.com/PPWangyc/neurona/blob/main/src/models/fmri/simple_cnn.py) combines regional projections, convolution, nonlinear activations and pooling, region-identity embeddings, and pair representations. Linear concept heads act on these learned features; they do not make the complete decoder linear. Reuse this architectural reasoning rather than leaving the decoder as an unspecified compact readout.

Adapt atlas indexing, dimensions, and response-window layout to the Deniz recordings; the released code's five-sample input arrangement is not an automatic timing prescription for continuous reading. On the LLM side, retain corresponding semantic targets and compositional operations with an input adapter for model states, without inventing anatomical parcels from arbitrary coordinate groups. Document changes from the reference architecture. NEURONA's fMRI feature encoder is a component of brain-to-meaning decoding, distinct from the meaning-to-brain encoding model above.

The readout may predict labels directly or answer short symbolic queries about specified meanings. Natural-language question generation is not required. Retire the all-prefix answer catalogue and the requirement to recover complete annotation records. A query specifies the question; its inputs must not also supply the unknown answer from the graph.

For a negative or alternative answer, say what establishes it. An unannotated relationship is not automatically false. Use explicit alternatives or a clearly defined observation-level presence task where absence is justified. Preserve unknown cases.

Show what the decoder recovers beyond an appropriate frequency or input-only prediction. If a query supplies partial semantic information, compare against what that query alone reveals; this need not be a separate query-only neural network. Direct multi-label prediction instead needs an appropriate label-frequency reference.

The selected grounding architecture learns concept and relational evidence without concept-to-region supervision. Preserve that training objective and explicit composition while adapting the implementation; this does not require reinstating V1's QA catalogue or its decoder tournament.

**6. Produce concept-associated brain maps, and explain their colours.**

Encoding and decoding serve the same localization interest through different estimates. Neither needs an anatomical answer key for each concept.

An **encoding response map** shows the spatial response that the fitted semantic model associates with a concept or relationship. Retain the temporal response profile and explain any summary across delays. Context-dependent predictions or contrasts can show how a relationship changes the fitted pattern. A contrast involving edited semantic inputs is a model prediction, not a newly recorded human response to edited text.

For nonlinear encoding, derive these maps from predicted responses in specified, supported semantic contexts, reconstructing voxel-space outputs before display. There is no single fixed concept coefficient to read from the network. Keep the source model and context definition attached to each profile, including profiles used in concept RSA.

A **decoding evidence map** shows how spatially identified activity supports recognizing the meaning. When a decoder produces concept scores for particular recordings, preserve those occurrence-dependent maps and explain any overall summary. Do not relabel raw classifier coefficients as activation. For linear readouts, associated activity-pattern estimates can aid spatial interpretation; other grounding scores retain their actual model-based meaning. [Haufe et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24239590/).

Atlas parcels are named groups of measured brain locations. They help display and summarize results and can organize a decoder's spatial inputs. They are not presumed homes of concepts. The existing parcel assignment has incomplete native-voxel coverage: preserve all usable coordinates for encoding and report the coverage of parcel-based analyses. LLM coordinates are not anatomical parcels.

Accompany maps with supporting passages, unseen-story prediction or recovery evidence, and variation across occurrences and participants. Correlated concepts can remain difficult to separate. An associated region need not be a unique or causal storage site, and encoding and decoding maps need not be identical to be informative.

Spatial encoding follows the semantic mapping logic of the [Deniz reading/listening study](https://pubmed.ncbi.nlm.nih.gov/31427396/). Here it is combined with semantic recovery, structured relationships, and brain-model comparison.

**7. Retain RSA as an explicit, standard part of the study.**

RSA asks whether two systems make similar distinctions among the same items. In each system, compare pairs of items' activity patterns to build a representational dissimilarity matrix, or RDM. Then compare the matrices. Brain voxels and model units need not correspond individually. A practical starting choice is correlation distance within each system and Spearman rank correlation between matching off-diagonal entries. Save these metric choices. [Kriegeskorte, Mur, and Bandettini, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2605405/).

Implement two views through the same RSA code:

- **Observation RSA:** compare measured brain patterns and temporally matched LLM patterns for the same story observations. This comparison does not derive both representations from semantic fitting or a learned brain-LLM spatial transformation. It describes organization of the observations; concept attribution comes from the semantic analyses.
- **Concept RSA:** compare relationships among concept-associated profiles estimated separately from brain and LLM activity. Encoding supplies a straightforward profile for each supported meaning; interpretable decoder-derived activity or grounding profiles can supply a separately named view. This addresses the original interest in relationships among concepts within each system.

For concept RSA, retain the source of each profile, its concept identity, supporting occurrences, fit identity, and context summary. Fit each system's semantic model separately; do not first train their distances to agree and present that agreement as an independent result. Common labels identify concepts across systems, while the fitted representation still influences geometry. Label projections for unsupported concepts as extrapolations.

There is no requirement to return to one passage-average vector per concept or the old four-family geometry grid. Context-specific profiles can be retained and a summary shown when its meaning is explained. Independently fitted concept maps and raw observation patterns remain distinct sources of RSA evidence.

Save RDMs with identical item order and explicit item identities, together with comparison scores. An RSA correlation describes agreement in relative dissimilarities, not a percentage of shared concepts or explained brain activity. Assess brain-pattern reliability where the recordings support it so that noisy geometry is not mistaken for a precise disagreement.

RSA and linear translation are complementary. Different scaling or mixing of coordinates can change original distances while leaving information linearly recoverable. Similar broad relationships can also appear despite noisy pointwise prediction. Neither outcome automatically invalidates the other.

**8. Test linear brain-LLM correspondence, compare nonlinear prediction, and explain which meanings they carry.**

The hypothesis is that related semantic information may appear as different combinations of coordinates in the two systems. A linear transformation tests whether a comparatively simple translation connects their recorded activity.

Fit a regularized linear map from temporally aligned, delayed LLM features to brain responses. Train on paired activity without semantic labels directing alignment, and evaluate on unseen stories. This follows the model-to-brain mapping approach of [Schrimpf et al., 2021](https://doi.org/10.1073/pnas.2105646118) and tests our explicit linear-accessibility hypothesis. Include a reduced-rank version to ask whether a smaller shared set of activity combinations is sufficient. Choose regularization and compression using training-story validation, retaining the full linear map as its reference.

Reuse the nonlinear predictor architecture specified in section 5 for a separate fit from LLM features to brain responses, using the same observations and evaluation splits. Report its predictions alongside the linear mappings rather than replacing the linear hypothesis test with whichever fit scores highest. Compare semantic transfer from each mapping using the fixed brain decoder below. Share the implementation and training-fitted brain compression where compatible, not fitted weights across different input feature spaces.

A nonlinear semantic decoder is compatible with a linear correspondence hypothesis: recognizing a concept and translating between representations are different operations. State which representations the translation connects. If learned nonlinear encoders precede a linear map, its linearity claim concerns those learned spaces, not automatically the original recordings. Observation RSA remains available without that learned spatial transformation.

Save transformations on both sides when interpreting a shared space. Compare projections of actual held-out brain and LLM activity, not a prediction with itself. Separately fitted coordinates can rotate or change sign; their numbered dimensions are not automatically identical concepts.

Use the existing semantic readouts to connect correspondence to meaning. Apply a brain semantic decoder trained on the training recordings to predicted brain activity on the unseen story. Compare semantic recovery with recovery from actual brain activity and with the direct LLM readout. Keep the decoder fixed during this test. This asks whether translation carries the meanings the brain readout recognizes without inventing another QA catalogue.

Use semantic records to interpret shared variation and its spatial distribution, with any learned interpretation fitted on training data. Predictable brain-model variation can include presentation or other nonsemantic effects; generic predictivity alone does not identify which concepts it carries.

A successful map supports linearly accessible shared information under these measurements, not identical mechanisms. Retained rank describes a useful predictive representation, not the number of concepts or mental dimensions in the brain. Linear fitting is central because linear accessibility is a hypothesis, not because every pair of latent spaces must have a linear relationship. [Ivanova et al.](https://arxiv.org/abs/2208.10668).

**9. Use shared evaluation and understandable comparisons.**

Keep the ten development stories: train on nine and evaluate the excluded story, repeating across all ten. Choose fitted preprocessing, regularization, compression, and selected layers using training data. The existing three inner story folds are a reusable implementation default. Keep story 11 reserved for final evaluation after development decisions are fixed.

Remove the special stories 01-05 versus 06-10 partitions from the default analysis. They arose from the previous averaging design. Assess story variation through ordinary evaluations. Generalizing to a new story does not by itself establish systematic generalization to unseen combinations; identify actual combinations when making that stronger claim.

Use comparisons with clear purposes: presentation predictors for basic stimulus effects, matched content-only features when asking what relationships contribute, frequency or query-only predictions for semantic recovery, and disrupted timing when asking whether correspondence depends on the actual pairing. Preserve temporal structure in disruptions, for example through documented within-story shifts beyond the response window. Reuse these comparisons rather than expanding them into unrelated experiment grids.

Report participant, story, model, layer, concept, and brain-location variation where relevant before collapsing results into a mean. Participants and stories, or suitable within-story blocks, provide uncertainty units appropriate to the analysis. RSA entries and neighbouring scans are dependent; their large count is not a large number of independent replications.

Repeated scans can characterize reliability where available. They are not additional participants, and reliability from one repeated story is not a universal ceiling. Maps making formal significance claims need an appropriate map-level error procedure; descriptive maps should say they are descriptive.

There is no universal correlation or accuracy threshold that certifies the project or a venue. Explain what was predicted, which distinction was recovered, what an RDM compared, where an association appeared, and how consistently it recurred. Give real examples alongside summaries. These definitions support interpretation without requiring every measurement to be positive.

**10. Implement one integrated V2 workflow.**

Reuse verified data readers, the reviewed annotation archive, frozen-model caches, numerical solvers, anatomical mappers, and execution infrastructure where their meanings match this plan. Replace old measurement choices instead of carrying them forward through defaults.

The following are proposed components, not existing implementation claims:

| Component | Responsibility |
|---|---|
| neurosym/v2/meaning.py and features.py | Readable occurrences, reusable concepts, role-linked features, and direct semantic targets |
| neurosym/v2/timeline.py | Common text, model-state, and fMRI alignment with explicit response timing |
| neurosym/v2/encoding.py and decoding.py | Regularized semantic encoding, the nonlinear brain-response comparison, and adapted NEURONA compositional recovery |
| neurosym/v2/mapping.py | Linear and reduced-rank correspondence, reuse of the nonlinear predictor architecture, and semantic transfer using fixed readouts |
| neurosym/v2/maps.py | Named encoding-response and decoding-evidence maps with spatial provenance |
| neurosym/v2/rsa.py | Shared RDM construction and comparison for observations and concept profiles |
| neurosym/v2/report.py | Joint interpretation, real examples, uncertainty, and traceable figures and results |
| configs/science_v2.json and scripts/run_v2.py | One scientific configuration and executable workflow |

Keep prepared inputs and results under distinct V2 identities. Record the accepted annotation export, corpus/timing identity, target definitions, model revisions, architecture sources and adaptations, preprocessing, and any optional descriptor encoder. Module names and storage layouts are engineering choices; changing the meaning of a target or map is a scientific change that must be explained.

Implement the shared occurrence and alignment interface first, then the specified encoding and decoding architectures, correspondence comparisons, maps, RSA, and reporting against it. Complete the integrated workflow before requesting the full cluster experiment. This is a dependency order, not a sequence of miniature scientific pilots.

Once the interface is defined, meaning/timing work and numerical fitting/reporting can proceed concurrently. Integration uses real annotations and available real data evidence. Missing recordings or metadata must be reported as concrete requirements, not hidden behind dummy observations.

**The completed workflow should deliver five connected research outputs.**

- Concept and relationship recovery from brain and LLM activity, with understandable targets and prediction evidence.
- Brain maps showing where concept-associated responses and decoding evidence occur.
- Observation and concept RSA showing which distinctions are organized similarly or differently across systems.
- Linear correspondence and its nonlinear comparison, showing how activity translates and which meanings survive that translation.
- Supporting predictions, occurrences, fitted profiles, RDMs, spatial identities, and uncertainty that trace interpretations to data.

Interpret these outputs together. Strong RSA with weaker translation, or recoverable meanings with different spatial evidence, can be informative outcomes. The project does not prescribe a winner or one required pattern of agreement.

**Local implementation and cluster execution remain separate practical responsibilities.**

Locally, implement the complete workflow, compile existing annotations and timing, and verify interfaces against saved dataset inspection and available real arrays. Reuse existing caches and metadata. On allocated cluster compute, use full recordings and model-state caches to prepare numerical inputs, fit models, and export evidence. The first full-data execution verifies dimensions, time indices, identities, and anatomical mappings; local checks cannot certify unseen arrays.

The researcher operates SSH and authorizes remote actions. Build environments, populate caches, and run jobs only on allocated compute, never on the login node. For inaccessible resources or long terminal runs, report the concrete requirement and give the researcher the steps. Local user-run commands must use CMD, not PowerShell.

Reuse scheduler, checkpoint, bundle, and concurrency mechanisms after removing dependencies on the V1 scientific inventory. Cost the new workload from its actual operations; the old GPU-hour estimate does not predict this revised experiment. Runtime and launch status remain in the engineering handoff.

Implement this accepted scope as one coherent code path. Preserve the original scientific interests, use simpler measured meanings, and keep encoding, decoding, RSA, maps, and linear correspondence connected through the same observations. Routine engineering choices do not require new approval checkpoints.
