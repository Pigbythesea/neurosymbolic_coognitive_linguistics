# Scientific measurement review from first principles

**Date:** 2026-10-08. **Status:** Findings and proposed scientific corrections; no production method change or new cluster launch is authorized by this document.

This review revisits the definitions behind [FIRST_DAY_RESULTS_REVIEW.md](FIRST_DAY_RESULTS_REVIEW.md). Its numerical summaries remain descriptive records of the completed experiment. Several interpretations require greater caution because the present implementation does not fully operationalize the original scientific question. Optimizer diagnosis alone is insufficient.

The intended question remains: **Which distinctions in linguistic meaning are available in human fMRI and frozen language-model representations; how are these distinctions organized; and what reproducible neural associations support them?** This does not require a new decoder architecture to outperform every baseline. Architectural superiority, direct representational comparison, and anatomical interpretation are separate claims.

The [human reference](Structured_Meaning_Brain_LLM_Human_Reference.md) illustrates the distinction with swapping two participants in a lending event. That example requires preserving occurrence identity as well as abstract entity type. A sound operationalization must also define when the distinction becomes available and what neural observations measure it.

**Verified findings**

The [local audit script](../artifacts/review_measurement_contracts.py) reads the actual accepted semantic build and saves [its evidence](../artifacts/measurement-contract-review-2026-10-08.json). It runs no fits and generates no synthetic observations. The source build is `9bce4af5463a57683ed375a8b0fdcbcb83620bce923624d7075bd0c4f4ac847f`.

| Decision | Verified implementation/evidence | Scientific consequence |
|---|---|---|
| Annotation unit defines neural sampling unit | Every unit's decoder uses four delayed response rows after the full unit endpoint. Native geometry averages those rows. | This measures an endpoint-associated state, not necessarily the local response to every item mentioned earlier in the unit. |
| Unit durations vary greatly | Story 01: 129 units, mean 4.71 seconds. Story 06: 56 units, mean 12.68 seconds. The longest story-06 unit contains 372 words over 100.32 seconds. | Questions from different units impose very different temporal/memory demands. A timestamp-valid association can still be a poor measurement of the desired event. |
| Symbolic encoding deposits updates at unit endpoints | The feature vectorizer bins each source's complete feature count vector at `raw_feature_bin`; model encoding uses word-level features. | Feature families have different temporal definitions. Paired response masks alone do not make their information timing comparable. |
| Structural features are exact categorical conjunctions | A predicate/role/filler combination gets an independent feature key; unseen test keys contribute no column. The route name `factorized` does not implement learned factorization. | This is largely a repeated-category test, not a general test of transferable compositional semantics. |
| Most test conjunction occurrences lack training support | Combined PB/PBR seen occurrence mass is 14.13% for story 01 and 14.81% for story 06. | Approximately 85% of these test feature occurrences cannot express their exact conjunction through a trained coefficient. These counts precede contrast-specific temporal masking. |
| Typed fillers omit individual occurrence identities | Encoding/role geometry use abstract filler descriptions. Decoding alternatives retain token anchors. | A swap between two different people both typed `person` can change the decoding answer while leaving typed binding features unchanged. The branches can measure different distinctions under the same word “binding.” |
| Query difficulty depends heavily on compilation | An actual binding query contrasts person-as-driver/campus-as-place against campus-as-driver/person-as-place. An actual role query asks for the experiencer of nostalgia among 356 prefix candidates. | Some choices are answerable from semantic plausibility; others require exact discourse retrieval among many heterogeneous alternatives. Aggregate accuracy is not a single measure of semantic recovery. |
| Query descriptors are learned from the available training corpus | Descriptor tokens use trained embeddings, numerical anchor features, and an unknown token for unseen vocabulary. | Decoding jointly learns descriptor semantics, neural access and task behavior. Failure is not an isolated test of whether the neural representation contains the target distinction. |
| “Linear” is a restricted trained comparator | A learned observation projection to 64 coordinates combines with learned query/candidate embeddings and a prior branch. It is linear in observations conditional on the query, but uses joint neural training and a rank bottleneck. | Its failure does not establish failure of the best regularized linear mapping between fixed representation spaces. |
| Grounding starts from a specified anatomical hypothesis | Brain observations use 200 sites, four time samples, up to eight local PCs; role evidence uses ordered site pairs. | The map is an inference made by a spatially structured readout. It is not a directly measured concept-activation map or connectivity matrix. |

The longest unit, `story_06_u0007`, runs from approximately 170.256 to 270.578 seconds. Its selected fMRI sample times are 273.614, 275.619, 277.623 and 279.628 seconds. This is appropriate only for a question about the state measurable at that later endpoint. It cannot automatically be interpreted as event-local evidence for everything read over the preceding 100 seconds. Working memory and narrative persistence remain possible signals; they are different scientific objects from a local evoked response.

The train-supported encoding mass is:

| Feature group | Story 01 | Story 06 |
|---|---:|---:|
| C: exact concept/predicate/literal keys | 37.94% | 45.12% |
| BC: matched unbound constituent keys | 62.93% | 59.87% |
| PB: non-reference predicate/role/filler conjunctions | 4.97% | 3.29% |
| PBR: reference-channel conjunctions | 22.06% | 22.65% |
| PB and PBR combined | 14.13% | 14.81% |

These are counts over included compiled feature occurrences in the heldout stories, using vocabulary membership from the other nine development stories. They are not estimates of the fraction of fMRI information lost, nor the final paired response-row coverage. They establish a substantial representational-support limitation before any optimizer is considered.

An actual identity-collapse example occurs in `story_06_u0023`: the query asks who is speaker versus recipient of `ask`. Both people have the same abstract concept/kind/denotation, but different token anchors. Swapping their roles changes the public candidate assignment. Removing occurrence coordinates makes the two assignments identical at the typed-filler level used by the encoding representation. This does not erase every role distinction; it erases precisely some within-type identity distinctions central to “who did what to whom.”

**What the geometry statistic measures**

For native geometry, the code gives every eligible semantic item in a source the same source observation vector. It averages source vectors within item/story, then averages stories equally. The brain source vector averages the selected delayed fMRI samples; the model source vector is the frozen prefix endpoint state. Training-fitted standardization and optional presentation residualization precede the aggregation.

Cosine distances between item signatures form an RDM. Spearman correlation between two RDM upper triangles measures agreement in the ordering of item-pair distances. A value of 0.5 does not mean 50% of neural information or 50% of meanings is shared. RSA is an established method, but its numerical scale depends on the measured patterns, items, distances, noise and controls; there is no general publication threshold. [Kriegeskorte, Mur and Bandettini, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2605405/).

The current signatures are **item-conditioned passage averages**. They do not separate the item's contribution from other concepts, events or contextual properties of the same passages. This is a useful descriptive object, but it must be named accurately.

A mathematical reason for concern is shared aggregation. If `W` records which sources contribute to each item, the two signature matrices have the forms `W Y` and `W X` on matched source support. Even for independent isotropic source vectors, their expected within-system Gram matrices share a component proportional to `W Wᵀ`. Temporal overlap and serial correlation add further shared structure. This is an analytical explanation of a possible confound, not a simulation or a claim that it explains all current alignment.

The existing co-occurrence view uses one-hot source vectors and addresses an important part of this issue. Its positive partial correlations do not remove all effects of source support, shared temporal sampling, nonlinear rank relationships or other context variables. The earlier partial-correlation result should therefore be described as survival of one adjustment, not isolation of a shared semantic code.

Stories 01–05 and 06–10 are complementary context partitions. They contain different item inventories, frequency distributions, unit durations, narrative content and potentially different signal quality. Standardization/residualization is also fitted on the opposite training partition. Their coefficients cannot be compared as if they measured identical items under identical conditions. The difference is not evidence of a cognitive change at story 06.

**What the decoder is asked to learn**

The LLM annotator authored semantic graphs. Deterministic code constructs the public question programs, candidate catalogues and acceptable answers from those graphs. The participant was reading, not answering these generated questions. The probe receives the question program and candidate descriptions, plus the selected brain or frozen-model observation; the private answer and original passage are withheld.

The prior receives only the public question/candidates. It measures what these already reveal through types, predicates, positions, prevalence and other regularities. Its strong performance is not by itself evidence of answer leakage. It does show that chance accuracy is an inadequate reference for neural contribution.

Linear, MLP and structured readouts answer different method questions: simple accessibility, accessibility with a flexible mapping, and utility of a particular spatial/compositional inductive bias. They are not three mandatory definitions of understanding. A project primarily about shared latent representations need not make a new QA executor its central empirical test.

If model and brain observations are linear measurements of a common retained latent state, a linear mapping can relate them. The existence of shared information alone does not guarantee linear measurement, invertibility or preservation through temporal averaging and PCA. Ordinary cosine RSA is also not invariant to every invertible linear transformation. Mapping choice must follow the scientific claim. [Ivanova et al., 2022](https://arxiv.org/abs/2208.10668).

**Grounding and absence of anatomical ground truth**

A unary grounding profile contains learned site scores. A role profile contains learned ordered-pair scores weighted by the two argument-routing distributions. With 200 supported parcels, the latter has 40,000 entries. The reported map correlation mean-centers and correlates corresponding entries for the same semantic item across fits, then averages items. Near-zero role-map correlation means weak reproducibility of this fine-grained pair-score object; it is not a near-zero brain/model RSA or direct evidence that no coarser anatomical association exists.

NEURONA reports approximately 70.4% overall accuracy on its two QA datasets and approximately 0.85 grounding consistency. Its consistency statistic concerns repeated region selections, not Pearson correlation over our 40,000 pair scores. Its paper explicitly limits the maps to model-dependent decoding patterns and does not claim established neural representational compositionality. These numbers are not interchangeable benchmarks. [NEURONA](https://arxiv.org/html/2603.03343v1).

No anatomical ground truth permits competing models of localization. It does not make every model-internal score a valid localization. Different maps can support similar predictions; a constant arbitrary map can even be perfectly consistent. Interpretation therefore needs evidence that maps track the relevant observation and semantic distinction, at the anatomical resolution actually claimed. Decoder parameters and neural activation patterns are distinct objects, even for linear models. [Haufe et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24239590/).

**Consequences for the next scientific decision**

1. Separate annotation batching from measurement timing. Define evidence-supported semantic availability, local neural exposure and any intended longer-term memory target explicitly. Preserve causal information boundaries; do not simply backdate an annotation that relies on later text.
2. Decide which object each branch measures: concept type, individual occurrence, role-conditioned type, instance binding, scope operator or discourse state. Preserve identity when the contrast requires it. A transferable encoding of structure needs a defined way to represent new combinations; exact keys remain a useful baseline.
3. Design answer alternatives around the intended distinction. Type-compatible binding alternatives test a different question from swapping a person with a campus. Stable compositional targets test a different question from choosing an exact annotation profile among many long, heterogeneous profiles. Keep full graph provenance without making every annotation field a prediction target.
4. Make a regularized direct representation mapping and interpretable semantic contrasts central where they address the latent-representation question. Evaluate the structured executor as a separate hypothesis if its spatial/compositional assumptions are part of the scientific contribution.
5. Define item geometry and its comparison against constituent/context explanations on genuinely matched support. Use independent contexts or appropriate crossvalidated estimates where the claim concerns stability beyond the same passages; preserve meaningful context dependence where that is the target.
6. Specify anatomical claims at a defensible scale. Unary evidence, parcel-pair scores, network-level marginals and encoding selectivity answer different questions. Do not select a resolution merely because its result looks favorable.
7. Report absolute prediction quality alongside increments, and use appropriate spatial summaries. The original Deniz study used voxelwise semantic prediction and tuning; our whole-cortex mean increment is not numerically comparable to a reliable-voxel absolute correlation. [Deniz et al., 2019](https://pubmed.ncbi.nlm.nih.gov/31427396/).

These proposals preserve the scientific scope while changing how it is measured. The accepted graph archive, real stimulus timing, raw recordings, frozen-model extraction and much of the infrastructure remain useful. An integrated measurement revision should precede further production expansion. Existing results remain evidence about the current implementation, but should not be used to reject structured neural meaning or certify stable semantic localization.
