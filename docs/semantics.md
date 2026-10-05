# Semantic experiment compiler

The compiler turns saved Deniz annotations into a common measurement layer for encoding, decoding, and concept geometry. It is ordinary deterministic Python: it makes no LLM calls, does not train a model, and never contacts the cluster. Annotation generation continues separately.

## Run locally (Windows CMD)

From the repository directory, with the existing environment:

```cmd
.venv\Scripts\python.exe -B scripts\compile_semantics.py
.venv\Scripts\python.exe -B scripts\verify_semantics.py
```

The compiler prints the build directory and HTML report path. `data/processed/semantics-v5/latest.json` identifies the latest completed snapshot. The verifier prints a separate verification report under `artifacts/semantic-verification/<build-hash>/`.

Re-run after more annotations have been saved. Each build captures a contiguous accepted prefix per story; it does not require annotation to stop. An existing record changing during compilation causes an error. Later records after a missing unit are withheld because their references may depend on the missing history. Missing records produce missing coverage, never invented graph nodes or observed zero features. `--require-complete` makes incomplete corpus coverage return exit status 2, while preserving the actual compiled output.

Dependencies are listed in `requirements/semantics.txt` and are already present in the project environment. Configuration is in `configs/semantics.json`. The configuration selects the annotation run and pass; the compiler reads archived graphs and the joint-source-v5 protocol through the versioned GraphDelta schema. Current outputs use `data/processed/semantics-v5`; earlier builds remain separate.

## Scientific purpose and implemented choices

This layer supplies explicit semantic variables derived from the same text seen by people and frozen models. It preserves concepts, ordered event arguments, reference, scope, discourse links, and entity updates as distinct measurements. Downstream analyses can test which of these variables explain brain responses, which can be recovered from representations, and how concept/configuration geometry differs between human and model representations.

The compiler preserves annotation labels and optional sense IDs verbatim. It does not silently merge synonyms, invent senses, or correct questionable relations. Schema validation establishes structural consistency; it does not establish that a semantic judgment is correct. Both protocol provenance and semantic review status accompany each source.

### Prefix graphs and observations

`history.jsonl` contains ordered graph deltas. `SemanticDataset.state_at(story_id, source_id)` replays only the available prefix. `state.json` contains the final captured ledger and active property assertions. Assertions and retractions remain in the ledger; an unresolved update never silently overwrites a resolved assertion. Properties are tracked by entity, attribute, and value, allowing genuinely multivalued attributes.

Feature rows describe **new annotation observations at each unit endpoint**. Persistent entities are not counted again simply because they remain in memory. This distinguishes accumulated graph state from newly observed semantic information.

### Semantic feature groups

| Group | Measurement | Interpretation |
|---|---|---|
| L | Mention surface and form | Optional lexical comparison |
| C | Locally attested entity concepts and predicates | A pronoun does not automatically become a locally observed antecedent concept |
| B | Predicate-role, role-local-filler, local binding, directly grounded typed binding | Local event structure |
| BR | Typed/referent bindings requiring reference or a previous event | Binding whose interpretation uses reference |
| GB / GBR | Role-filler pairs, without a predicate; local / reference-dependent | General semantic roles |
| PB / PBR | Predicate-role-filler triples; local / reference-dependent | Predicate-specific role binding |
| R | Resolved concepts, reference anchors, mention links and distance classes | Reference resolution |
| S | Polarity, event status, tense and scope-parent links | The status of a proposition, distinct from world truth |
| D | Directed event/discourse relations | Relation type and ordered endpoints |
| U | Entity property assertions and retractions | State changes |

Graph schema 2 adds literal values, nested contexts and identity links. C includes
literal observations; event-role queries can target literal values. S retains full
nested scope profiles, including attribution and context holders. Scope-choice
queries select the whole recorded profile, so accepting one of several statuses
cannot incorrectly count as recovering their combination. Candidate profiles are
shared across events in the available prefix and include canonical alternatives;
query-only controls receive the identical profile catalog.

Discourse and identity questions include context selectors where needed. These
selectors identify a context by kind and source position, like an event's trigger
selector; source evidence text and answer indices remain private. A reported
identity claim is never silently promoted to narrator-level identity. Explicit
unscoped identity revelations expose prefix-available equivalence classes without
merging stored entities. `state.json` separates active narrator-level properties
from contextual properties; retractions apply within their recorded context.

The GB/PB comparisons are separately selectable encoding groups. Existing B/BR
remain available for the earlier combined feature definition. Avoid including
both the combined and decomposed groups and interpreting them as independent
variance sources: their feature content overlaps.

Each feature has evidence spans, graph identities, and declared dependencies. B never receives an earlier mention's surface as though it occurred locally. Unresolved reference bindings do not receive resolved typed features. Reference-dependent bindings remain separate from local bindings so their effects can be assessed separately.

Shared factors and exact event conjunctions have separate routes. The default vectorizer uses shared factors; exact conjunctions require `include_exact=True`. Even shared features include structured pairs such as predicate-role: this is a sparse interpretable representation, not a learned graph encoder. Rare/unseen labels are reported as out of vocabulary rather than silently mapped to a new test-derived dimension.

`SemanticVectorizer.fit(dataset, train_stories, groups=..., include_exact=False)` fits the vocabulary on explicitly selected development training stories. Call it separately inside every training fold. The globally published feature catalog is descriptive metadata, **not** the fitted numerical vocabulary. `transform` returns unit features and coverage; `design` sums them into the verified TR mapping and applies the existing causal FIR delays. Standardization, feature selection, dimensionality reduction and regularization are tasks for the downstream training folds and are not fitted here.

### Queries and graph-side targets

Queries use a typed AST for concept occurrence, event roles, reference, status, polarity, directed relations, binding verification, and compositions through event arguments, scope parents, or event relations. Multi-answer role/relation queries preserve all supported answers. Candidate orders are reproducible and independent of the answer index.

An absent edge is not evidence of falsehood. Role-swapped candidates remain unlabelled unless the graph actually supports them. Concept-presence queries currently contain supported positives only and are diagnostic, excluded from scored decoder examples. Binding polarity contrasts concern the selected proposition **within its recorded scope**; they are not statements about what happened in the real world. They must be reported separately from role-swap judgments: success on polarity contrasts alone is not evidence of compositional role understanding.

Public query inputs contain selectors, operations and candidate descriptors. Candidate descriptors expose available concept types, explicitly introduced names, and mention/event positions. They therefore carry meaningful priors. Any query-only comparator must receive exactly the same inputs, and a graph-generated query distribution must not be interpreted as unrestricted text comprehension. Direct role queries are generated for annotated roles; this does not supply labels for arbitrary absent roles.

`SemanticDataset.decoder_inputs(query)` returns only the AST and candidate descriptions. Answer indices, graph identities, evidence, provenance and review flags are stored separately. `decoder_examples(story_id, reviewed_only=False, families=None)` returns inputs and targets as separate fields and balances query families within source units. Several queries from one unit do not become independent human observations; later uncertainty estimation must respect stories, people and shared BOLD support. `reviewed_only=True` excludes unreviewed labels and can correctly return no examples.

The implemented graph-side oracle is a target-construction mechanism. It is not the trained neural decoder and does not prove human concept access. Scoring eligibility means eligibility **against the current annotation targets**, with known unit timing; it does not mean human-adjudicated ground truth. Explicit overlapping annotation uncertainty masks scoring. Causal interpretations and missing predicate senses enter review queues without silently changing their labels.

### Occurrences and compositional splits

The occurrence registry links concepts, predicates, roles, reference links, configurations, discourse relations and updates to source units, available text prefixes and timing. Locally attested and reference-resolved concept occurrences remain distinguishable. Counts are distinct source counts, not independent sample counts.

`composition_holdout(heldout_keys, train_stories, test_stories)` constructs strict **type-level configuration** exclusions. It purges whole training stories containing a held-out configuration so it cannot remain inside a later text prefix or delayed neural sample. Incomplete training stories are excluded because their unannotated suffixes cannot certify absence. A test configuration is eligible only when its grounding is resolved and its constituents occur in retained training stories. The method returns unavailable/unsupported cases explicitly; it does not guarantee the corpus can support every proposed compositional comparison. Configuration keys preserve status and polarity. Relation-path and reference-chain holdout protocols are not asserted by this method.

The registry is the input to later geometry estimation. It does not yet compute measured neural signatures, decoded concept weights, model latent prototypes, or encoding-implied brain signatures. Those remain distinct estimators rather than interchangeable compiler outputs.

### Timing and shared support

Sources carry unit/model row identities, text spans, the verified raw feature bin, and delayed raw/trimmed fMRI response rows. Features become available at the annotation unit endpoint. The compiler does not infer unknown timestamps. Missing timing and uncompiled suffixes are masked in encoding designs, including FIR support. Downstream model/brain comparisons must intersect their support with these masks and the actual extraction/response coverage.

Decoder support is an **offline delayed BOLD window**. It can contain responses to text presented after the semantic endpoint. This must remain explicit in interpretation; the compiler does not turn delayed fMRI into a strictly online human prediction. Query eligibility requires the full declared delayed window to lie within the retained response rows.

## Output contract

Each build is under `data/processed/semantics-v5/builds/<content-hash>/`:

| Output | Purpose |
|---|---|
| `identity.json`, `complete.json` | Annotation/corpus/alignment/code/package identity and artifact hashes |
| `report.html`, `report.json` | Actual coverage, query eligibility, protocol counts and review flags |
| `feature_catalog.json`, `candidate_sets.json` | Feature definitions and deduplicated candidate descriptions |
| `occurrence_catalog.json` | Source/story coverage by semantic label and view |
| `splits.json` | Existing story-disjoint nested folds and final held-out story |
| `worked_example.json` | One actual annotated passage with its graph, features, queries and review items |
| `stories/<id>/sources.jsonl` | Identity, timing and configuration joins |
| `stories/<id>/features.jsonl` | Sparse counts and evidence |
| `stories/<id>/queries.jsonl` | Public queries referencing candidate catalogs |
| `stories/<id>/answers.jsonl` | Private targets, eligibility, review flags and source weights |
| `stories/<id>/oracle.jsonl` | Private graph-identity mappings for graph-side verification |
| `stories/<id>/occurrences.jsonl`, `review.jsonl` | Occurrence support and annotation review requests |
| `stories/<id>/history.jsonl`, `state.json`, `timing.json` | Replayable history, accumulated state, and support masks |

Build artifacts are immutable after publication. `SemanticDataset` verifies their hashes by default. Cached compilation reuses an identical snapshot; more accepted annotations or changed compiler code create another build. Retain the build identity with downstream fits. The verifier uses the actual corpus, captured graph records and numeric semantic features to check identities, prefix boundaries, query/target joins, graph-side answers, directionality, vocabulary isolation, source weights, and FIR values/masks. It uses no synthetic data and makes no claim to semantic annotation accuracy.

## Next integration boundary

No transfer or cluster job is required for this compiler. Annotation can continue concurrently. Once the intended annotation coverage and review are ready, recompile the final snapshot and use its interfaces for the encoding/decoding/geometry implementation. Those later stages will consume the actual subject response arrays and frozen-model extraction artifacts; they require trained readouts or estimators where the scientific question calls for them. The compiler neither substitutes for these analyses nor depends on the currently blocked GPU extraction to finish.
