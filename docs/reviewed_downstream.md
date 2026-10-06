# Accepted annotations: integrated downstream implementation

The input is the accepted independent-review build
`5cd83e0779bb5440da4761cbaf26dd9791966212bbc2bc787c1efd8669524751`.
Review is closed. Its archive, annotation code and review protocol are not edited
by this implementation. Statements in the archived review about pending human
adjudication describe its original provenance; they do not reopen review.
Acceptance is also not an independently measured annotation-accuracy estimate.

`configs/semantics.json` pins the export and full hash, rather than following the
annotation directory's mutable latest pointer. Output goes to immutable builds
under `data/processed/semantics-reviewed`; `configs/analysis.json` uses that
destination. Previous semantic builds are retained separately.

## Scientific purpose and representation

The scientific question remains how concepts, role bindings and reference or
discourse structure are accessible in human recordings and frozen language-model
states, and how their representational organization compares. Encoding,
answer-supervised decoding and geometry remain distinct branches. The compiler
makes no LLM calls and learns no parameters.

The compiler consumes all 1,217 units across 11 stories. Every original graph
record, evidence span and qualifier value tree remains available in the compiled
history and copied provenance. Reversible aliases come only from the accepted
interface. Predicate-specific role maps preserve argument order, including
repeated roles; explicit relation reversals retain their original endpoints.
No label-similarity heuristic creates new equivalences.

The expression layer retains local polarity, context DAG paths, attribution,
full qualification, scoped identity edges and references to endpoint expressions.
There is no global identity union or actual-world projection. An entity's
introduction is a discourse observation, not proof that it exists. A relation to
an earlier intention does not inherit that intention's agency into a later event.

Explicit reviewed scope cases take precedence over independent application of
raw fields. These include single modal denial, conjunction inside impossibility,
degree-limited negation, report continuation, genuinely nested speakers, local
factivity, compound conditional antecedents and prospective clarification.
Consumed fields remain in the original record. Other operator orders remain
typed symbolic expressions with explicit unresolved interpretation; they are
not flattened into Boolean facts. Qualifiers on contexts include the target
context itself, and empty attachment scope cannot remove the target's scope.

Graph uncertainties and the 29 accepted review questions retain their affected
IDs and availability. They withhold dependent answers, not every question in
the unit. Later observations can add supported information without rewriting an
earlier prefix. A null argument or missing assertion never creates a negative.

## Three clocks

| Clock | Representation and use |
|---|---|
| Text exposure | Original word spans and alignment. Unresolved word times stay unresolved. |
| Interpretation availability | Full annotation-unit endpoint, including later qualifiers, identity claims and clarifications. An earlier evidence span never backdates the interpretation. |
| fMRI measurement | Ordered delayed response rows at the existing FIR lags, with their sample times and trim mapping. These samples are offline measurements and can include effects of later nearby text. |

The four unresolved annotation endpoints remain unavailable for timed fitting.
The frozen-model input is the prefix at the unit endpoint. It does not receive
future stimulus text or a decoder question during extraction. fMRI and model
observations consequently have different temporal integration; their comparison
does not establish exact online localization.

## Encoding and decoder targets

Encoding retains L, C, B/BR, GB/GBR, PB/PBR, R, S, D and U: surface information,
concepts, binding and reference-dependent binding, general and predicate-specific
binding factors, reference, scope, discourse and state updates. Scoped binding
and relation features supplement shared factors. Exact configurations remain a
separate feature route for compositional holdouts. Opaque qualification still
contributes typed dimension features and retains its complete symbolic value.

Uncertain feature contributions are withheld, and their group's affected
temporal support is marked unavailable. Each encoding comparison explicitly
declares the union of groups required by its conditions. Those conditions share
that comparison's mask across training, validation and test, including timing,
word alignment and all declared frozen-state availability. Unrelated uncertain
groups do not remove rows. The earlier all-group intersection remains an explicit
robustness policy. Coverage and exact row identities are reported per story;
withheld contributions are not treated as observed zeros. See
[encoding_support.md](encoding_support.md) for the policy and execution interface.

All feature vocabularies, frequency thresholds, scaling, parcel-local PCA,
decoder embeddings, ridge tuning and stopping decisions are fitted within the
appropriate training fold. Story 11 is excluded from fitting and selection.
Compiling its records and checking their structural integrity are preprocessing,
not model selection. Existing story-disjoint nested partitions are retained.

| Decoder task | Annotation-supported target |
|---|---|
| Concept | Recorded concept, predicate or literal at a source anchor; not binary world existence |
| Role | Filler of a specified ordered role |
| Reference | Recorded referent of an anaphoric mention |
| Scope | Complete symbolic operator profile, separate from lexical predicate/argument content |
| Polarity | Recorded local polarity, explicitly separate from composed proposition truth |
| Binding | Ordered assignment among the original and role-swapped alternatives |
| Relation / identity | Type of the recorded directed or scoped link |
| Property / qualifier | Recorded symbolic attachment value |
| Composition | Explicit event-content-to-role, context-to-holder or linked-expression-to-role traversal |

Candidates use available prefix records or declared task syntax. Binding swaps
are assignment alternatives, not asserted false events. Observationally identical
public descriptors are deduplicated; decoders are never required to guess an
inaccessible graph ID. Conflicting indistinguishable questions remain unscored.
Candidate descriptors are pooled on disk without changing the candidate list
seen by a decoder. Queries and candidates may carry priors: all comparator
readouts receive exactly the same public information.

The existing query-only, linear, MLP and structured readouts consume these tasks.
The structured readout learns unary local-site evidence and typed ordered-pair
evidence; ordered composition propagates support through those operators.
Assignment scores combine ordered role evidence. This is answer supervision
from accepted text annotations, without concept-to-parcel ground-truth labels.
Language models remain frozen; fitted readouts are explicitly learned models.
Successful heldout prediction plus specificity and stability controls can
support an association between a distinction and decoder evidence at particular
parcels. They do not establish a unique biological location or causal necessity.

## Human–model comparisons

Native, grounding, latent and encoding-implied geometry remain separate views.
Grounding traces support unary concept/predicate/literal/scope profiles and
ordered-pair binding, reference, discourse, identity and attachment profiles.
Whole configurations use native, latent or encoding-implied signatures; their
component bindings have ordered-pair maps. Missing trace support stays missing.

Geometry has two explicit scope policies: `--scope-mode pooled` compares labels
while retaining the constituent scope observations in `scope-coverage.json`;
`--scope-mode scoped` additionally requires matching symbolic scope. Pooled
geometry marginalizes contexts and does not reinterpret them as root facts.
Scoped matching may have less repeated support, which is a measured coverage
limitation. Human/model comparison matches semantic items and compares distances
within each independently fitted representation, never raw latent coordinates.
Model coordinate groups are computational groups, not brain regions.

The existing heldout prediction controls, disruption/retraining controls,
source/family weighting, anatomical-map stability and semantic-label permutation
interfaces remain available. No predictive accuracy or biological conclusion is
claimed by local compiler verification.

## Verification and execution

`scripts/verify_reviewed_semantics.py` checks the full real corpus against the
accepted hashes, normalized records, scope sidecars, explicit interpretations,
uncertainty targets, all public query envelopes and timing support. It checks
vocabulary fitting in every outer partition and story 11 exclusion. Its receipt
is `artifacts/reviewed-semantics-verification.json`.

`scripts/verify_analysis.py` exercises real compiled tasks and all three
composition routes across all four readouts with forward/backward checks.
It also checks serialized ordered grounding profiles, the existing ridge
algebra, local projection, timing and nine participant atlas contracts. Numerical
inputs are actual released textual features, explicitly not fMRI or substitutes
for modern hidden states. Its receipt is `artifacts/analysis-verification.json`.

`scripts/verify_encoding_support.py` independently reconstructs availability
from all real uncertainty records, checks actual paired designs across the whole
corpus, reproduces the all-group mask and rejects unmatched comparison support.
Its detailed receipt is `artifacts/encoding-support-verification.json`.

In local Windows CMD, the complete compilation, verification and packaging path is:

```cmd
cd /d C:\Users\pigby\neurosymbolic_coognitive_linguistics
call scripts\prepare_analysis.cmd
```

Any failed check stops packaging. A successful package is
`artifacts/analysis-source.zip`. It contains isolated analysis code, this semantic
build, accepted-input provenance, atlas/mapping artifacts and verification
receipts. It excludes annotation runners, prompts, credentials, full fMRI and
model weights. Its installer checks hashes and refuses conflicting shared data.

After local verification, the user can transfer from CMD:

```cmd
call scripts\upload_analysis.cmd
```

The upload performs file transfer only. Cluster setup and scientific fits must be
submitted by the user in their existing authenticated session. Setup and all
caches remain on project storage and run on allocated compute nodes. This work
does not contact the cluster or assume its earlier scheduler issue is resolved.
Full fMRI, modern hidden-state fits and large inference runs belong there. The
analysis preflight identifies unavailable observations rather than substituting
data or authorizing a missing model condition.
