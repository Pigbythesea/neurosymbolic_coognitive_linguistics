# Scientific measurements and execution definitions

This records the measurement definitions, updated for protocol 2 on 2026-10-07. The primary
scientific scope is consolidated in [SCIENTIFIC_STATUS_HANDOFF.md](SCIENTIFIC_STATUS_HANDOFF.md): concepts,
binding, reference, scope and discourse in human recordings and frozen model
states, studied through encoding, decoding and representational geometry.
These definitions clarify what each implemented measurement does; they do not
prohibit new analyses or require another annotation round.

## Encoding: distinguish content from its arrangement

`C` retains its original meaning: newly introduced entity/literal content and
current event predicates. A later reference to an old entity does not itself
add that entity to C. Its existing comparisons remain usable, but adding a
binding feature to C can add both constituent content and structure.

`BC` is the new unbound binding-constituent control. For every argument incidence
used by PB/PBR, it emits three separate categorical features: the predicate,
the role, and the filler descriptor. Each retains the same local/reference
channel as PB/PBR and the same incidence count and uncertainty eligibility.
These are summed independently within the unit: no feature binds a predicate
to a role or a role to a filler. Old referents therefore enter this baseline
whenever they enter the structural features. Repeated arguments retain their
multiplicity. No role position, graph ID or event-specific bag identifier is
inserted into BC. This is an unbound marginal representation, not a new graph.

The main structural contrast uses the same `matched-binding` support profile
for both conditions:

| Condition | Groups |
|---|---|
| Unbound baseline | presentation, C, BC |
| Predicate-role-filler structure | presentation, C, BC, PB, PBR |

The analogous `matched-general-binding` adds GB/GBR. The broader
`matched-scoped-binding` adds B/BR, which also carries scope information: its
gain is not an isolated role-assignment effect. BC matches predicate, role and
filler content; it does not claim to control every lexical or contextual
variable in the narrative. Reference, scope, discourse and state comparisons
remain explicitly named feature-package comparisons against C.

The full-corpus verifier independently projects PB/PBR columns onto their
three marginals and checks exact equality with BC for every unit. It also checks
matching uncertainty eligibility and baseline/augmented FIR row identities.
Training-only vocabulary coverage is reported by the existing vectorizer.
PB/PBR remain exact categorical conjunctions: seeing their separate constituents
does not create a new conjunction coefficient. The historical `factorized`
route name is an implementation category, not evidence of learned compositional
generalization. The separate decoder composition tasks and composition holdouts
retain their own definitions.

## Geometry: match retrieval queries to items

`--view latent` now uses only the queries linked to the semantic occurrence.
The compiler records these links as analysis provenance outside the public
decoder input. It matches the task, public anchor and operation, including the
argument position for role retrieval. Introduced concepts use their concept
query; resolved anaphoric content uses its reference query; scopes, assignments,
links and attachments use their corresponding retrieval query. If no legal
query was compiled, or no eligible trace was captured, that signature remains
unavailable. There is no substitution of a neighbour's query.

All occurrence links survive source/item deduplication. Repeats are averaged
within each query, then distinct matched queries receive equal weight within
the source/item. Source means are averaged within each story, then stories
receive equal weight. Selection never depends on whether the decoder answered
correctly. `latent-coverage.json` records selected and missing query IDs.

`--view latent-passage` retains the passage-level interpretation: it averages
all captured queries in the unit, so co-occurring items share the same vector.
Both views use explicit repeat-then-query averaging. The previous implementation
deduplicated equal numerical vectors before averaging; this accidental
value-dependent weighting is not retained.

Query matching does not alter the decoder architecture. Structured latents
depend on the query anchor or composed execution; two roles at the same event
anchor can still share a latent. Linear/MLP latents are source projections and
are query-independent. These are learned retrieval-context measurements, not
guaranteed item-separable embeddings. Ordered grounding profiles remain the
view for explicit role-specific site-pair evidence. Native, grounding and
encoding-implied geometries remain available alongside both latent variants.

## Concrete experiment defaults

[configs/experiments.json](../configs/experiments.json) specifies the full nine
participant panel, ten development outer folds, final story holdout, three
readout seeds, all five pinned models, encoding contrasts and readout controls.
The primary model comparison uses each model's final block output. Descriptive
layer profiles use fixed quarter-depth positions, resolved to concrete indices
from the lock file. They are reported as profiles, not selected using story 11.
Selecting a best layer would require an additional training-only selection
procedure; the current defaults do not silently perform that selection.

The four readouts are query-only, linear, MLP and structured. Heldout mismatched
observation controls accompany fits; separately retrained structured nulls
address learning under disrupted observation/target pairing. All controls keep
the registered task and candidate definitions. Per-task accuracy, chance,
accuracy minus chance and NLL accompany story summaries. Seeds/repeats do not
increase the participant count.

Protocol 2 applies the full readout/control panel to the nine humans and five
final-layer model observations. Intermediate layers use fixed linear probes
(seed 11), native geometry and encoding profiles. Three whole-story inner
folds select nonlinear settings once at seed 11 for each outer condition;
refits at seeds 11/29/47 share those settings. Linear probes use 20 epochs at
0.0003. Structured nulls reuse matched settings and are explicitly fixed-procedure
correspondence ablations. Encoding uses one union of 25 weight candidates
(equal weights plus eight draws each from seeds 11/29/47), not three replications.
Decoder minibatches contain 16 sources and optimize the declared normalized
story/source-weighted objective. These changes alter selection/training protocol;
they do not claim identical old AdamW trajectories.

Heldout primary structured fits compare query-route-selected site replacement
against equal-count random sites using the same nonoverlapping donor. Context
parents retain full traces for anatomical stability and the three learned
geometry views. Source co-occurrence geometry and actual semantic-support
reports accompany the comparison panels. These are decoder-reliance and
conditional-geometry controls, not biological interventions or full confound removal.

Ordinary outer-fold geometries describe a heldout story. Context stability uses
two additional fixed development partitions, holding out stories 01–05 and
06–10 in turn. Fit the required readout on the complementary stories; do not
pool raw coordinates across separately fitted models. Item signatures require
at least two heldout stories for this context analysis. Report actual common
item support, including when the half-story reliability calculation is
unavailable. Whole configurations have no separate grounding view; use their
component role profiles or their native/latent/encoding-implied views.

The plan retains pooled and fully scoped labels, raw and training-residualized
native geometry, RDM association and cross-participant anatomical-map stability.
Permutation tests concern semantic-label association within the declared view.
Population intervals require paired participant/story cells; final-story
results alone cannot estimate variation across new stories. Repeated anatomical
association supports decoder evidence, not a unique biological implementation
or causal necessity. Unit-end interpretation and delayed BOLD remain different
clocks; these corrections do not backdate annotations to word onset.

`run_analysis.py plan` resolves these definitions, model revisions/layers and
partitions to `artifacts/experiment-plan.json` without executing fits. This is a
reviewable execution definition, not a scheduler and not a prohibition on
scientifically justified amendments. Analysis run identities save its hash.
The default CLI still permits explicit alternative views/partitions; record
their rationale and distinguish them when reporting.

## Local completion and cluster boundary

The accepted annotation archive is unchanged. Compiler changes require a new
immutable semantic build. Old verification receipts and the old ZIP cannot
certify this implementation. Packaging requires matching current code,
configuration, complete-corpus verification and resolved experiment definitions.

Run the integrated local preparation in Windows CMD:

```cmd
cd /d C:\Users\pigby\neurosymbolic_coognitive_linguistics
call scripts\prepare_analysis.cmd
```

This compiles the complete real corpus, checks the accepted archive and all
measurement links/marginals, exercises actual released textual arrays and real
queries through the readouts, and packages the verified result. These numerical
inputs are not substitutes for fMRI or modern states and do not establish
scientific accuracy. No fabricated data or annotation calls are used.

After it reports `ANALYSIS PACKAGE READY`, the reviewed package can be transferred
through the existing `upload_analysis.cmd` workflow. Installation, environment
creation and cache writes remain inside the user-operated compute-node setup.
No step here opens SSH or submits a job. Cluster preflight now checks the whole
declared model panel's actual source/aligned file headers and provenance, in
addition to response-file sizes. Missing models no longer permit an overall
ready status. It does not reread every array value or replace the original
download/extraction verification.

Scientific fits begin only after the user returns the preparation and cluster
readiness results. The absence of local full fMRI and model states is expected;
no local replacement dataset is constructed.
