# Independent annotation review: scientific handoff

The completed package selected by
`data/annotations/deniz-independent-review-v1/latest.json` is the recommended
annotation input. Its `recommended_input` names an immutable
`exports/<build>/annotations.jsonl`; pin the full build hash for an analysis run.
The exporter refuses to recommend an incomplete review. The package manifest and
verification report give the exact unit, correction and open-question counts.

The completed review covers all 11 stories and 1,217 units. The adopted result
contains 56 targeted operations across 38 units; 1,179 unit graphs retain their
baseline values exactly. These are coverage and change counts, not accuracy
estimates.

The review preserves the project's full scope: compositional concepts and
relations, participant roles, reference and discourse for encoding human brain
responses, decoding semantic information, and comparing representational
geometry with language-model representations. No neural measurements or model
performance selected the labels. Downstream feature extraction, decoders,
analysis implementation and cluster execution were left to other threads.

Each story received sequential source-first interpretation and comparison for
every unit. The reviewer saved an interpretation before revealing that unit's
graph and committed the comparison before the next source. Story reviewers used
fresh target-story contexts; the coordinating reviewer completed stories 04 and
06 sequentially, then performed retrospective corpus/interface integration.
Individual exposure declarations and story assessments document details.
Immutable receipts verify the supported sequence but cannot prove absence of
other filesystem access or pretrained familiarity. All reviewers were agents in
the same model system, not independent human annotators; the exact model revision
was not exposed. No measured semantic accuracy or human validation is claimed.

Corrections address specific source-supported omissions, incorrect reference
candidates, role/scope or quantification defects and misclassified mentions.
The adoption manifest separately excludes 20 redundant introduction-mention
proposals from story 11 and one redundant qualifier-context copy from story 08.
The reviewers and integration owner agreed that existing entity evidence already
supplied the mention bindings. Both the actual proposals and
the narrower adopted result remain reconstructible; no saved review receipt was
rewritten. This policy decision adds no later-source meaning to an earlier unit.
The story 08 qualifier already targets a scoped event; its interpretation stays
within that target scope even with no additional qualifier context.
Valid existing readings and plausible alternatives remain. Source chronology
inconsistencies and disfluent wording are retained with uncertainty; outside
story facts were not used to repair the narrative. Low correction counts do not
establish high accuracy. Graph uncertainty is retained in `uncertainties.jsonl`;
new review questions are in `unresolved.jsonl`, and consequential retained cases
appear in story assessments and the human packet.

Use these package files:

| File | Purpose |
| --- | --- |
| `annotations.jsonl` | Recommended reviewed graphs, preserving baseline and operational provenance |
| `coverage.jsonl` | All-unit interpretations, comparisons, hashes, chronology and status |
| `changes.jsonl` | Exact field operations, affected IDs, anchored source evidence, availability and later-reference checks |
| `unresolved.jsonl`, `uncertainties.jsonl` | Review questions and retained graph uncertainties; join by affected/target IDs before treating a disputed reading as settled |
| `adoption.json`, `excluded-proposals.jsonl` | Explicit exclusions from the recommended view, with preserved proposed operations and reasons |
| `question-bindings.json` | Target IDs supplied for one saved review note that omitted them; original note unchanged |
| `review/story_XX/` | Immutable source/graph exposure receipts, interpretations, decisions and assessment |
| `interface/definitions.json` | Explicit reversible label and predicate-conditioned role mappings |
| `interface/scope-cases.json` | Consequential scope formulas, attribution continuations and later-update semantics |
| `interface/scope-bindings.jsonl` | Available context paths for every scoped statement |
| `interface/relation-orientations.jsonl` | Explicit canonical endpoint directions for mixed-direction restatement and example labels; raw relations preserved |
| `human-review/index.html` | Source-first adjudication of changes, consequential retained cases and three reproducibly selected unflagged units per story |
| `verification.json`, `checksums.json` | Coverage, patch/reference/evidence/timing checks and file integrity |
| `baseline/` | Unchanged starting author-corrected and original operational annotations |

Read `independent_review_interface.md` before implementing the compiler. Preserve
all contexts, local polarity, modality, full qualifier values, ordered roles,
uncertain reference, evidence and availability. Derive qualified expressions,
context paths, role aliases and scoped identity only according to the contract.
Treat unsupported qualification and unresolved alternatives as unknown. Missing
assertions are not false. Entity introduction does not assert actual existence.
Later identity and modifiers cannot be backdated to an earlier trigger.
Read the review questions alongside the graphs: a retained graph is a defensible
reading, and a flagged alternative remains available for adjudication. Preserve
unaffected information while withholding conclusions that require resolving the
flagged attachment, speaker or reference choice. A question does not negate its
target and does not make every other assertion in that unit uncertain.
These are prefix-specific questions. Later explicit clarification can resolve a
choice from its own availability point onward, while the earlier question remains
an accurate record of what was unknown then (for example story 10's named day).

The normalized sidecar is reversible and optional. It does not erase fine senses
or license blanket agent/theme substitutions. Broad context names alone do not
settle factivity. Explicit formulas resolve redundant negative/modal encodings;
for example no-way-to-excel is NOT POSSIBLE(excel), not IMPOSSIBLE(NOT excel).
The raw graph remains available for every interpretation.
Canonical relation direction also requires care: raw `restatement` uses both
directions in the corpus. The interface enumerates those cases rather than
guessing from chronology or collapsing physical recurrence into repeated content.

Story 11 remains the held-out story. The all-story vocabulary inventory describes
the corpus after review; it is not a permitted source for fitting analysis
vocabularies, pruning or thresholds. Fit empirical transforms inside training
folds. No downstream representation was fitted during this review.

The package passes deterministic reconstruction of all saved graph patches,
lossless node/statement/argument compilation, evidence coordinates, reference
availability and context acyclicity. Unchanged graphs retain identical JSON
values and canonical hashes; original export checksums remain unchanged. Source text, word
alignment labels, endpoint-resolution flags and null timing values are copied
without interpolation. These checks support the semantic work and do not
replace human judgment.

Human adjudication is the next scientific validation step. The HTML packet asks
for a source interpretation before annotation reveal and exports judgments
locally. It includes source prefixes without displaying future text. Review in
order with reviewers who have not read ahead; browser gating is not a security
guarantee. No human responses are included or implied by the package.
The selected cases retain their proper unit availability. Whole-story assessment
prose is disclosed only after the final unit's human interpretation: such prose
can mention later outcomes even without an explicit future ID. Earlier cards
show only saved prefix comparisons, graphs, corrections and local scope cases.
Later-reference audit details are also deferred to the final card; the complete
change log retains those details for retrospective implementation review.
The packet's source/reveal, story progress and local note export behavior were
checked in a JavaScript DOM harness. No browser was connected for visual layout
inspection; functional checks are not human adjudication or visual QA.

Windows CMD verification, from the repository root:

```cmd
.venv\Scripts\python.exe -B scripts\export_independent_review.py audit
```

To verify an immutable package, pass its directory from `latest.json` to
`scripts\export_independent_review.py verify --directory`. The package captures
the validation/export code and runtime dependency versions. Preserve the pinned
baseline archive when retaining full historical authoring provenance.

The matching ZIP and SHA-256 receipt in
`data/annotations/deniz-independent-review-v1/bundles/` preserve the complete
review export, including its baseline copies, receipts, code and human packet.
The data directory is ignored by Git; retain this bundle separately from the
repository. To recreate the same verified bundle from the current recommendation:

```cmd
.venv\Scripts\python.exe -B scripts\package_independent_review.py
```
