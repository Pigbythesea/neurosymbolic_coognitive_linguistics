# Joint source-anchored annotation

Current protocol: `joint-source-v5`. Output: `data/annotations/deniz-luna-v5`.
Graph schema version 2 retains literals, nested contexts and later identity links.
Earlier runs remain archived. V4 repaired graphs never enter v5 history.
Compatible original v4 drafts can be reused as described below.

## Start or resume in Windows CMD

```cmd
call scripts\annotate_all.cmd
```

This uses the existing ChatGPT CLI login, `gpt-6-luna`, reasoning `none`.
No API fallback, model upgrade, cluster operation or scientific fit occurs.
The user starts the long run. The same command resumes matching saved work.

```cmd
.venv\Scripts\python.exe -B scripts\annotation_status.py
```

## Generation and compilation

One joint request normally handles each unit: referents/mentions, events and
states, roles, literals, nested scope/attribution, discourse, identity links,
updates and uncertainties. The guide is `prompts/graph_annotation_source.txt`.
There is no GROUND/BIND/REVIEW negotiation or semantic critic gate.

The model receives readable preceding text, the current passage and a compact
reference index. It describes each current referent independently and optionally
links it to an existing entity. Local descriptions and selected links are saved
in provenance. A selected existing identity keeps its canonical graph descriptor;
the local description does not silently overwrite historical entity records.
The prompt gives source text priority over prior descriptions and allows a new
or unresolved referent when the index is misleading. This contains a source of
error propagation; it does not guarantee correct reference decisions.

Python resolves quotations, assigns permanent IDs, and validates references and
structure. Quotes normally are plain strings. The corpus tokenizer handles
punctuation, whitespace and transcript markers; word identity and case remain
unchanged. Unique matches need no model-generated count or coordinate. Repeated
expressions can use a surrounding quote, or a correction selects a supplied
source ID with before/after excerpts. No fuzzy lexical substitution is performed.
Mentions/triggers belong to the current unit; evidence can use the available
prefix but must reach the current unit. Event evidence must contain its trigger;
null evidence uses the selected trigger itself. Evidence can disambiguate a
repeated trigger. All resulting source spans are recorded.

Short local keys connect current records; the index retains every earlier
referencable entity, mention, event, context and literal. Forward event/context
references are legal. The compiler selects the latest preceding mention of the
annotated entity as a reproducible pointer, not a model of human memory retrieval.
Update values distinguish actual strings from explicit `{"literal":"key"}`
references. The compiler resolves a literal reference to its recorded value.

New observations become available at the unit endpoint, separately from their
source anchors. A later identity revelation adds a link without merging previous
records. Nested desired/reported content retains both contexts. Names and numeric
values are literals rather than additional discourse entities.

## Recovery and quality

Concrete source/reference/structure errors trigger corrections; independent
errors are collected together. V5 distinguishes two correction modes:

- **Location selection:** when all defects are ambiguous quotations, named fix
  fields select exact supplied source locations. Semantic records stay unchanged.
- **Record completion:** missing/wrong-kind references, absent source expressions
  or structural defects permit replacement/removal of the affected current-unit
  records and additions of missing referents, literals, events or contexts.
  A reference-dependency calculation includes connected records so links can be
  updated atomically. Unrelated records outside that component are immutable.
  Existing node IDs are suggestions, never a forced set of answers. Historical
  records cannot be edited. Null deletes a record; nested mentions/arguments can
  be removed inside a replacement. All additions and links are revalidated.

Malformed JSON/schema output can require a fresh structured response; it is never
accepted into history. The limit is three completed responses per unit across
restarts and sweeps, including a reused original draft. Exhaustion defers that
story while independent stories continue.
Restarting does not purchase the same failed work repeatedly.
Transient service errors have finite retries; quota/auth failures stop with saved
units retained. Missing annotations remain missing.

Prompts, schemas, raw outputs, CLI events, timing and hashes are checkpointed.
Each correction records its parent request, parent annotation hash and allowed
targets. Cache recovery and the final audit replay that chain and require an
identical graph. A malformed correction retains the previous typed draft.
Old v4 correction traces remain replayable for inspection, but their forced-choice
repair procedure is not used to generate v5 corrections.
Request size includes the schema and is checked before invocation. History is
never silently truncated. No model-generated confidence or semantic rule patches
are used to declare an interpretation correct.

`SAVED` means mechanically valid, not human-validated. Confidence is not requested
or fabricated. Heuristic span checks are diagnostics. End-of-run audit material
is under `artifacts/annotation-audit/deniz-luna-v5/`: `review.json` contains model
outputs, and `source-only.json` supports independent annotation including omitted
content and unflagged units. Optional blind-model agreement is repeatability,
not human accuracy. Manual correction remains a later user-operated activity;
there is no new human approval stage or automated semantic judge in this runner.

## Reusing original drafts

`reuse_original_drafts_from` points to the stopped v4 run. Only original
`source_annotation` requests are eligible; corrected graphs and corrected drafts
are excluded. Model, reasoning setting, corpus, unit, full prompt (including
history), and schema must match. An eligible original CLI trace is copied
byte-for-byte with checksums and explicit `source_import` provenance. This logs
`REUSE ORIGINAL DRAFT` and consumes no new model request. Its current defects are
then handled by the v5 completion procedure. If corrected history changes a later
prompt, that later draft is ineligible and is generated afresh. Raw v4 files are
not modified. Reuse establishes compatibility, not semantic correctness.

## Existing downstream pipeline

Defaults now consume v5 annotations and write `data/processed/semantics-v5`.
Earlier builds retain their identities and are not selected as substitutes.
After annotation completes:

```cmd
.venv\Scripts\python.exe -B scripts\compile_semantics.py --require-complete
.venv\Scripts\python.exe -B scripts\verify_semantics.py
```

Then use the existing analysis verification/packaging workflow. Dataset preparation
and annotation-independent model extraction remain reusable.

## Offline verification

```cmd
.venv\Scripts\python.exe -B scripts\validate_preparation.py
.venv\Scripts\python.exe -B scripts\verify_source_annotation.py
.venv\Scripts\python.exe -B scripts\verify_annotation_completion.py
```

These check actual corpus words, all passages, archived v3 responses/failures,
source-choice corrections and downstream query compilation without model calls
or scientific fits. Archived labels are projected only in memory, never adopted
as production annotations or semantic gold. Worked prompt examples use the opening units
of training stories 02, 07 and 10; none contain story 11 or later story content.
They are authored demonstrations, not an independent accuracy evaluation.
`artifacts/source-annotation-verification.json` records the checks and remaining
archived failures. `artifacts/annotation-completion-verification/report.json`
records regression checks on actual v4 failures: missing values/context creation,
record/mention deletion, unchanged location correction, immutable unaffected
records, exact-input draft reuse and trace replay. Authored regression corrections
remain in memory and never enter the annotation corpus. These checks do not
measure fresh v5 model accuracy or guarantee successful generation.
