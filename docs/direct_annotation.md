# Deniz production direct annotations

The current reviewed input is selected by
`data/annotations/deniz-independent-review-v1/latest.json`. Read the
[independent scientific review handoff](independent_review_handoff.md) for its
semantic interface and validation limits. This document retains the original
authoring and author-correction provenance.

The current production attempt is `data/annotations/deniz-direct-v2`, protocol
`direct-semantic-graph-v2`. The researcher explicitly selected direct authoring
and discarded CLI model generation. Earlier Luna runners and the
36-unit `deniz-direct-v1` diagnostic package remain historical development
evidence; neither supplies production labels.

The scope is the full actual Deniz reading transcript corpus: 11 stories,
1,217 prepared units and 23,655 source words. Story 11 is held out. The source
wording, token coordinates and released timing interface remain intact. Missing
timing endpoints remain null rather than being interpolated.

Information availability is recorded at the endpoint of the whole exposed unit.
Evidence spans locate source words; they do not assert that the author made a
word-by-word decision before seeing the remainder of that same unit.

Read [the conventions](direct_annotation_conventions.md) for semantic decisions
and the actual progressive authoring protocol. Fresh isolated actors author
stories 01–04 and 07–11. The integration context authored stories 05 and 06
progressively without prior target-story text or graphs; it had seen development
material for other stories. Each actor receives one current unit, authors it
directly, self-checks it, and commits before seeing the next. The parent hash connects
current input to previously committed history. Agents are instructed not to
open full source files, later passages, legacy graphs, or other story outputs.
The recorded gate and traces support auditing this process; they are not a
technical sandbox preventing every possible out-of-protocol file read, nor a
claim that pretrained models have never encountered an underlying narrative.

## Saved evidence and reproducibility

The package saves a pinned source/convention manifest, schema, story-author
declarations, actual current-unit requests, authored raw drafts, accepted graphs,
per-unit Python author modules, rejected structural attempts, and an operational
trace. Records retain model-visible evidence quotations, exact source spans,
availability endpoints, parent graph hashes and author source hashes. No LLM
confidence score is used as a quality estimate.

The runtime identifies the annotator family as GPT-6-based Codex. An exact model
revision and reasoning setting are not exposed to the annotation code, so those
fields are null with an explicit visibility note. They must not be replaced by
the old Luna configuration or guessed from output. Direct model generation is
not asserted to be byte-for-byte repeatable. The saved decisions and deterministic
serialization/exports are reproducible without a model call.

`run-instructions.json` preserves the researcher's scope and authorizations.
`access-provenance.json` distinguishes isolated contexts, target-unexposed
integration authoring, recorded input lineage and audit limits. The pinned guide
is retained; a complete provider transcript of every model context is unavailable.

Each author module is an explicit record of semantic decisions. The shared
`neurosym/direct_annotations.py` code does not infer content from the transcript.
It locates quotations using the existing transcript tokenizer, validates record
and reference structure, checks information availability, and saves/exports
unchanged labels. Open typed roles and relationships remain inspectable labels,
not silent mappings into the older closed graph enums.

## Inspection and exports (Windows CMD)

```cmd
.venv\Scripts\python.exe -B scripts\direct_annotation.py status
.venv\Scripts\python.exe -B scripts\direct_annotation.py verify
.venv\Scripts\python.exe -B scripts\direct_annotation.py export --require-complete
.venv\Scripts\python.exe -B scripts\direct_annotation.py verify-export
.venv\Scripts\python.exe -B scripts\verify_direct_export_links.py
.venv\Scripts\python.exe -B scripts\package_direct_annotation.py
```

`data/annotations/deniz-direct-v2/latest.json` names the latest immutable export snapshot.
Its directory uses the first 20 digest characters to keep Windows paths short;
`snapshot.json` and the report retain and verify the complete build hash. Complete coverage must
be checked separately from successful export; the completeness option reports
incomplete coverage as an error while preserving actual partial exports.
The packaging command verifies complete coverage and freezes the selected export
as a ZIP under `bundles/`, with a separate SHA-256 receipt. This provides a portable
copy of the data directory, which the repository's existing ignore rules exclude
from Git. It does not invoke an annotator or change saved labels.

The exports contain complete annotation records, nodes, statements, ordered
argument rows, targeted uncertainties, source-only units, concept/predicate
introductions, conventions/schema/provenance, per-story HTML review, and recursive
checksums. `schema.json` describes raw authoring drafts; `schemas/accepted-graph.json`
describes saved graphs with derived evidence spans and verbatim source text. Both
record views place that graph under `graph`. Availability and provenance remain
record metadata, and source/reference constraints are checked by replay.
`annotations.jsonl` retains the whole accepted records. `nodes.jsonl`
and `statements.jsonl` partition every semantic collection without changing it.
`arguments.jsonl` preserves role order, filler support, event context IDs and
polarity. `concept-introductions.jsonl` contains introductions only. IDs are local instance IDs; concept and predicate
labels are semantic type keys. Qualifiers and targeted uncertainty remain in the
exports. Introductory concept rows are not an exhaustive occurrence catalog: all
mentions, event arguments, and evidence are available for later compiler design.

`source-only.jsonl` and `sources/<story>.jsonl` retain exact wording, character and
token coordinates, released word timing, unit timing and availability. The frozen
`sources/<story>.json` contains only the accepted prefix captured for that export;
partial builds contain no later source passages. `provenance/` includes actual
unit requests, raw drafts, author modules and self-reviews, structural attempts,
available traces and historical validation code. The captured compiler and its
local dependencies are in `code/`. No model call is needed to replay or compile.

`compile_records()` is a pure function over saved records, request packets,
captured source prefixes and split metadata. It retains nested context IDs,
unresolved/null fillers, alternative identities, modifier values and property
operations. It performs no identity merging, ontology harmonization, semantic
correction, scope flattening or feature fitting. Re-running it from the same
inputs produces identical UTF-8 JSONL bytes.

The annotation package does not generate numerical features, decoder questions,
fit vocabularies, change analysis configuration, or adapt downstream modules.
Compatibility and scientific analysis compilation will be addressed in a
separate thread, as requested by the researcher.

## Versioned corrections from fresh prefixes

A substantive issue found in self-review can receive a fresh progressive recheck.
The new context first receives only source units through the disputed unit and
commits an independent interpretation of each unit. It receives no later passage,
completed story graph, original labels, or description of the suspected mistake
during this stage. The recheck package retains its requests, drafts, graphs,
author declarations and code evidence under `prefix-rechecks/`.

After that independent prefix is committed, `original_prefix_input()` supplies
only the matching original prior graphs and current draft for ID alignment.
The rechecker authors an explicit correction and `save_revision()` saves it under
`revisions/`; `revision-index.json` selects the version. Original accepted records
remain unchanged. Retaining their IDs permits later saved references to replay;
the correction may introduce additional source-supported records.

Within the original authoring export, the recommended view is `author-corrected/annotations.jsonl` and its
companion tables. The root JSONL files preserve the original attempt. Both views
use the same lossless compiler. Each corrected row distinguishes the actual
original operational parent from the parent in the recomputed view, so later
authors are never represented as having seen a correction written afterwards.
Per-unit review links identify selected corrections and the original evidence.
Author self-reviews describe the original attempt; a flagged issue may have been
addressed by a selected correction. Consult the unit's revision links and the
export report's `selected_prefix_corrections` before treating it as still open.
Export verification replays both views and verifies the independently committed
prefix and the subsequent ID-alignment input. This establishes recorded process
and structural consistency; it is not independent semantic validation.

## Quality and subsequent review

Mechanical verification replays each accepted draft against its source unit,
checks exact quotations and references, checks nested contexts for cycles,
verifies hash lineage, and detects altered author modules. It establishes
structural and provenance consistency, not semantic correctness. Annotation
self-checking is also distinct from independent review.

`review.html` links each story and its author self-review. Per-story HTML
juxtaposes each current passage and its graph, including summary,
role fillers, scope, modifiers and uncertainty. The source-only export permits
fresh interpretation without viewing committed labels. Human review remains
pending until the researcher records review/adjudication. Later corrections
should be versioned and should distinguish an original prefix interpretation
from a retrospective reading; they must never silently overwrite history.

The gate controls the supported workflow, rather than filesystem permissions.
The input lineage can be mechanically verified; broader context access is
documented by author declarations and task instructions. Early accepted units
predate the later-added operational trace or historical-code-hash field. These
missing records are reported explicitly and have not been invented. Neither
pretraining exposure nor an independent audit of every model context is claimed.
