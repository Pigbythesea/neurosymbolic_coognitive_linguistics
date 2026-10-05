# Independent semantic review protocol v1

This review preserves the complete project scope in `PROJECT_HANDOFF.md`: concepts,
roles, reference and discourse support encoding, decoding and representational
comparisons. Labels are judged from source text, never neural results or model
performance. Corrections must be necessary, source-supported and surgical.

The input is pinned by `data/annotations/deniz-independent-review-v1/baseline.json`,
resolved from the production `latest.json`, using `author-corrected/annotations.jsonl`.
Original annotations, author corrections and exports remain unchanged.

Each reviewer starts without target-story text/graphs, registers one story, and
uses the gate below. Do not open complete source files, original author modules,
author self-reviews, other stories' content, or future graphs. The gate is a
workflow with receipts, not a filesystem security boundary. Report any accidental
exposure honestly. Pretrained familiarity with a narrative is not ruled out.

For every unit:

1. Read the current source only, in the context of the already reviewed prefix.
2. Save a brief independent interpretation of that source before revealing its
   annotation. Include consequential scope, reference or uncertainty when present.
3. Read the complete corresponding annotation, compare meaning and omissions,
   retain legitimate stylistic variants, and record the comparison.
4. Apply only defensible field changes/additions/removals. Preserve stable IDs when
   meaning allows. Flag genuine ambiguity with evidence and candidate readings.
5. Commit the decision before seeing the next source unit. Consider the effect
   of earlier corrections on every later reference as the story proceeds.

Availability is the full unit endpoint, not the trigger/evidence endpoint. A later
revelation is a later assertion; never backdate it. No missing assertion is false.
Distinguish narrative truth from speech, belief, purpose, negation, condition,
modality and figurative comparisons. Check roles, identity, quantification,
collectivity/distribution, temporal order and omitted substantive meaning.

Story 11 is held out. This same review protocol applies unchanged. Later global
label mappings must distinguish semantic interface definitions from empirically
fitted vocabularies. Do not use held-out content to fit downstream representations.

## Gate use (Windows CMD)

```cmd
.venv\Scripts\python.exe -B scripts\independent_review.py register --story story_01 --actor /root/review_story_01
.venv\Scripts\python.exe -B scripts\independent_review.py interpret --story story_01 --input data\annotations\deniz-independent-review-v1\work\story_01_interpretation.json
.venv\Scripts\python.exe -B scripts\independent_review.py commit --story story_01 --input data\annotations\deniz-independent-review-v1\work\story_01_decision.json
```

`interpret` takes `{"unit_id":"story_01_u0000","interpretation":"Source-based meaning ..."}`
and returns the graph with raw evidence quotes/start coordinates. `commit` takes
`unit_id`, substantive `assessment`, `operations` (default empty), and `unresolved`
(default empty). It validates the graph and returns the next source. The work JSON
files can be overwritten; the interpretation and decision receipts are immutable.
Use `apply_patch` to write JSON, then run the gate; avoid shell quote pitfalls.

Each operation has `op` (`set`, `add`, `remove`), `collection`, full story-local
`id`, `reason`, and `evidence` (`{"quote":"exact current/prefix text","start":123}`;
start is optional for unique text). A `set` also has dot-separated `field`, exact
`before` value and replacement `value`; nested list indexes are supported.
An `add` has `value` containing a complete raw record. A `remove` has the complete
raw record as `before`. Graphs use the existing direct annotation schema. New IDs
should contain `review_uNNNN_`. Keep existing conventions where defensible.

An unresolved item has `reason`, `evidence`, `affected_ids`, and optionally
`alternatives`. Avoid using unresolved notes to excuse a clearly false commitment.
For unresolved cases needing changed graphs, set null/unresolved references or
add appropriate targeted uncertainty as justified by the source.

Read `docs/direct_annotation_conventions.md` and schema definitions in
`neurosym/direct_annotations.py` for field semantics, treating prior judgments as
fallible. `ledger` returns only already reviewed decisions. Recover prior semantic
records only from this story's committed review decision files.

At story completion, write `stories/<story>/assessment.json` with concise findings,
consequential issues, real-ID examples for compiler semantics, any interface
ambiguities, and explicit exposure/procedure limitations. Review coverage is the
saved per-unit work, not a checklist or an automatic validity score. No human
validation or measured semantic accuracy is claimed.
