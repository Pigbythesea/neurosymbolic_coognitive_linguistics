# Reviewed semantic interface v1

This is a symbolic annotation contract for concepts, roles, reference and discourse
in encoding, decoding and representational comparisons. It specifies what a
downstream implementation must preserve. It does not implement features, fit a
vocabulary, establish semantic accuracy, or convert the corpus to the legacy
analysis schema.

The complete reviewed package identifies its input in `manifest.json` and the
review root's `latest.json`. Use the package's `annotations.jsonl`. Original
author-corrected records and original operational records remain separately
available under `baseline/`. The optional `interface/normalized-records.jsonl`
contains every original record plus explicit, reversible aliases. It is not a
replacement source of unqualified facts.

## Units, evidence and availability

Every observation is available at its **full annotation-unit endpoint**.
`evidence.start_token` and `end_token` locate linguistic support; they do not
determine when an interpretation was available. Apply this rule to qualifiers,
identity links and later statements about earlier events as well as new events.
Keep `available_at_seconds`, endpoint resolution, word alignment labels, timing
counts and null times exactly. Token availability does not license invented
seconds when alignment is unresolved.

Review timestamps document the review procedure. They are not stimulus times.
`operational_parent_graph_hash` preserves the original authoring history;
`compiled_parent_graph_hash` describes the reviewed view's compilation chain.
The baseline compiled parent is retained separately. A correction is versioned
replacement of a defective annotation; a later narrative update is an additional
observation at its later unit. Neither rewrites the original archive.

## Typed records and reference

| Record | Required interpretation |
| --- | --- |
| Entity | A discourse referent or kind, with denotation and introduction status. Introduction alone asserts neither physical existence nor an actual event. A prospective child or desired tool can be mentioned without existing. |
| Event | An occurrence, state or predication token with predicate, sense, ordered role bindings, polarity, tense, mode and context membership. Repeated labels do not imply event identity. |
| Mention | A source span that targets an entity/event/literal. Null target and alternatives preserve unresolved reference. Null antecedent means no explicit antecedent link was supplied; do not manufacture one. |
| Literal | Typed value and unit. Approximate counts, ranges, names and exact numbers remain distinct. |
| Context | A perspective or operator node with parents, holder, attribution event and source evidence. Preserve the full DAG. |
| Relation | Directed source/target assertion with its own support and contexts. Endpoint meanings include local polarity and relevant qualification. Direction cannot be inferred from English label spelling alone. |
| Property | Scoped assertion or retraction about a target. Retraction changes a claim in that scope, not all versions of the entity. |
| Qualifier | Scoped modification of its explicit target, possibly an earlier record or context. Preserve the complete value tree, including references, domains and alternatives. |
| Identity link | Scoped same/different/possible claim between like-typed nodes. Apply only when available and within scope. |
| Uncertainty | Targeted alternatives and resolution status; not an extra positive proposition. Preserve conservative choices as choices, not certainty. |

IDs are story-local discourse identifiers. A shared concept label is not identity.
A generic node reused in two alternatives need not denote one actual individual.
A single character can have incompatible states in imagined worlds without
contradiction. Unresolved references are unknown, not false or empty sets.

Relations such as `same_event` express token identity; `same_episode_as` only
places descriptions in the same episode. `realization_of` links later actual
action to earlier intended or prospective content and does not collapse their
assertion statuses. Keep identity claims as edges before deriving any quotient
graph. Do not apply a global union-find across all times and contexts.
Likewise, `story_10:u0034_rel1` (`realizes_intention`) links actual pile location
to the earlier commanded placement. It does not import the commanded father's
agency into that later actual result. Correspondence is not unrestricted
argument inheritance.

## Predicate and role consistency

`interface/definitions.json` lists the approved alias groups, role maps and
reasons. The normalization view retains each entire original record, its label,
argument position and graph hash. Examples include:

| Canonical predicate | Observed labels and roles | Real examples |
| --- | --- | --- |
| `buy` | `buy`, `purchase`; agent/buyer → buyer; theme/object → purchased object | `story_01:u0013_buy`, `story_03:u0085_buy`, `story_06:u0001_buy` |
| `be_age` | `be_age`, `be_aged`; theme/person → age bearer; age/value → age value | `story_01:u0083_age`, `story_06:u0003_age_claim`, `story_07:u0017_age` |
| `obtain_job` | `get_job`, `obtain_job`, `obtain_employment`; acquiring person and job roles | `story_06:u0054_obtain_job`, `story_08:u0056_get_job`, `story_10:u0058_getjob` |
| `be_able` | `be_able`, `be_able_to`; capacity bearer and capacity content | `story_04:u0064_ability`, `story_06:u0001_ability` |
| `receive_phone_call` | `receive_call`, `receive_phone_call`; caller/source → caller, preserving null | `story_05:u0000_receive`, `story_08:u0114_receive` |
| `be_naked` | `naked`, `be_naked`, `be_nude` | `story_03:u0035_naked`, `story_07:u0014_naked`, `story_07:u0082_nude` |
| `reside` | `reside`, `live_at`, `live_in`; resident and residence roles | `story_02:u0005_reside`, `story_01:u0068_live`, `story_03:u0107_live` |

These are linguistic interface aliases, not a new gold ontology or a complete
synonym lexicon. Adjectival `afraid`/`be_afraid` and spelling `be_ok`/`be_okay`
also have explicit aliases, without collapsing their stimulus/content roles or
context-specific senses. Inspected adjectival pairs also cover certain,
different, disgusting, excited, fortunate, frightening, fun, great, happy,
interesting and nervous with their `be_` variants; farewell speech-act variants
share one key. This is an enumerated list, not a general rule stripping `be_`.
The definition file retains each justification and predicate-specific role map.
Unlisted predicates retain typed original keys. In
particular, bare `live`, ambiguous `can`, `be_allowed_or_able_to`, `know`, `believe`,
`work_with` and `work_for` are not reduced to nearby labels.

Role mappings are **predicate-conditioned**, never global substitutions of
agent, theme, person, object or content. Preserve repeated argument positions.
Two agents can jointly perform one action; repeated bearers with a distributive
qualifier can each hold a property. Repetition alone chooses neither reading.
Do not equate beneficiary, intended beneficiary and recipient, or colleague and
employer. Specialized content roles retain their distinctions.

Relation normalization also preserves direction. The optional view aligns
`contrasts`/`contrasts_with`/`discourse_contrast` with `contrast`,
`explains`/`explanation_for` with `explanation`, and the inspected
elaboration, answer, continuation and example variants listed in the definitions.
It keeps concession, event recurrence and calendar coincidence distinctions.
Explaining a proposition does not automatically assert physical causation.

Raw `restatement` has inconsistent orientation. The eight explicit bindings in
`restatement_orientation_by_id` yield canonical `restates(new, earlier)`:
`story_05:u0042_rel2` reverses its raw endpoints, while
`story_07:u0059_rel0` already points from the narrator's echo to the girls' claim.
`story_08:u0153_rel0` uses `restates` in that same canonical direction.
Likewise `story_01:u0048_rel2` runs from a general opportunity to jousting;
its canonical `example_of` view reverses the endpoints. The raw record and an
`endpoints_reversed` flag make each mapping reversible. No chronological sorting
or blanket reversal is allowed; unlisted ambiguous labels retain their originals.

## Scope composition

Treat the graph as a scoped expression graph, not a bag of positive triples.
`interface/scope-bindings.jsonl` enumerates outer-to-inner context paths for each
statement at its own availability point. The following order is implementable:

A qualifier is an attachment to its target expression. Its explicit contexts
can restrict or attribute that attachment; an empty list does not project the
target's qualified content out of the target's scope. In
`story_08:u0128_qua0`, intensity qualifies `u0128_cool` inside the quoted appraisal
and knowledge context. Copying those contexts onto the qualifier is redundant,
so that proposed edit is excluded from the adopted graphs. The scope sidecar
lists explicit paths and the qualifier target's context paths separately;
a compiler must retain that target-expression dependency.

1. Resolve only currently available references. Collect ordered arguments,
   local polarity, tense, mode, qualifiers and uncertainty for the requested
   statement. Do not attach future modifiers to earlier prefixes.
2. Apply an explicit case in `interface/scope-cases.json` when present. These
   cases name consumed polarity/qualifier fields, preventing duplicate operators.
3. Otherwise preserve a typed local expression with all unresolved scope
   structure. A compiler that cannot interpret a qualifier must keep it as a
   symbolic operator or report an unsupported case, rather than discard it.
4. Attach context paths from outermost parent to innermost context. A context
   with an attribution event describes that event's content; it does not by
   itself create a second saying, thinking or modal event. Consult attribution
   predicate, polarity, sense and the explicit continuation cases.
5. Resolve relation and content endpoints to those qualified expressions.
   Context membership of the relation itself is separate from endpoint scope.
   Keep a referenced expression rather than flattening it into root assertions.

An endpoint may itself be a context expression. The corrected
`story_10:u0002_rel2` points from lack of known innocent-story examples to the
**denied protective-purpose context**. The explanation is asserted at the root;
it does not explain positive protection and is not itself negated. Consumers
must accept typed context endpoints rather than silently replacing them with a
positive child event.

Absence of contexts means asserted in the narrator's current discourse, not
independently verified real-world truth. A positive predicate inside a question,
belief, condition, command, desire, ability, depiction or counterfactual remains
inside that operator. Counterfactual content is not automatically a negative
actual-world event. Generic and habitual statements are not single episodes.

Broad context aliases are operator-family aliases only: possibility/possible,
capacity/ability, appearance/apparent, belief/believed and required/obligatory.
Original kinds remain available. `believed` also hosts remembered and discovered
content in this corpus. Factivity therefore depends on the attribution and
projects only relative to its parent world. `story_01:u0118_knowledge` is factive
within an imagined scene; it does not assert an actual-world affair.

Negation on the attitude and negation on its content differ: NOT THINK(p) is not
THINK(NOT p), and WANT(NOT p) is not NOT WANT(p). Neg-raising may be an explicitly
recorded alternative, never an automatic rewrite. Modal denial also differs
from uncertain denial: NOT POSSIBLE(p), POSSIBLE(NOT p), NOT ABLE(p) and
ABLE(NOT p) must not collapse.

`scope-cases.json` supplies machine-readable formulas for consequential examples:

| Example | Required composition |
| --- | --- |
| `story_03:u0050_failure` + `u0050_qua0` | NOT POSSIBLE(excel), because negative polarity and “no way” are one modal denial. Never IMPOSSIBLE(NOT excel). |
| `story_06:u0037_impossible` | NOT POSSIBLE(grow up AND NOT know). The without-clause belongs inside the impossible scenario. |
| `story_04:u0063_matter` | NOT substantially-matter. Strategies enable this negative qualified proposition, not positive mattering. |
| `story_04:u0050_continued_report`, `u0051_report` | Continue the same maternal explanation in `u0049_explanation`; do not introduce a duplicate saying operator. |
| `story_06:u0049_possibility` + `u0049_qua0` | One possibility inside the question, with two redundant graph representations. |
| `story_06:u0044_qua0` | Later explicit left/right clarification supersedes an earlier inferred pair type only from unit 44 onward. |
| `story_09:u0057_rel0`, `u0057_rel1` | Have(candidate) AND believe(universally-right(candidate)) jointly form one conditional antecedent; neither edge is independently sufficient. |
| `story_09:u0084_pro0` | The stored surface coordination is AND; the container name `alternatives_or_combination` does not change it to exclusive OR. Preserve the possibilities inside belief. |
| `story_09:u0086_capability_content` | Follow the negative ability attribution within feeling and possibility; the broad context name does not assert a separate positive ability or possibility of trust. |

No universal “deduplicate equal words” rule is authorized. Distinct nested
speakers can truly report other speakers. Use attribution IDs, evidence and
explicit continuation instructions, retaining the original DAG.
For example, `story_07:u0048_report` continues `u0047_report`, while
`u0048_nested_report` introduces a hypothetical customer's speech inside that
warning. The packet's explicit cases preserve the latter operator. Similarly,
an uncertain abandoned start such as `story_07:u0076_grab` is a retained partial
frame with unresolved factual commitment, not a completed event with an invented
patient. The `assertion_eligibility` case makes this constraint explicit.

`interface/modal-scope-bindings.jsonl` lists all 30 inspected negative-modal
encodings that need the same composition convention. The attached operator is
already a denial, prohibition or refusal; negative event polarity is consumed
once. Both “ability” with a negative predicate and “inability” with a negative
predicate denote NOT ABLE(positive content), while permission/ability ambiguity
remains unresolved. Original lexical force and the contexts are preserved.
The definition file supplies the instance IDs and quantifier/frequency rules;
these conventions do not turn symbolic qualification into fitted features.

## Quantifiers, questions and figurative language

Quantifiers bind the stated domain/role, not every participant in the event.
Preserve existential versus universal, free choice, negated existential,
collective versus distributive, comparison exclusions, approximate cardinality
and branch-local alternatives. `story_01:review_u0031_whatever` means unrestricted
opportunities within an intended plan, not an actual universally experienced set.
`story_03:review_u0078_every_other` and `review_u0079_every_other` exclude different
speakers from their comparison domains.

“No one” plus event negative often represents one negated existential. “Never”
plus event negative often restricts one denial over time. Combine scope from the
quantifier and polarity; do not negate twice or silently convert no-event to an
event. “Not really” preserves a degree threshold rather than no degree at all.
Disjunctive role alternatives do not create multiple actual occurrences.

Question focus matters: why asks for a reason; what/who/where bind an unknown
slot. The positive event can be presupposed while its reason is questioned
(`story_04_u0047`). Preserve null fillers and focus qualifiers. Prepared answers
to anticipated questions (`story_06:review_u0007_answer_reason`) do not assert
that the question or answer occurred.

Figurative contexts and sense/qualifier records distinguish literal imagery
from intended meaning. The toothpick in `story_06_u0045` is a feared metaphor for
an inadequate bat, not a verified object in a player's hand. The moral in
`story_06_u0054` does not prove a tool was brought into existence. Preserve
hyperbolic counts without treating their literal numeral as measured quantity.

## Open vocabulary and implementation limits

All observed labels, including rare ones, are inventoried with real IDs and
separate held-out counts. This retrospective inventory is diagnostic. It must
not determine feature inclusion, thresholds, embeddings or fitted vocabulary
using held-out stories. Training-fold fitting remains a separate analysis task.
The review protocol for story 11 was unchanged.

The interface intentionally retains source-specific predicates and opaque
qualifier payloads where no justified common meaning has been established.
Implementations may derive typed semantic structures using these definitions;
they must preserve a lossless path back to the graph and explicitly track
unsupported or unresolved interpretations. Human adjudication remains necessary
for documented alternatives. Structural replay is evidence of reproducibility,
not semantic accuracy.
