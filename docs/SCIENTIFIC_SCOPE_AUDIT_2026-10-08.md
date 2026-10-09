**Scientific scope audit against the original brainstorm**

**8 October 2026. Assessment, not a replacement implementation plan.**

**Subsequent decision, 8 October 2026:** This audit concerns the earlier V2 draft. The revised [Codebase plan V2](CODEBASE_PLAN_V2.md) now governs implementation. It restores direct semantic decoding, simplifies measured concepts, and explicitly retains RSA alongside linear correspondence. The researcher also corrected this audit's overly sharp separation of goals: encoding and decoding maps serve the same localization interest through different estimates, and RSA and linear translation are complementary views of organization and correspondence. Preserving the original interest does not require making every reference, scope, or discourse annotation a separate prediction task. Read the assessment below as historical reasoning, not an outstanding implementation checklist.

The original shared conversation was recovered successfully by a direct HTTPS request after the web reader returned a cache miss. A [local transcript](ORIGINAL_BRAINSTORM_TRANSCRIPT.md) now preserves the user messages and assistant final replies. This audit gives the researcher's questions and corrections priority over the assistant's proposals and later summaries.

**Judgment: the central scientific interest is coherent, but both the V1 implementation and the V2 plan have drifted in different ways.** V1 accumulated machinery that made the questions difficult to recognize. V2 removed much of that machinery, but also narrowed some of the questions and favored one measurement route. V2 is a useful redesign proposal; it is not yet a faithful simplification of the full agreed study.

The implementation inspected remains V1. The proposed neurosym/v2 directory and scripts/run_v2.py do not exist. This review changes no scientific code and does not rewrite the V2 plan.

**What the researcher actually asked for**

In [user message 1](ORIGINAL_BRAINSTORM_TRANSCRIPT.md#user-message-1), the starting idea was to connect two research directions: concept grounding without anatomical labels for each concept, and mappings between language-model activity and measured brain responses. The semantic structure was meant to extend from individual concepts to events and relationships across longer text. The researcher also proposed comparing relationships among concepts within the brain-derived and LLM representations.

In [user message 47](ORIGINAL_BRAINSTORM_TRANSCRIPT.md#user-message-47), the researcher pushed back against making conventional prediction scores the central purpose. The intended sequence was to obtain concept-associated maps or signatures, examine the relationships among them within each system, and compare those organizations. The opportunity was an interpretable research framework without having to obtain a correct cortical location for every concept.

The assistant then proposed making conventional neural prediction secondary. The researcher's next correction, [user message 100](ORIGINAL_BRAINSTORM_TRANSCRIPT.md#user-message-100), rejected that hierarchy as well: “do in parallel instead of presetting a "primary/secondary" tier” and “integrating instead of replacing.” The same message rejected fixing one expected result shape in advance.

That correction matters more than either adjacent assistant response. The agreed direction was neither an encoding-first atlas nor a decoder-first accuracy study. Encoding and decoding were complementary ways of investigating concept organization, localization, and human–model correspondence.

In [user message 182](ORIGINAL_BRAINSTORM_TRANSCRIPT.md#user-message-182), the researcher asked for a first-principles explanation because the technical proposal had become too long and difficult to understand. In [user message 209](ORIGINAL_BRAINSTORM_TRANSCRIPT.md#user-message-209), the concerns returned to semantic annotation, dataset suitability, what must be trained, dependence of downstream findings on labels, and the unjustified use of older models. The concern about drift was therefore present in the original conversation itself.

**The scientific question that survives these corrections**

What concepts and structured relationships can we identify in brain and LLM activity; how are those meanings organized within each system; where does their brain-associated evidence lie; and which relationships are shared, different, or dependent on context?

“Structured relationships” means such things as who gave an object to whom, which earlier person a pronoun refers to, whether an event happened or was imagined, and how two events are connected. The graph records those distinctions so they can be studied. It does not dictate the result.

The researcher's hierarchy idea principally concerns meanings combining into events and connected discourse. The later distinction among taxonomic, compositional, and temporal hierarchies is useful clarification. It does not create an obligation to run three separate hierarchy projects or to find a matching cortical hierarchy.

There is also an important correction to an early factual premise. The original exchange clarified that NEURONA trains on semantic answers, while concept-to-region assignments lack direct anatomical labels. The useful freedom is to investigate those assignments without an anatomical answer key. It is not freedom to treat any internally coherent map as supported by the recordings.

**How the current plan compares**

| Scientific interest | Position in V2 | Assessment |
|---|---|---|
| Individual concepts plus roles, reference, scope, and discourse | Preserved in the semantic records and feature blocks | Retained in scope, although preserving a field is not yet an experiment showing that activity distinguishes it |
| Recovering specified meanings from brain and LLM activity | Direct semantic decoding becomes optional | A substantive reduction from the explicit agreement to retain both directions |
| Brain localization without concept-to-region labels | Spatial semantic encoding becomes the main anatomical route | Legitimate and useful, but it changes the main route from inferred decoding evidence to fitted response associations |
| Relationships among concepts within each system | Mainly compared through encoding-implied profiles | The question survives, but the evidence becomes concentrated in one fitted representation |
| Direct LLM-to-brain correspondence | Explicit full and compressed linear mappings | Consistent with the Schrimpf-inspired part of the original idea; compression makes one latent question more explicit |
| Independently estimated organization versus organization after alignment | Both ideas are discussed, but the division could be clearer | A fitted translation and independently similar geometry answer different questions |
| Higher-level composition and accumulation of context | Mentioned, but not given a comparably explicit experiment | Underdeveloped; story generalization alone does not test new combinations or the contribution of earlier discourse |
| Modern models, real recorded stimuli, and trustworthy annotations | Preserved | Useful continuity; the old checkpoint roster and every suggested dataset need not return |
| No predetermined winner or required result shape | V2 selects an anatomical experiment as primary | Overcorrects toward a particular empirical emphasis despite retaining several possible outcomes |

**The main V2 scope change is the loss of a direction of inquiry.**

Predicting a recording from known meaning asks whether that description accounts for variation in activity. Recovering meaning from a recording asks whether the activity supports identifying that meaning. These are related, but one fitted experiment does not automatically perform the other.

It was appropriate to question the 356-candidate retrieval design and the requirement to recover complete annotation records. It did not follow that semantic recovery itself should become optional. A direct, understandable recovery task can retain that question without restoring the old candidate catalogue, every decoder architecture, or the previous execution matrix.

The same applies to neurosymbolic grounding. Encoding-derived semantic maps are a valid way to study localization. They do not reproduce the particular NEURONA-inspired question of where a trained semantic decoder obtains concept and relation evidence. Both can contribute to the project. Choosing only one is a change of emphasis that should be recognized explicitly.

**The organization question should not become an incidental visualization of prediction models.**

The researcher originally proposed building concept relationships within each system and then comparing them. That is a scientific analysis in its own right. It should be able to reveal meaningful agreement or disagreement even when aggregate prediction scores tell a different story.

V2's encoding-implied profiles can contribute to this analysis, but they are one possible estimate. Its added frozen descriptor encoder also introduces another model's semantic organization into both sides. That is not automatically invalid; shared semantic descriptions are necessary for many comparisons. However, choosing MPNet as the default was a new implementation proposal, not a requirement recovered from the brainstorm.

The representation choice should follow what the concept signature is intended to measure. A fixed descriptor encoder may help share information across rare labels. It should not silently define what human conceptual distance means. Nor should removing raw passage averages require discarding all concept-conditioned evidence derived from measured recordings or learned semantic groundings.

We do not need to restore all four old geometry families as equally large pipelines. We do need to say which estimated concept relationships come from which observations and which fitted mapping.

**A learned translation and similar organization are complementary claims.**

A learned mapping can stretch or rotate a representation to predict another. It can therefore generalize successfully even when distances in the two original spaces differ. Conversely, a noisy high-dimensional recording can make pointwise prediction difficult while some concept relationships remain similar.

The direct correspondence experiment is useful. It should retain its held-out prediction test and its interpretation through semantic distinctions. The comparison of independently estimated within-system concept relationships should also remain visible. Neither result should stand in for the other.

This is consistent with the original discussion, which already included LLM-to-fMRI prediction without symbolic labels. V2 makes the compact shared-space interpretation more concrete; it does not newly invent the label-independent branch.

**Composition and context need an empirical question, not just richer storage.**

A graph can store the correct giver and recipient while the numerical representation or fitted readout fails to use that distinction. A compiler check showing different feature values is useful engineering verification, but it is not evidence that brain or LLM activity supports distinguishing the roles.

Similarly, training on nine stories and testing on another asks about new stories. It does not by itself establish that familiar meanings can be recovered in new combinations. Nor does preserving reference links establish how earlier sentences influence the representation of the current event.

Retain those questions explicitly: what changes when familiar participants fill different roles, when a relation reaches across sentences, and when meaning depends on prior context? Their implementation can use the available naturalistic observations and clearly defined model comparisons. This does not require recreating the old exhaustive grid or pretending that edited text has a newly recorded human response.

**The documents themselves also show how scope became mixed with prescriptions.**

The [first-principles reference](Structured_Meaning_Brain_LLM_Human_Reference.md) preserves the central questions well. Its opening asks about recoverable meaning, organization, and agreement or difference across systems. Its discussion of separate measurement directions and informative dissociations remains useful.

The [technical proposal](Structured_Meaning_Brain_LLM_Research_Proposal.md) contains valuable reasoning about identity versus type, directed relationships versus similarity, and learned maps versus measured patterns. However, it also turns many candidate methods into fixed panels, signature families, numerical defaults, and workflow rules. Those details are not the scientific objective.

The [historical handoff](PROJECT_HANDOFF.md) records the researcher's rejection of deadline-driven narrowing and unnecessary machinery. Before this retrieval, it explicitly said the shared conversation had not been reopened. Its summaries were useful secondary evidence, but the recovered user messages now provide a better source for the original decisions.

Thus two mistakes should be avoided: treating every line of the technical proposal as a binding commitment, and treating frustration with its complexity as permission to discard the underlying questions.

**What remains useful now**

Keep the common semantic description of the actual recorded stimuli. Keep the separation between concept types and particular referents, between events and their participants, and between assertions and imagined or reported content. Keep contemporary frozen LLMs as comparison objects, with trained measurement models around them.

Keep semantic recovery and neural prediction as complementary measurements of those meanings. Keep concept relationships as an explicit outcome, including differences across roles, discourse, contexts, and model layers. Keep brain localization through clearly named kinds of evidence. Keep direct brain–LLM correspondence as another informative relationship.

The original compact reasoning about validity also remains useful: the recording must contribute information; a purported relationship should recur appropriately; and a structural result must concern relationships rather than merely the presence of the same words. These explain what an experiment means. They do not prescribe an unlimited collection of controls.

Retain V2's useful implementation corrections: observation timing should follow the evidence, exact rare combinations need not be separate unshareable categories, arbitrary model-coordinate groups are not anatomical regions, large answer catalogues are unnecessary, and the five-story halves are not a scientific commitment.

**Recommended correction before implementing V2**

Keep its simpler preparation and mapping infrastructure, but revise the scope statement and experiment list. Semantic decoding should remain an implemented scientific measurement, using direct semantic targets and interpretable structured readouts. Concept organization should remain explicitly comparable within and across systems, with the origin of each signature stated. Context and composition should have named empirical questions. Direct correspondence should remain complementary rather than replacing the independent organization comparison.

Do not appoint encoding, decoding, or one particular atlas as the required successful outcome. The research can reveal a clear anatomical association, a recoverable semantic distinction, shared organization, a useful transformation, or a well-supported difference among these. The scientific value lies in explaining those relationships.

This is a correction to V2's scope, not a recommendation to restore V1's implementation wholesale. The next plan should simplify the machinery while preserving the questions the researcher actually chose.
