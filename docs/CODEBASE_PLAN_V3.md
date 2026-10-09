# Codebase plan V3

**Scientific direction, 9 October 2026. Plan only: no V3 code, fits or results exist yet.**

Read this file first for the current direction. V3 replaces the measurement design of [Codebase plan V2](CODEBASE_PLAN_V2.md). It keeps V2's preparation choices: per-occurrence timing and story-wise evaluation.

Other documents:
- [SCIENTIFIC_STATUS_HANDOFF.md](SCIENTIFIC_STATUS_HANDOFF.md) records V1.
- [FIRST_DAY_RESULTS_REVIEW.md](FIRST_DAY_RESULTS_REVIEW.md) and [FIRST_PRINCIPLES_MEASUREMENT_REVIEW.md](FIRST_PRINCIPLES_MEASUREMENT_REVIEW.md) explain why V1's results cannot be interpreted.
- [ENGINEERING_HANDOFF.md](ENGINEERING_HANDOFF.md) owns cluster operations.

## 1. Thesis

**Text or Experience? How Language Models and the Human Brain Organize Concepts**

Large language models learn about concepts from text alone, whereas people also learn them through perception, action, emotion and social life. We ask whether this difference leaves a measurable mark on how concepts are organized.

Using fMRI recorded while people read and listen to natural stories, we annotate every mention of a concept, including pronouns that refer back to it. For each concept, we estimate its pattern of brain activity and its internal representation in a panel of LLMs. We compare which concepts each system treats as alike, measured against how consistently human brains agree with one another.

We then test how far up a ladder of increasingly rich information one must go to explain the brain's organization:
1. simple text co-occurrence,
2. text-trained LLMs,
3. LLMs trained on text and images,
4. human ratings of sensory, motor, emotional and social experience.

We also test whether the brain's organization during comprehension resembles LLM representations formed in context or out of context. The results will show where text statistics suffice to model human concepts, and where the brain draws on experience or context that current LLMs lack.

## 2. Research questions

| ID | Question | Answered by |
|---|---|---|
| Q1 Agreement | How closely does each LLM's concept organization match the brain's, relative to how closely human brains match each other? | LLM–brain agreement reported as a fraction of the human benchmark, beyond the time-shifted control |
| Q2 Basis | Which kinds of information explain the brain's organization, and the part the LLM misses: text co-occurrence, text-trained LLM, text+image LLM, or rated experience (sensory, motor, emotional, social)? | Splitting the brain table into parts explained uniquely or jointly by each reference |
| Q3 Context | Is the reading brain's organization closer to LLM representations formed in context (inside the story) or out of context ("a mother")? | Comparing the brain table with in-context and out-of-context LLM tables, and with out-of-context human ratings |

Cross-cutting comparisons, using the existing model panel:
- **Text-only vs. text+image:** OLMo-3 vs. Qwen3.5/3.8.
- **Base vs. instruction-tuned:** two pairs.
- **Size:** 9B vs. 27B.
- **Layer:** every layer.

These are descriptive contrasts, not controlled causal tests. The models also differ in data and architecture.

Supporting question, **localization**: where in the brain are concepts represented, and where is the brain's organization closest to text-based or to experience-based references? Extension, **roles**: do the systems organize people-in-roles (person-as-doer vs. person-as-receiver) alike?

Working expectation, to be confirmed or rejected:
- The brain's organization contains experience-based structure beyond what text statistics explain.
- In this story corpus, that structure is mainly emotional and social, because sensory and motor variation among the recurring concepts is small.

## 3. Terms

| Term | Meaning here |
|---|---|
| Concept | A reusable meaning that one content word or fixed phrase can name: a kind of thing (*mother*, *telephone*) or a kind of event or state (*give*, *think*). |
| Occurrence | Any place where the text refers to a concept, including other wordings and pronouns that refer back to it. |
| Concept pattern | The activity an encoder associates with one concept: across voxels for the brain, across hidden units for an LLM. |
| Similarity table | For every pair of concepts, how alike their patterns are (an RDM). Every system and every reference produces one over the same concepts. |
| Agreement | Rank correlation between two similarity tables. It measures whether the same pairs come out more or less alike, not whether internal codes are identical. |
| Human benchmark | Each participant's brain table compared with the average of the other eight. This is the agreement any model can reasonably be expected to reach. |
| Time-shifted control | The same pipeline run on brain data shifted in time relative to the text. It measures agreement that the shared concept timeline creates on its own. |
| Reference table | A similarity table from outside the brain/LLM pair: co-occurrence, category, rated experience or human similarity judgments. |

## 4. Materials

| Material | Status | Facts that matter |
|---|---|---|
| Deniz reading fMRI | On the cluster | 9 participants, 11 stories, 23,655 words, TR 2.0045 s. Stories 01–10 are for development; story 11 (with repeats) is held out. Roughly 2 hours of development data per participant. |
| Deniz listening fMRI | In the same release, not downloaded | Same participants and stories. `manifests/datasets.json` lists `responses/subjectNN_listening_fmri_data_trn.hdf`. Doubles the data with no new annotation. |
| Reviewed concept graphs | In git | `data/annotations/deniz-independent-review-v1/bundles/5cd83e0779bb5440da47.zip` (commit `aaf05d9`), 1,217 units. Records entities (concept, kind), mentions (form, antecedent, target) and events (predicate, roles). |
| LLM activity | On the cluster | `words[layer, word, hidden]` for every word and layer, full-story context, last token of each word (see [extraction.md](extraction.md)). Layer 0 is the embedding layer. |
| Word co-occurrence features | In the release | The Huth/Deniz 985-dimension `english1000` features, already readable as the `legacy` group (`neurosym/encoding.py:138`). |
| Rating norms, GloVe, WordNet | To download | Lancaster sensorimotor norms, Glasgow norms, socialness norms, GloVe vectors, WordNet through `nltk`. None are in `requirements/` yet. |
| Expansion datasets | Candidates | LeBel (`lebel` in `manifests/datasets.json`, unused; needs annotation). Pereira et al. 2018 Experiment 1 (180 concepts across categories; not registered). |

The model panel (from `manifests/frozen-models.lock.json`):

| ID | Model | Training input | Layers × width |
|---|---|---|---|
| `olmo3-7b-base` | allenai/Olmo-3-1025-7B | Text only (`Olmo3ForCausalLM`) | 32 × 4,096 |
| `olmo3-7b-instruct` | allenai/Olmo-3-7B-Instruct | Text only | 32 × 4,096 |
| `qwen35-9b-base` | Qwen/Qwen3.5-9B-Base | Text + images (vision encoder in the checkpoint; the model card describes early-fusion multimodal training) | 32 × 4,096 |
| `qwen35-9b-post` | Qwen/Qwen3.5-9B | Text + images, post-trained | 32 × 4,096 |
| `qwen38-27b` | Qwen/Qwen3.8-27B | Text + images, post-trained | 64 × 5,120 |

Annotation support measured on stories 01–10 (concept recurs = at least N occurrences in at least 3 stories):
- **Simple concepts:**
  - N = 10: 27 entity concepts + 42 event concepts.
  - N = 5: 53 + 93.
  - These counts come before any predicate merging.
- **Person:** "person" is 46% of entity mentions and lumps 186 entities, including the narrator.
- **Pronouns:** 94% resolve to an entity.
- **Predicate labels are fragmented:** 1,674 labels after the interface's 21 alias groups (e.g. `give`, `give_joy`, `give_feedback`).
- **Sense:** the `sense` field is a sparse free-text note, not a sense identifier.
- **Roles and frames:**
  - Concept-in-role items: 37 recur, almost all person or kinship terms.
  - Specific typed frames such as pick(person, object): only 13 recur.

## 5. Keep, replace, drop

**Keep and reuse:**
- Corpus and timing readers: `neurosym/dataset.py` (`DenizReader`), `corpus.py`, `temporal.py` (`event_bins`, `causal_bin_features`, `fir_design`).
- `model_features.py` (`FrozenFeatureReader`).
- Annotation readers: `reviewed_archive.py`, `reviewed_graph.py`.
- `spatial.py` (Schaefer mapping).
- Ridge solvers in `encoding.py`.
- RDM utilities in `geometry.py` (`cosine_rdm`, `rdm_comparison`, `crossed_interval`, `fdr_bh`).
- Execution and packaging infrastructure, after removing V1 job inventories.

**Replace:**
- **Concept vectors:** V1's passage-averaged vectors (every concept in a passage got the passage's vector) become encoder-estimated concept patterns.
- **Timestamps:** annotation-unit end times become per-occurrence word times.
- **Features:** exact predicate–role–filler categories become concept columns, with roles as a separate extension.

**Drop:**
- The question-answering decoder and its answer catalogues.
- The five V1 geometry views.
- The stories 01–05 vs. 06–10 context partitions.
- The V1 job inventory.

V1 outputs keep their identities as historical evidence. V2 and V1 names such as "execution protocol 2" must not be reused for V3.

## 6. Measurement design

### 6.1 Concept inventory

- **Entity concepts:** an entity's `concept` label. Every mention that targets the entity is an occurrence: names, descriptions and pronouns.
- **Event concepts:** an event's canonical predicate.
  - Merge a compound label onto a base verb only where meaning is kept (e.g. `give_feedback` → `give`).
  - Record every merge in a reviewed mapping file.
  - Respect the interface's `do_not_merge` list.
- **Split "person":** at minimum separate the narrator from other people. Use the annotated descriptions (man, woman, mother…) where they exist.
- **Ambiguous words:** flag common ambiguous words for a manual meaning check. The `sense` field cannot be used for this.
- **Inclusion rule:**
  - At least N occurrences in at least 3 development stories.
  - Fix N before any brain analysis.
  - Choose concepts from stories 01–10 counts only.
- **Variants kept for analysis:**
  - with and without pronoun occurrences;
  - entity and event concepts separately and combined.
- **Spot check:** people check a sample of the occurrences used, especially pronoun links and merged predicates. Annotation accuracy has not been measured.

### 6.2 Occurrence timeline

- **Occurrence time:** the on-screen time of the mention's head word, or the event's trigger word, from the released word timings. Unknown times stay missing.
- **Concept table:** one row per 2-second scan, one column per concept, counting its occurrences. Built with the existing causal binning.
- **Reading controls:** the released `numwords`, `numletters`, `letters`, `word_length_std` and `pauses` features, in every model.
- **Delays:** the brain encoder uses 1–4 scan delays (`fir_design`); the matched LLM encoder uses none.

### 6.3 Brain concept patterns

- **Encoder:**
  - Per participant: ridge regression from the delayed concept table plus controls to each voxel.
  - Penalty chosen per voxel by story-wise cross-validation inside the training stories.
- **Concept pattern:** the concept's weights summed over delays.
- **Voxels:**
  - those the concept model predicts in held-out training stories, selected inside the training folds only;
  - plus a whole-cortex variant.
- **Distances are cross-validated:**
  - Estimate patterns on two disjoint sets of stories.
  - Compute concept distances across the sets, so noise that differs between sets does not inflate or shrink them.
- **Output:** one brain similarity table per participant.

### 6.4 LLM concept patterns (per model and layer)

1. **In context (direct):**
   - The cached hidden state at each occurrence's word, averaged over occurrences.
   - Pronoun occurrences use the pronoun's state; a variant excludes them.
2. **Matched:** LLM word states averaged into the same 2-second scans, then the same encoder as the brain, without delays. This makes the brain–LLM comparison like-for-like.
3. **Out of context:**
   - The last-token state for a short phrase such as "A mother" or "To give", as in Bogdan et al. 2026.
   - This needs a small new extraction job: one short phrase per concept per model.

All layers are kept as a depth profile. Any headline layer is chosen on stories 01–10.

If the direct and matched versions give different tables, the difference is attributed to measurement and reported.

### 6.5 Reference tables

| Reference | Source | Role in the ladder |
|---|---|---|
| Text co-occurrence | `english1000` features; GloVe | Simple text statistics |
| Category | WordNet hierarchy plus annotation entity kinds | Taxonomy |
| Rated experience | Lancaster (6 senses, 5 body parts), Glasgow (valence, arousal, dominance, concreteness, imageability…), socialness ratings | Experience, tested per dimension and combined |
| Human similarity judgments (optional) | Our own odd-one-out survey over the final concept list | Explicit human organization, with split-half reliability |

Each reference needs a reviewed concept → word mapping. Coverage is reported. Every comparison uses only concepts present in all tables involved.

### 6.6 Comparisons and controls

- **Agreement:** Spearman correlation between the off-diagonal entries of two tables.
- **Human benchmark:**
  - Lower bound: each participant vs. the mean of the other eight.
  - Upper bound: each participant vs. the mean of all nine.
- **Time-shifted control:**
  - Within each story, circularly shift the brain responses by more than 30 seconds (at least 15 scans) relative to the text.
  - Refit and repeat 100 times.
  - Agreement must exceed this null.
- **Uncertainty:**
  - Bootstrap over participants and concepts together.
  - Model and layer comparisons are paired across participants, with false-discovery-rate correction.
- **Story 11:** stays held out until every choice is fixed. The headline analyses then run on it once.

### 6.7 Ladder analysis (Q2)

1. **Unique and shared parts.**
   - Explain each participant's brain table with the reference tables and the LLM tables (text-only OLMo, text+image Qwen).
   - Report the parts each explains uniquely and jointly, against the benchmark.
2. **Gap analysis.** Take the part of the brain table the best LLM misses, and test which references explain it.
3. **Per dimension.**
   - Test social, emotional, interoceptive, sensory and motor dimensions separately.
   - Report how much each dimension varies across our concepts. Little variation means a weak test, not a negative result.
4. **Text-only vs. text+image.**
   - Compare OLMo-3 and Qwen3.5 on the experience-based parts.
   - Report this as suggestive, because the families differ in more than images.

### 6.8 Context analysis (Q3)

Compare the brain table with:
- the in-context LLM tables,
- the out-of-context LLM tables,
- out-of-context human ratings and judgments.

Matched pairs:
- brain vs. in-context LLM;
- out-of-context ratings vs. out-of-context LLM.

The cross pairs show how much context alone changes agreement.

Not attempted: story-specific brain patterns. There are only about 3 occurrences per concept per story.

### 6.9 Localization (supporting)

- **Region-by-region comparison:** repeat Q1 and Q2 inside predefined regions. Use Schaefer 200 parcels in 17 networks, fixed in advance. The current mapping covers about 40% of native voxels, so every map reports coverage.
- **Encoding map:** where concept groups change activity, from the encoder weights.
- **Decoding map:**
  - Per region, a regularized linear classifier identifies concept presence in held-out stories, scored by AUC.
  - Baselines: concept frequency, and time-shifted data.
  - Positive control: the same decoder must succeed on LLM activity before brain decoding results are interpreted. V1's decoder failed this control.
  - Decoder weights are not read as activity maps.
- A location claim needs the encoding and decoding maps to agree, and to recur across participants.

### 6.10 Roles (extension)

People-in-roles items include person-as-doer, -experiencer, -undergoer and -receiver: 37 recurring items, mostly person and kinship terms.
- **Brain:** role columns are added beyond concept columns.
- **LLM:** apply Bogdan et al.'s subtraction method to in-context states. Remove each concept's mean and each role's mean, leaving a role component.

## 7. Subgoals

**Must** goals define the paper. **Should** goals strengthen it. **Could** goals are optional.

| ID | Subgoal | Output | Done when | Depends on | Priority |
|---|---|---|---|---|---|
| G1 | Freeze the concept inventory | Concept list, occurrence records, reviewed merge and word-mapping files, counts | Inclusion rule fixed; at least 50 concepts pass after merging; occurrence spot check recorded | — | Must |
| G2 | Build the occurrence timeline | Concept × scan tables and controls for every story | Spot checks on real passages show occurrences on the right scans; missing times preserved | G1 | Must |
| G3 | Check norm coverage | Reference tables; coverage and per-dimension variation report | Coverage per source reported; usable dimensions identified | G1 | Must |
| G4 | **First check (go/no-go)**: are brain tables measurable? | Brain tables for 2 participants, then all 9; benchmark; time-shifted null | Brain tables agree across participants and across disjoint story sets beyond the null, under a criterion fixed before running. Failure → G9 before anything else. | G2 | Must (gate) |
| G5 | LLM concept tables | In-context, matched and out-of-context tables for every model and layer | Tables agree across story halves; a simple decoder identifies concepts from LLM activity (positive control) | G1, G2 | Must |
| G6 | Q1 Agreement | LLM–brain agreement against benchmark and null for all models and layers | Complete results with uncertainty | G4, G5 | Must |
| G7 | Q2 Text vs. experience ladder | Unique and shared parts per reference; gap analysis; text-only vs. text+image contrast | Results per dimension, with uncertainty and variation reported | G3, G6 | Must |
| G8 | Q3 Context | Brain vs. in-context and out-of-context LLM tables and ratings | Results with uncertainty | G5, G6 (G3) | Must |
| G9 | Expand data | Listening data aligned and passed through G2/G4; then LeBel and/or Pereira 2018 | Each added dataset passes its own G2/G4 checks | Triggered by G4, or planned | Should; listening becomes Must if G4 is marginal |
| G10 | Human similarity judgments | Odd-one-out survey (ethics approval), judgment table, split-half reliability | Reliability reported; table covers the final concept list | G1 | Could |
| G11 | Localization | Region-wise Q1/Q2, encoding and decoding maps | Maps with coverage and cross-participant consistency | G6 | Should |
| G12 | Roles | Role tables for brain and LLM; agreement | Results or a documented "not measurable" | G6 | Could |
| G13 | Held-out confirmation and write-up | Story 11 rerun of headline results; figures; paper | All Must goals complete and choices frozen | G6–G8 | Must |

Order of work:

```text
G1 ─┬─> G2 ─┬─> G4 (gate) ──> G6 ─┬─> G7 ─┐
    │       │      │ fail         ├─> G8 ─┼─> G13
    │       │      └─> G9 ─> G4   └─> G11, G12 (optional)
    │       └─> G5 ─────────────> G6
    ├─> G3 ──────────────────────> G7
    └─> G10 (optional) ──────────> G7, G8
```

G1, G3 and the out-of-context extraction in G5 can start immediately. G4 is the first decision point. Q1–Q3 are not run at full scale until it passes.

## 8. Proposed code layout

These are destinations, not existing files:

| Component | Responsibility | Reuses |
|---|---|---|
| `neurosym/v3/concepts.py` | Inventory, merges, word mapping, occurrence records | `reviewed_archive.py`, `reviewed_graph.py` |
| `neurosym/v3/timeline.py` | Concept × scan tables and controls | `dataset.py`, `temporal.py`, `corpus.py` |
| `neurosym/v3/patterns.py` | Brain and matched-LLM encoders; cross-validated distances | `encoding.py` solvers |
| `neurosym/v3/llm_concepts.py` | In-context and out-of-context concept vectors | `model_features.py`; small new extraction |
| `neurosym/v3/references.py` | Co-occurrence, category, norms and judgments → tables | — |
| `neurosym/v3/rsa.py` | Agreement, benchmark, time-shifted null, bootstrap, unique/shared parts | `geometry.py` |
| `neurosym/v3/localization.py` | Region-wise comparison; encoding and decoding maps | `spatial.py` |
| `neurosym/v3/report.py` | Figures, tables, real examples | — |
| `configs/science_v3.json`, `scripts/run_v3.py` | One configuration and workflow | Execution infrastructure |

- Prepared inputs go in `data/processed/concepts-v3`.
- Results go in `data/analysis-v3`.
- Downloaded norms go in `data/external/norms`.
- Dependencies to add, with pinned versions: `nltk` (WordNet), and optionally `rsatoolbox` and `himalaya`.

**Compute estimate:**
- No new fMRI and no new annotation.
- One small LLM extraction for out-of-context phrases.
- Each encoder is a single regression on about 3,500 scans per participant.
- Rough total: under 10 GPU-hours, mostly the time-shifted control. G4 will measure it; the V1 estimate does not apply.
- Inventory, mapping and norms work runs locally. Fits run on the cluster, where the responses and model states are.

## 9. Data expansion

| Dataset | What it adds | Cost | Trigger |
|---|---|---|---|
| Deniz listening | Doubles data for the same people and stories; also a reading-vs-listening check | Download and alignment only | Planned; required if G4 is marginal |
| LeBel story listening | Many more hours and more stories, for more occurrences per concept | Annotation of new stories with the existing pipeline | G4 fails even with listening data, or concept counts are too low |
| Pereira et al. 2018 Exp. 1 | 180 concepts spanning objects, actions and abstract ideas: a strong sensory/motor test and a balanced concept set | Acquisition and registration; isolated-sentence design, so Q3 does not apply | To make Q2 general beyond story concepts |

Each added dataset runs through its own G2 and G4 before entering Q1–Q3.

## 10. Risks and open decisions

**Risks:**
1. **Brain tables too noisy.** With 9 people and about 2 hours each, this is the main risk. Bogdan et al. 2026 report region-level agreements of r ≈ .005–.04 with 60 people in an event-related design. G4 decides.
2. **Too few concepts** after merging and thresholding.
3. **Little sensory and motor variation** among story concepts. This weakens that part of Q2 unless Pereira is added.
4. **Text and experience overlap.** Language describes experience, so the unique parts will be small. Report shared and unique parts.
5. **"Experience" is measured through people's ratings**, not experience itself. State this.
6. **The text-only vs. text+image contrast is confounded** by model family.
7. **Annotation accuracy is unmeasured.** G1 includes a human spot check.

**Open decisions:**
- The occurrence threshold N.
- Predicate-merge rules.
- How to split "person".
- The G4 pass criterion.
- Which expansion dataset comes after listening.
- Whether to run the judgment survey.
- Target venue and timing.

## 11. Operating rules carried over

- The researcher operates SSH and authorizes every remote action.
- Environments, caches and jobs run on allocated compute nodes, never the login node.
- Local user-run commands use Windows CMD, not PowerShell.
- V1 and V2 outputs keep their identities.
- Story 11 stays reserved until development choices are fixed.
- An inaccessible resource or long run is handed to the researcher with exact steps, never filled with placeholder data.
