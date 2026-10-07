# Analysis implementation and scientific report

This implements the analysis layer in `PROJECT_HANDOFF.md`: semantic encoding,
answer-supervised decoding, spatial grounding, and four distinct geometry
families. Representation models remain frozen. The compiler makes the targets;
the readouts learn from targets and real observations. This layer makes no LLM calls.

## Current result and execution boundary

The isolated local analysis environment is ready. The spatial adapter has run
on actual released mappers for all nine participants. Numerical/interface checks
have run on released textual feature arrays and real compiled queries. Those
arrays verify algorithms; they are never represented as fMRI or contemporary
model states. **No scientific brain/model fit has run locally.**

The subsequent CPU/GPU optimization pass changes these sources; earlier numerical
receipts do not certify it. Current verification status, caches, checkpointing,
resource profiles and researcher-run preparation are in [compute.md](compute.md).

The accepted annotation input is independent-review build
`5cd83e0779bb5440da47`, pinned by its full hash in `configs/semantics.json`.
Review is closed. The complete downstream contract, scope interpretations,
uncertainty handling and current query definitions are described in
[reviewed_downstream.md](reviewed_downstream.md). Compiled output is
`data/processed/semantics-reviewed`. Earlier legacy verification receipts do not
authorize this build; packaging requires receipts for the current build and code.

Full responses and modern hidden states remain on the cluster. The 18 complete
local response files needed for all brain fits are absent. Previous cluster
reports established four complete aligned models and an incomplete fifth;
current cluster status was not re-inspected. No SSH connection, cluster change,
installation or submission was performed in this work package.

## Spatial support

`spatial.py` uses the Schaefer 200-parcel, 17-network fsaverage atlas, pinned to
CBIG revision `634f676630929a71297852d01dd92a287103e861`. The
[official source](https://github.com/ThomasYeoLab/CBIG/tree/634f676630929a71297852d01dd92a287103e861/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations)
supplies left/right FreeSurfer annotations. Source URLs, Git blob identities,
SHA256 hashes, parcel names and mapper hashes are saved with the assets.

A parcel is a predefined cortical subdivision, not a concept label. The dataset
supplies native cortical voxels and fsaverage mappings, allowing common atlas
labels to be related to each participant's voxel order. Participant 04 has
separate left/right matrices; others follow the release's concatenated
left-then-right convention. Registration was not re-estimated, and an independent
landmark registration check has not been performed.

Several interpolation matrices have negative coefficients. Atlas-label voting
uses **absolute interpolation influence, normalized per surface vertex**, and
assigns each native voxel its largest unique parcel vote. Ties, medial-wall and
unmapped voxels are excluded. This is an explicit assignment convention, not
probabilistic interpretation of interpolation weights. Original data are unchanged.

| Participant | Assigned voxels | Native voxels | Observed parcels |
|---|---:|---:|---:|
| 01 | 33,578 | 81,133 | 200 |
| 02 | 33,833 | 80,350 | 200 |
| 03 | 31,671 | 72,965 | 200 |
| 04 | 31,293 | 72,966 | 200 |
| 05 | 33,563 | 80,615 | 200 |
| 06 | 37,196 | 87,742 | 200 |
| 07 | 38,180 | 92,970 | 200 |
| 08 | 33,516 | 79,936 | 200 |
| 09 | 34,913 | 85,277 | 200 |

All parcels being represented does not mean full native-voxel coverage. Spatial
decoding uses this covered view; encoding and native geometry retain all native
voxels separately. Paper maps must report coverage and the assignment convention.
Alternative parcellations/assignments are additional sensitivity analyses, not
results established here.

## Observations and partitions

`analysis_data.py` checks semantic, response, extraction and timing identities.
Missing arrays produce errors rather than substituted observations.

Brain decoder inputs are ordered delayed response samples for a compiled unit;
repeats remain separate. Training-only means, scales and up to eight PCA
components are fitted independently within each parcel. Rank-zero sites are
masked; actual ranks are saved. Local grounding precedes cross-site composition.

Model inputs are cached full-prefix unit endpoints. Their coordinates are
deterministically partitioned into 32 groups with training-only local PCA.
**Model groups are computational sites, not anatomical regions.** Their saved
assignments and seed make partition sensitivity possible through separate configs.
No empirical partition-sensitivity result is claimed yet.

The public query enters only after state extraction. Decoders receive neither
the source passage nor answer graph, private evidence or answer IDs as inputs.
Public candidate descriptors are also available to the query-only baseline;
their possible shortcuts are part of what that baseline measures.

Folds 0–9 leave out one development story. `final` reserves story 11 and both
repeats. Inner stories select hyperparameters. Explicit multistory development
holdouts, such as `stories:story_01,story_02`, permit cross-context geometry within
one fitted basis. Choose these partitions before inspecting effects. No heldout
story fits a vocabulary, scaler, PCA, penalty, learning rate or stopping epoch.
Layers are explicit conditions, never selected on final-test results by this code.

Delayed brain windows can include responses to later nearby text. Full-prefix
model states and fMRI integrate time differently. The supported interpretation
is offline delayed measurement, not exact online concept localization.

## Encoding

`encoding.py` implements group-kernel ridge regression. Training-only semantic
vocabularies yield C/B/BR/R/S/D/U groups; L is optional. Presentation controls
contain released word/letter counts, letter identity, word-length variation and
pauses. Contemporary model states and optional released legacy semantics are
separate groups. Within-story FIR designs use explicitly declared comparison
support. Baseline and augmented conditions share training, validation and test
rows for the union of their required groups; unrelated uncertainties no longer
exclude rows. `--comparison` selects a named union, or `--mask-groups` declares a
custom union. `--mask-policy all-groups` retains the earlier global intersection
as a robustness condition. Frozen-state contrasts additionally declare identical
`--mask-model` lists on every condition, including baselines. See
[encoding_support.md](encoding_support.md) for definitions and coverage reporting.

Training-only standardization and feature-count normalization balance group
variance. Inner folds select the shared ridge penalty and positive kernel mixture
using MSE normalized by training voxel variance across all variable native
voxels. Exact sufficient statistics avoid materializing predictions for every
hyperparameter/voxel combination. Mixture proposals use a recorded seed.

The selected estimator is refitted on all allowed outer-training stories.
Each run's `artifact.json` resolves a shared `encoding.h5` containing FP64 group
response operators, target means, training/heldout row indices, per-repeat
correlations/MSE, and mean-response correlations. Predictions and contributions
are reconstructed in voxel batches from these operators and the pinned training
responses; downstream readers also accept the earlier dense format. Full native
voxels and group contributions remain available without duplicating their arrays
in every run. See [compute.md](compute.md) for the storage contract.
Story 11 also gives measured repeat correlation;
that is saved as reliability, not silently converted into a universal noise ceiling.

Use matched ablation runs to compare presentation+C with structural groups added,
and model with model+semantic groups, retaining presentation controls. Correlated
feature-group contributions remain model-dependent, not unique causal effects.

`support.json` records exact row identities and exclusion reasons for all stories
used by a run. `compare-encoding` verifies matched support, participant and nested
partitions before reporting paired descriptive effects. Encoding-implied geometry
inherits the parent support; it does not fill excluded predictions.

## Decoding and anatomical grounding without grounding labels

`decoders.py` and `decoder_fit.py` provide four readouts:

| Family | Observation use | Purpose |
|---|---|---|
| prior | None | Query/candidate regularities and answer priors |
| linear | Linear in observations, conditional on public query | Simple semantic accessibility |
| mlp | Flexible global readout | Generic learned mixing comparator |
| structured | Local evidence and typed ordered-pair composition | Spatially auditable answer-supervised grounding |

Descriptors use training vocabulary tokens and numeric position features.
Unknown tokens stay unknown. No pretrained semantic model is hidden in the readout.

The structured model transforms observations locally, then produces unary site
support by concept-conditioned dot products. Low-rank left/right factors produce
typed ordered-pair support. Role binding combines anchor/filler support with that
relation; composition propagates support through typed steps. Reviewed binding
queries compare ordered role assignments. Concept queries identify recorded
content at an anchor. Local polarity and full scope are separate recovery tasks.
Unmentioned edges are not labelled false, and concept identification is not a
claim of actual-world existence. The older binary verification head remains only
for explicitly selected legacy builds.

There is no direct query-only output branch in the structured model, although
learned parameters and context can still supply priors. Baselines and disruptions
test this empirically. Parameter counts and local ranks are saved; the architectures
are not claimed to have identical capacity.

The loss sums probability over all acceptable answers. Unknown/ambiguous labels
and diagnostics are excluded. Query families share equal source weight; stories
share expected epoch weight; repeats share source weight. Reports aggregate
repeats, then questions within sources/families, then stories. Inner validation
selects learning rate and the median best epoch for refitting. Seeds are separate
runs, never extra participants.

Heldout traces save unary scores/routing, ordered-pair factors, intermediate
compositions and query-conditioned latents. Pair factors reconstruct raw pair
scores with the saved site mask. A successful result supports reproducible
decoder evidence for a semantic distinction at particular sites, provided
heldout accuracy, controls and stability support that claim. It does not identify
a unique biological concept location or causal connectivity.

Each fit is tested with matched observations and circular within-story mismatch
whose delayed TR supports are separated. Mismatched model states still share
earlier prefix context; the report states this. `--retrain-null` instead rotates
whole training stories while matching relative endpoint rank, trains anew and
tests on real heldout observations. Mappings are saved. Test-time disruption and
retraining under disrupted pairing are distinct controls.

`--composition-keys` takes a JSON list of selected configuration hashes. Whole
training stories containing those configurations are purged; incomplete stories
cannot certify absence. Constituents must remain observed. Tests select role,
binding, scope, polarity and composition queries on the exact heldout event,
not every query in its story. Insufficient support stops the requested run. Polarity contrasts
remain polarity evidence, not proof of role-swap discrimination.

## Geometry and inference

`geometry.py` preserves exact labels/senses, deduplicates source/item occurrences
and averages stories equally.

| View | Signature |
|---|---|
| native | Training-standardized measured native voxels averaged over delayed samples/repeats, or model endpoint coordinates |
| grounding | Unary concept/predicate site scores, or role/discourse ordered-pair evidence weighted by endpoint routing |
| latent | Latents from occurrence-matched retrieval queries, within one fitted basis; not guaranteed item-separable |
| latent-passage | All-query passage mean, shared by co-occurring items |
| encoding-implied | Heldout predicted voxels or a selected group's prediction contribution |

These are geometries of labelled contexts, not isolated causal concept responses.
Co-occurrence, lexical content and query composition matter. Native geometry can
subtract a training-fitted presentation component using source word/character/
letter counts, word-length variability, prefix position and elapsed time. These
controls use only prefix-available information. Raw and residualized views have
different identities; residualization does not remove every lexical confound.

Grounding profiles support unary concept/predicate/literal/scope and ordered-pair
role, discourse, reference, identity and attachment labels. Native, latent and
encoding-implied views also support whole configurations. Missing signatures and
zero norms are explicit exclusions, not zero distances. Scope can be pooled with
saved provenance or matched explicitly with `--scope-mode scoped`. Site coordinate
metadata accompanies grounding matrices.

Distances are computed within one fitted coordinate system. Across human/model
views compare common-item RDMs, never raw latent coordinates. Label permutation
respects shared RDM cells. Anatomical map stability compares common sites across
fits and permutes semantic labels without treating parcels as independent.
Its scope can be seed, participant or context consistency, not biological necessity.

Multistory test partitions can estimate split-story geometry consistency. A
one-story test reports that estimate unavailable. Story 11 repeats are not
independent contexts. `crossed_interval` supplies paired participant/story
bootstrap intervals for complete panels, averaging repeated seed estimates within
cells; `fdr_bh` adjusts a specified hypothesis family. Contrast selection, semantic
support and paper-level inference remain explicit research choices. No such
result-level inference has run yet.

## Verification and user-operated workflow

`artifacts/analysis-verification.json` records the real inputs, code hashes and
checks: local projection/serialization, ridge primal/dual equivalence, exact
tuning-MSE algebra, timing masks, reviewed tasks across four readouts with finite
gradients, answer-input rejection, prediction/trace serialization, real-feature
RDM identity, and all nine spatial contracts. This does not establish annotation
accuracy, convergence, heldout brain performance or map reliability.

The preflight reports absent full local observations and verifies the declared
model panel's source/aligned headers and provenance where present. A missing
model prevents an overall ready status. Accepted annotation review
is closed; its runner, prompts, configuration and original records remain intact.

To compile, verify and package the accepted review, in local Windows CMD:

```cmd
cd /d C:\Users\pigby\neurosymbolic_coognitive_linguistics
call scripts\prepare_analysis.cmd
```

This compiles with complete-coverage enforcement, verifies the actual new snapshot
and packages it with analysis code and spatial assets. It stops on missing inputs
or changed verification hashes. To upload from Windows CMD:

```cmd
call scripts\upload_analysis.cmd
```

Upload copies files only. Code installs under `analysis_code/<content hash>`, so
extraction/annotation modules are not replaced. Shared input collisions stop
installation. `setup_analysis.sbatch` builds a separate cluster environment with
CUDA-capable PyTorch and project caches on an allocated compute node. It submits
no scientific fits. The user operates cluster submissions in their authenticated
session; the last known billing-cap block is not assumed resolved.

`run_analysis.py` exposes `plan`, `preflight`, `encoding`, `decoder`, `geometry`, `compare-encoding` and
`compare` (`--maps` selects anatomical map stability). Its help lists explicit
conditions. `run_analysis.sbatch` defaults to CPU; CUDA requires a user-requested
GPU allocation plus `--device cuda`. Full fits and large permutations are long
user-run compute jobs. One decoder outer condition includes inner fits for every
training story and learning rate, then a refit, repeated per seed. No runtime or
allocation sufficiency is claimed before real execution.

Submission and log-inspection commands should be supplied together after the
completed annotation package and current allocation are available. No cluster
submission is needed now.
