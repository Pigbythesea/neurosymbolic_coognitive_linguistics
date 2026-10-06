# Encoding comparison support

The default temporal mask is now specific to an explicitly declared comparison.
All conditions in that comparison use the same training, inner-validation and
held-out response rows. The old all-semantic-group intersection remains available
as `--mask-policy all-groups`; it is no longer silently applied to every analysis.
Accepted annotation build `5cd83e0779bb5440da47` and the compiled semantic feature
values, scope interpretations and uncertainty records are unchanged.

## Meaning and scientific choice

An encoding mask selects timepoints, not brain parcels. It does not erase raw
responses or graph records. Three clocks remain distinct: word exposure,
interpretation availability at the full annotation-unit endpoint, and delayed
fMRI measurement. Missing times are not imputed. The four original-timeline FIR
lags are applied before selecting rows, never after compressing a masked timeline.
The mask is constructed from timing and annotation/model availability; it neither
reads brain outcomes nor learns parameters. Vocabulary, scaling and regression
selection remain fitted inside their training folds. Story 11 cannot fit a
vocabulary or select a model.

The comparison's support requires unit timing, word timing, and the availability
of every semantic feature group used by any competing condition. For example,
presentation+C versus presentation+C+R both require C+R availability. An unrelated
B or S uncertainty no longer removes that observation. A separate concept-only
question can use a larger mask, but its score cannot be subtracted from a score
on the reference comparison's smaller sample as a paired effect.

Known components remain preserved in partially specified annotations. The current
complete-vector design still withholds a group's affected rows when a contribution
to that group is unresolved. This change does not fill unknown bindings with zero,
create uncertainty probabilities or resolve ambiguous scope. The verification
report counts uncertain units that also contain known contributions within each
group. A different within-group missing-data estimator would be a separately
specified analysis, not an undocumented change in what absence means.

## Declaring comparisons

Every encoding invocation requires either `--comparison` or `--mask-groups`.
`configs/analysis.json` supplies the following comparison unions; presentation
controls are included in each. These profiles describe permitted conditions,
not automatically launched experiments or restrictions on the research scope.

| Comparison | Semantic groups in the shared support |
|---|---|
| concepts | C |
| binding | C, B, BR |
| general-binding | C, GB, GBR |
| predicate-binding | C, PB, PBR |
| reference | C, R |
| scope | C, S |
| discourse | C, D |
| state | C, U |
| joint | L, C, B, BR, GB, GBR, PB, PBR, R, S, D, U |

For the reference comparison, the baseline arguments include
`--comparison reference --groups presentation C`, and the augmented arguments
include `--comparison reference --groups presentation C R`. Repeat both with
`--mask-policy all-groups` for the stricter robustness comparison. Changing a
profile or choosing an explicit union is a recorded analysis choice; decide it
before reading final-test performance.

`--mask-groups` declares the union for a custom contrast, including surface or
legacy controls when used. A fitted condition must be a subset of that union.
For frozen-state comparisons, repeat `--mask-model MODEL_ID:LAYER` for every
model/layer in the contrast, identically on both baseline and augmented runs.
The latter notation specifies the option format, not a runnable example. Modern
state availability is checked from actual aligned HDF metadata even in the
baseline; a missing extraction stops execution. The baseline does not load full
hidden-state tensors solely to obtain this mask. The fitted condition's
`--model` and `--layer` must be in the declared list. Aligned run identities are
included in baseline as well as model run provenance.

This supports either a separate matched semantic comparison for each frozen
model, or one declared intersection across several models. Different models are
never silently allowed to score different rows in a paired comparison. No model
is fabricated or silently skipped when files are unavailable.

## Outputs and verification

Every run writes `support.json` before fitting. It records the exact comparison,
semantic build and model identities, original and trimmed retained response
indices, FIR lags, overlapping failure reasons, and additive timing/semantic/model
exclusion counts. Per-story coverage also counts full, partial and missing
delayed windows by real occurrence kind and uncertain feature group. Counts
describe coverage, not independent samples or an annotation-accuracy estimate.

The same support object governs every nested fold. A condition whose feature
design would narrow it further fails instead of silently changing the comparison.
`selection.json`, `complete.json` and `encoding.h5` carry its digest; the HDF also
saves training and held-out row indices. Feature vocabularies and scalers remain
specific to each training fold and condition.

`run_analysis.py compare-encoding` accepts two completed encoding directories.
It checks participant, nested partitions, run contracts and identical declared
support (including training rows) before reporting paired held-out correlation
differences. It uses a common finite-voxel mask for the descriptive subtraction.
It does not treat voxels/timepoints as independent replicates or generate a
population p-value. Downstream encoding-implied geometry retains the parent
run's selected response rows and support provenance. Decoder eligibility and
native/grounding geometry are not silently changed by this encoding policy.

Local full-corpus verification, in CMD:

```cmd
cd /d C:\Users\pigby\neurosymbolic_coognitive_linguistics
.venv-analysis\Scripts\python.exe -B scripts\verify_encoding_support.py
```

The verifier reconstructs availability independently from all real unit
uncertainty records, checks all 11 story timelines and all nine configured
comparison profiles, and compares actual baseline/augmented design masks.
It checks legacy all-group reproduction, differing training-vocabulary coverage,
undeclared-input rejection, mismatched-support rejection and story 11 fitting
rejection. It reports within-group partial information without changing labels.
No synthetic fMRI, dummy model states or scientific fit is used.

The accepted corpus passed 99 paired-design checks (nine comparison profiles
across all 11 stories), covering 1,217 units and 4,028 response rows. Retention
before any additional frozen-state availability requirement is:

| Comparison | Retained rows |
|---|---:|
| concepts | 3,539 |
| binding | 2,036 |
| general-binding | 2,155 |
| predicate-binding | 2,155 |
| reference | 2,979 |
| scope | 2,100 |
| discourse | 3,365 |
| state | 3,444 |
| joint / all-groups robustness | 1,993 |

These are measured coverage counts, not evidence of predictive performance.
The within-group audit also confirms that partially specified units often retain
known contributions: for example, 442 of 546 units flagged for B and 162 of 182
units flagged for R. Those known contributions remain in the immutable compiled
features. This implementation restores unrelated analyses' access to those
timepoints; it does not claim to have solved within-group missing-feature modeling.

`scripts/prepare_analysis.cmd` includes this verification. Packaging requires
receipts for the current code, configuration and accepted semantic build; the
bundle contains the detailed coverage report and this document. Full frozen-state
metadata and brain fitting remain researcher-operated compute-node work. This
local implementation neither contacts nor changes the cluster.
