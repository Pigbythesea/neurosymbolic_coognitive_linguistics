# Frozen representations and temporal alignment

This workstream consumes the existing real Deniz corpus and inspection contract. It does not import or modify annotation generation, graph schemas, prompts, or annotation outputs. The finished word/unit states and scan-aligned designs are inputs to the later encoding, decoding and geometry modules; those fitting modules are separate work.

The panel in `configs/extraction.json` includes Qwen3.5-9B base/post-trained, OLMo3-7B base/instruct, and Qwen3.8-27B. `manifests/frozen-models.lock.json` pins actual repository commits and every downloaded file's size and SHA-256 or Git blob hash. Complete checkpoints require about 115 GiB before environment and feature storage. Model provenance is available at [Qwen3.5 base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), [Qwen3.5 post-trained](https://huggingface.co/Qwen/Qwen3.5-9B), [OLMo3 base](https://huggingface.co/allenai/Olmo-3-1025-7B), [OLMo3 instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct), and [Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B).

## Measurement

Models receive the actual transcript, with brace-delimited nonverbal tags replaced by equal-length whitespace. Inputs contain no task prompt, chat template, graph labels, questions, brain responses, generated answers, or generated reasoning. BOS is added if the tokenizer defines it; EOS is not appended. This treats all models as frozen causal text processors, including post-trained checkpoints.

Every word endpoint and unit endpoint is independently tokenized from its available prefix. Trailing horizontal whitespace is removed; observed newlines are retained. When those IDs equal a prefix of the full-story tokenization, cached causal forwards supply the endpoint state. Any endpoint whose tokenization differs is evaluated from its independently tokenized prefix with a fresh cache. Full-story context is retained; exceeding a model's capacity fails rather than truncating. All embeddings and layer states are stored. Layer zero is the embedding; the final entry follows the model's final normalization, according to the upstream `hidden_states` convention.

The backend loads unquantized checkpoints in bfloat16 and saves resulting vectors as float32. It runs evaluation/inference mode with gradients disabled. Qwen's text backbone and OLMo's decoder are called directly, avoiding vocabulary logits and text generation. SDPA is used for full attention, with the upstream PyTorch implementation for Qwen's recurrent layers. GPU model, package versions, dtype counts, loading diagnostics, code hashes, source hashes and runtime are recorded. The source implementation follows the pinned [Transformers Qwen3.5 API](https://huggingface.co/docs/transformers/model_doc/qwen3_5) and [hidden-state output convention](https://huggingface.co/docs/transformers/main_classes/output).

Output HDF5 files contain `words[layer, word, hidden]` and `units[layer, unit, hidden]`, with chunk checksums. Each story is finalized by atomic rename. Completed stories are reused; an interrupted story is recomputed to reconstruct its recurrent/KV cache. An abrupt termination can leave `RUNNING.lock`; inspect its recorded job/process before removing it. Do not change extraction configuration, corpus or extraction code within a run directory containing computed states. For a failure before any states were written, a corrected run archives the old run record and empty HDF5 files under `failed-starts/`; this recovery refuses nonempty partials or any completed outputs. Loading metadata stores sets of parameter names as sorted JSON lists.

## Scan clock and splits

The [Deniz methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6764208/) specify TR=2.0045 seconds, five leading TRs, and midpoint sampling. Story time zero is therefore raw scan time 10.0225 seconds in this implementation. The released `numwords` features support this onset convention, but their counts do not match the supplied TextGrids exactly. The 10-second rounded description is not treated as a separate precise trigger log. The original inspection note about unavailable absolute trigger logs remains applicable.

For each completed word or unit, the first raw scan midpoint at or after its endpoint receives the feature. Multiple events in that bin are averaged; empty bins are zero and event counts are retained. Unknown times remain missing. Their possible bin intervals, inferred only from neighboring observed anchors, are masked as uncertain. This is causal binning, not the original paper's symmetric Lanczos interpolation; it never places a future event into an earlier bin. The resulting quantization delay is between zero and one TR.

FIR delays are explicit and configurable through the reader. The default is 1–4 TRs (2.0045–8.018 seconds). Delays are constructed inside each story before the published response trim `[10:-10]`; they never cross a story boundary. Model designs match `DenizReader.response(..., trim=True)` row-for-row. The reader returns both design and timing-validity mask. Downstream comparisons must use the intersection of masks for the compared feature sets. Delayed fMRI still reflects overlapping surrounding language; causal model input does not establish isolated neural responses to a single event.

`data/processed/alignment/splits.json` defines leave-one-story-out selection folds and nested outer/inner development folds. Story 11 and both of its repetitions are held out together. All transformations, feature/voxel selection and model selection must be fitted within training folds. No brain-response values were used to construct these manifests.

## User-run cluster workflow

On local Windows CMD:

```cmd
cd /d C:\Users\pigby\neurosymbolic_coognitive_linguistics
call scripts\upload_extraction.cmd
```

The script validates real prepared inputs and transfers a separate extraction bundle plus its installer and CPU setup job. It prints the submission command for the existing SSH session. Installation, environment creation, tokenization planning and model downloads execute only on a Slurm CPU compute node. All environments and caches use the inspected project directory. The installer refuses differing shared inputs or independently edited sources and excludes annotation-owned files.

The setup log must finish with `MODEL DOWNLOAD, HASH VERIFICATION AND FULL-CORPUS TOKENIZATION COMPLETE` and exit status zero. Its next printed submission command runs `scripts/extract_models.sbatch`, an H100 array with one model per task and two tasks concurrently. Each task extracts all 11 stories and then compiles their scan-aligned features. The 27B model requires an 80 GB class GPU for this configuration; the job uses the previously inspected H100 partition. Jobs stop on errors; no model, precision, CPU-offload or API fallback is attempted.

Cluster execution is user-controlled. Neither the packager nor the upload script submits jobs. Long runs should be monitored in the user's existing SSH session, with failures returned to the coding thread.

Use Slurm accounting and the final completion marker to establish job success. A scheduler cancellation is a failed run even if an older shell wrapper printed a zero exit status. The current wrapper handles termination signals and reports failure unless extraction and alignment both finished. Account billing-minute limits require the cluster allocation to permit further execution; increasing a job's wall-time request does not resolve them. After an interrupted array, resubmit only unfinished model indices under the unchanged extraction identity to reuse completed stories; inspect any remaining run lock against Slurm accounting before removing it.

Once environment setup, downloads and tokenization have succeeded, use `call scripts\upload_extraction.cmd --source-only` in local CMD for a source correction. It prints the submission command for `artifacts/setup_extraction.sbatch --source-only`. This performs only source installation on a CPU compute node. Submit the extraction array after that job succeeds, or use a Slurm `afterok` dependency; no package reinstall or model download is needed.

## Downstream interface

```python
from pathlib import Path
from neurosym.model_features import FrozenFeatureReader

features = FrozenFeatureReader(Path(project_root), "qwen35-9b-base")
X, timing_valid = features.design("story_01", layer=16, kind="words")
unit_states = features.events("story_01", layer=16, kind="units")
```

The validation script uses all actual transcripts, observed times and released word-count features. It checks offsets, causal assignment, preservation of missing times, feature conservation, FIR/response row alignment, story-disjoint splits and Python 3.11 syntax. It does not assert that GPU extraction or tokenizer planning ran locally. Those checks require the real downloaded checkpoint/tokenizer and are part of the cluster execution.
