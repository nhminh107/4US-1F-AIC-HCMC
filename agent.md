# Project Agent Instructions

## 1. Mission

This repository builds an offline preprocessing pipeline for AIC25 video
retrieval. The offline pipeline enriches organizer-provided data and produces
reliable, queryable records for the future online retrieval pipeline.

The required enrichment stages are:

- video metadata ingestion;
- shot boundary extraction;
- additional keyframe extraction;
- OCR;
- transcript/ASR segmentation;
- object detection and object tracking;
- clip construction;
- frame, clip, and optional shot embeddings;
- frame, clip, and shot captions.

The offline pipeline must be modular. Every stage exposes reusable functions
that consume and return explicit data contracts. Do not couple stages through
undocumented temporary files or implicit global state.

## 2. Sources of Truth

Use the following priority when requirements conflict:

1. the user's current explicit request;
2. `data/pipeline_offline.md` for the intended offline architecture;
3. `data/describe_data.md` for organizer-data semantics and invariants;
4. database schema and shared data contracts once they have been approved;
5. existing implementation details.

Existing modules are under development and may not yet match the intended
architecture. Do not treat current code as authoritative when it conflicts
with the two design documents. Report the mismatch and implement toward the
documented target without unrelated rewrites.

## 3. Development Environment

The project runs on Ubuntu with Miniconda. Use the `DL_Env` environment for
every Python-related command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
```

Prefer the non-interactive form:

```bash
source ~/miniconda3/etc/profile.d/conda.sh \
  && conda activate DL_Env \
  && python <command>
```

Do not create another environment or install, upgrade, downgrade, or remove a
dependency without explicit permission. If a dependency is missing, report
the exact error, confirm whether it is declared, propose the exact install
command, and wait for approval.

Prefer GPU execution for supported ML workloads, but verify framework-level
CUDA availability first. Never assume `nvidia-smi` alone proves that a Python
framework can use the GPU. Do not silently fall back to CPU when GPU execution
was explicitly required.

## 4. Organizer Data Model

### 4.1 Canonical identity

`video_id` is the common join key. Its canonical form is `Lxx_Vyyy`, for
example `L21_V001`. IDs are not guaranteed to be continuous. Never infer a
missing video from a gap in numbering.

For every organizer keyframe, preserve the pair `(video_id, n)` throughout
ingestion, embedding, object import, database persistence, and index metadata.
Do not use a row position or filename alone as a global identity.

Any generated ID must be:

- deterministic when derived from stable source data;
- unique within its database scope;
- within the database column length and integer limits;
- reproducible across reruns;
- documented in one shared ID-construction function.

Do not use Python's process-randomized `hash()` for persistent IDs.

### 4.2 Coverage is intentionally uneven

The metadata, keyframe maps, CLIP features, and object results cover 873 video
IDs and 177,321 logical keyframes. Local MP4 files and JPG keyframes cover only
54 IDs and 10,126 images.

Therefore:

- missing local `.mp4` or `.jpg` files do not imply missing derived metadata;
- metadata-only ingestion must remain possible;
- stages requiring media must mark the record unavailable/skipped with a clear
  reason, not delete or fabricate the corresponding logical record;
- summaries must distinguish logical coverage from locally available media.

### 4.3 Keyframe alignment

For keyframe number `n`:

```text
map-keyframes/<video_id>.csv row with n
    <-> keyframes/<video_id>/<n:03d>.jpg, when locally available
    <-> clip-features-32/<video_id>.npy row n - 1
    <-> objects-aic25-b1/objects/<video_id>/<n:03d>.json
```

The CSV field `n` starts at 1. NumPy row indexes start at 0. Always convert
explicitly using `row_index = n - 1`; never rely on iteration order without
validating `n`.

Before processing or indexing a video, validate when applicable that:

- CSV row count equals CLIP embedding row count;
- object JSON count equals CSV row count;
- local JPG count equals CSV row count for locally complete videos;
- `(video_id, n)` values are unique;
- expected files map to the same logical keyframe.

Do not assume keyframes were sampled at a fixed interval. Their time gaps are
irregular and the original selection algorithm is unknown.

## 5. Units, Shapes, and Coordinate Conventions

### 5.1 Time and frame indexes

Organizer keyframe maps contain:

- `n`: 1-based keyframe number;
- `pts_time`: presentation timestamp in seconds;
- `fps`: mapping frame rate;
- `frame_idx`: 0-based source-video frame index.

Pipeline/database contracts use field names with explicit units. Convert
seconds to milliseconds at a single boundary:

```text
timestamp_ms = round(pts_time * 1000)
```

Do not store seconds in a field ending with `_ms`. Do not derive the canonical
timestamp only from `frame_idx / fps` when `pts_time` is available; minor
differences caused by PTS and rounding are expected.

All temporal intervals use half-open or closed semantics consistently within
a module and document that choice. At minimum enforce:

```text
start_ms >= 0
end_ms > start_ms
frame_idx >= 0
fps > 0
```

### 5.2 Bounding boxes

Organizer object JSON stores boxes as:

```text
[y_min, x_min, y_max, x_max]
```

All coordinates are normalized to `[0, 1]`. Shared pipeline contracts and
database records use named fields:

```text
x_min, x_max, y_min, y_max
```

Conversion must be explicit. Never swap the TensorFlow Hub source order with
the shared contract order. Validate `x_min < x_max` and `y_min < y_max`.

Object JSON values are strings. Parse scores, labels, and coordinates into
their intended numeric types before filtering or persistence. Do not interpret
all 100 predictions in a file as confirmed objects; apply and record an
explicit confidence threshold.

### 5.3 Embeddings

Organizer CLIP feature files have shape `(K, 512)`, use `float16`, and are
already approximately L2-normalized. Load large arrays with:

```python
numpy.load(path, mmap_mode="r")
```

Before adding vectors to a search index:

- convert to a contiguous NumPy array of `float32`;
- enforce shape `(batch_size, embedding_dimension)`;
- reject NaN, infinity, empty, or zero vectors;
- normalize L2 before inner-product cosine search;
- verify the vector dimension matches the target index;
- preserve the exact metadata mapping back to `(video_id, n)` or the generated
  frame/clip/shot ID;
- ensure persistent FAISS IDs are unique and fit signed `int64`.

Do not silently pad, truncate, or substitute missing vectors with zero vectors.

## 6. Shared Pipeline Contracts

Create or update shared contracts before wiring a new stage into the pipeline.
Contracts must use English field names, type hints, explicit optional fields,
and unit-bearing names such as `start_ms` and `timestamp_ms`.

The principal stage boundaries are:

```text
Video -> list[Shot]
Shot -> list[Frame]
Frame -> list[OCRResult]
Video -> list[TranscriptSegment]
Frame -> list[ObjectDetection]
Shot -> object detections + tracks
Shot -> list[ClipWindow]
Frame -> normalized frame embedding
ClipWindow -> normalized clip embedding
list[clip embedding for one shot] -> optional shot embedding
Frame | ClipWindow | Shot -> Caption
```

Rules for contracts:

- distinguish organizer-provided frames from newly extracted frames;
- distinguish persisted keyframes from ephemeral tracking samples;
- use `Path` for local filesystem paths and strings for URLs;
- keep normalized coordinates as floats;
- keep model name, model version, prompt/config version, and language when
  relevant;
- represent generated database identity values as optional before persistence;
- avoid passing raw dictionaries between stable modules when a typed contract
  exists;
- do not let a module mutate its input contract in place.

## 7. Offline Stage Requirements

### 7.1 Video ingestion

Ingest YouTube metadata from
`media-info-aic25-b1/media-info/<video_id>.json`. Parse `publish_date` from
`DD/MM/YYYY` and convert `length` seconds into `duration_ms`. Preserve source
URLs and keyword lists. Support records without a local MP4.

### 7.2 Shot extraction

Use a shot-boundary model such as TransNetV2. A shot is identified by its video
and temporal/frame boundaries; a separate shot media file is not required.
Output ordered, non-overlapping shot records and validate all boundaries.

### 7.3 Additional keyframe extraction

Consume one shot and return frame contracts. Save only selected keyframe images
under a deterministic `data` path. Avoid frames that are blurred, black, too
close to transitions, or near-duplicates. Do not overwrite organizer keyframes
or confuse their `n` values with newly generated frame identities.

### 7.4 OCR

Consume one frame and return zero or more OCR records. Each record contains
recognized text, optional language, and one normalized box. Detection and
recognition transformations must preserve box alignment. Do not write crops to
disk merely to pass them between functions unless debug output is explicitly
requested.

### 7.5 Transcript

Consume one locally available video and return timestamped transcript segments.
Keep preprocessing, language selection, ASR, alignment, and segmentation
reproducible. Preserve empty/no-speech outcomes rather than fabricating text.

### 7.6 Object detection

Prefer organizer-provided detector classes when importing supplied object
results. If another model is introduced, define and test an explicit mapping to
the canonical class vocabulary. Store the model identity and confidence
threshold used for filtering.

### 7.7 Tracking

Consume a shot and sample frames internally for YOLO26 tracking. Tracking-only
samples are ephemeral: do not save image files or persist them as normal
keyframes or object detections. Return track summaries and independent YOLO
observations through contracts. Each observation stores its frame index,
timestamp, normalized bounding box, and confidence without referencing
`ObjectDetection`. Record model, tracker, mapping, sampling, and configuration
metadata needed for reproduction.

### 7.8 Clip extraction

A long shot may contain multiple actions and can be split into shorter clip
windows. A value near 10 seconds is a heuristic, not a hidden constant: expose
it through configuration and record it in run metadata. A short shot may have
no child clips; downstream embedding logic must handle that case explicitly.

### 7.9 Embedding

Frame embeddings use the preprocessing required by their own CLIP-compatible
model, not OCR-enhanced images. Confirm model compatibility before reusing the
organizer's supplied embeddings.

Clip embeddings represent the semantic content of a temporal window. Shot
embeddings, when needed, are derived deterministically from embeddings belonging
to the same shot, for example by mean pooling. Do not rerun a model merely to
produce a pooled shot vector.

Handle shots without child clips using an explicit, documented branch rather
than an accidental empty mean.

### 7.10 Captioning

Use a vision-language model to caption a frame, clip, or shot. A caption must
target exactly one of these entity types. Store model/prompt versions and, when
available, structured JSON alongside free text. Captions should describe visible
evidence and avoid unsupported speculation.

## 8. Persistence and Index Consistency

Treat relational metadata, vector indexes, and text indexes as coordinated but
separate stores.

- Persist source identity and relational metadata in PostgreSQL.
- Store embedding vectors in the vector index, not in relational metadata
  fields unless the approved schema explicitly requires it.
- Persist a mapping record for every indexed vector.
- Keep OCR, transcript, and caption source records traceable before building a
  text-search representation.
- Never delete or replace an existing database, FAISS index, or generated
  dataset artifact without explicit permission and a backup plan.

Vector insertion is not complete until both the index entry and its metadata
mapping are consistent. Validate before committing:

```text
index dimension == embedding dimension
FAISS IDs are unique
source IDs exist and are unique
index version/model metadata are present
number of vectors == number of mappings
```

Design rerunnable stages to be idempotent. Prefer upsert/checkpoint behavior
based on stable IDs and processing versions. Never silently duplicate records
on retry.

## 9. Performance and Reproducibility

- Stream videos and metadata; do not load the entire dataset into memory.
- Use memory mapping for organizer `.npy` files.
- Batch GPU inference within measured memory limits.
- Load large models once per worker/process, not inside per-frame loops.
- Use inference mode and evaluation mode for PyTorch inference.
- Make batch size, thresholds, clip duration, sampling rate, model version, and
  index version configuration values.
- Persist enough run metadata to reproduce generated artifacts.
- Log counts for processed, skipped, failed, and unavailable-media records.
- A single corrupt/missing record should be reported with identity and reason;
  it should not silently invalidate the remainder of a large batch.

## 10. File and Data Safety

The `data/` tree contains large organizer data and derived artifacts. Treat it
as read-only unless the task explicitly asks a stage to create a documented
output. Never rename, move, overwrite, or delete source videos, organizer
keyframes, maps, embeddings, object JSON files, model weights, database files,
or FAISS indexes without explicit permission.

Generated outputs must:

- live under a documented output directory;
- never overwrite organizer-provided files;
- use atomic writes where a partial artifact would be dangerous;
- include source/model/index version information when practical;
- be excluded from Git when large, reproducible, or environment-specific.

Never expose `.env`, credentials, API keys, tokens, or connection passwords.

## 11. Coding Rules

- Write source code, symbols, filenames, comments, docstrings, and commit
  messages in English unless an existing module has a stronger convention.
- Follow existing style where it does not conflict with these instructions.
- Use PEP 8, four-space indentation, type hints, and focused functions.
- Prefer `pathlib.Path`, context managers, explicit exceptions, and f-strings.
- Keep orchestration separate from model inference, parsing, persistence, and
  index management.
- Keep route/UI concerns out of offline processing modules.
- Do not add an abstraction until at least one real pipeline boundary needs it.
- Do not catch `Exception` and continue silently. Preserve context and identify
  the failed video/frame/shot/clip.
- Do not introduce a second representation of an existing shared contract.

Each functional module should expose a reusable programmatic entry point. A CLI,
demo, or README may wrap that API, but must not be the only way to use it.

## 12. Testing and Validation

Test at the smallest meaningful level first:

1. parsers and ID/unit conversion tests;
2. contract and validation tests;
3. focused module tests with tiny fixtures;
4. database/index integration tests;
5. a small end-to-end pipeline smoke test;
6. large batch/GPU runs only when explicitly needed.

Tests must cover critical data invariants:

- `n` to NumPy row `n - 1` alignment;
- CSV/embedding/object count agreement;
- seconds-to-milliseconds conversion;
- bounding-box order conversion;
- invalid coordinates, intervals, dimensions, and zero vectors;
- duplicate persistent IDs;
- missing local media with valid metadata;
- short shots with no clips;
- index/mapping consistency;
- retry/idempotency behavior.

Database integration tests must clearly state whether they retain or clean up
test records. Use unmistakable generated IDs and never delete records outside
the test's own namespace. Do not claim a database/index test passed when only a
mock was exercised.

Use the project's configured test runner. If no more specific configuration is
available, run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh \
  && conda activate DL_Env \
  && python -m unittest
```

For Python files without tests, at minimum run `python -m compileall` on the
changed path in `DL_Env`.

## 13. Definition of Done

An offline-pipeline change is complete only when all applicable items hold:

- the stage accepts and returns the approved contracts;
- organizer-data identity, units, shapes, and coordinate conventions are
  preserved;
- missing local media is handled explicitly;
- generated output cannot overwrite source data;
- model/config/index versions are recorded where relevant;
- database and vector/text index mappings remain consistent;
- rerun behavior is defined and does not silently duplicate data;
- focused tests and syntax checks were actually executed in `DL_Env`;
- GPU availability was verified when execution depended on CUDA;
- changed files, commands run, results, and remaining limitations are reported;
- no dependency, secret, dataset, checkpoint, database, or index was modified
  outside the authorized scope.
