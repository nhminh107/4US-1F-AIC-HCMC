# Pipeline Performance and Reliability Playbook

This document records the optimizations implemented for the AIC25 offline
pipeline. It is intended both as an operational reference and as context for
future agents. It describes only the changes present in this repository and
the measurements made during the completed 873-video run on an RTX A6000
(48 GB VRAM).

## Scope and final validated output

The completed run produced:

| Entity | Count |
| --- | ---: |
| Videos | 873 |
| Frames | 366,380 |
| Official frames | 177,321 |
| Shots | 96,796 |
| Clips | 115,835 |
| Object tracks | 228,070 |
| Track observations | 1,240,464 |
| OCR rows | 1,270,545 |
| Frame embeddings | 366,380 |
| Clip embeddings | 115,835 |
| Shot embeddings | 96,796 |

The embedding completion check verifies that the number of DB mappings equals
the number of vectors in each FAISS index. Do not treat a job as complete
until this check passes.

## Principles

1. Persist each durable unit before proceeding: a video for OCR/tracking, and
   a video for clip plus shot embedding.
2. Keep model-heavy work bounded. Parallelism is useful only while its VRAM,
   CPU decode, database-write, and FAISS-write owners are explicit.
3. Never allow two processes to write the same FAISS directory at once.
4. Preserve local media paths in `frame.frame_path`; public image URLs belong
   in `frame.img_url`. Replacing local paths breaks media-dependent stages.
5. Measure precision changes on actual data before adopting them globally.

## Tracking

### Optimizations implemented

- Tracking uses YOLO's batch tracking path. Batches are controlled by
  `TrackingConfig.batch_size`; this amortizes model invocation overhead across
  sampled frames.
- The decoder now yields PyAV frames first. Conversion to BGR NumPy arrays is
  deferred until a frame passes the tracking sampling/shot checks. Previously,
  every decoded frame was converted even when it was not sampled.
- `BackEnd.app.pipeline.tracking_resume` resumes only videos that have no
  persisted `ObjectTrack`, splits them into disjoint round-robin shards, and
  runs independent processes. Each worker commits through the normal
  `track_video` path; no partial video is marked complete.

### Why it was faster

The dominant prior cost was CPU decode and BGR conversion for frames that YOLO
never received. Deferring conversion removes that unnecessary copy. Splitting
non-overlapping videos lets one GPU worker wait on decode while another can
submit work, but this must be benchmarked for the selected GPU and codec.

### Safety checks

- A resume worker selects already completed videos from persisted tracks, not
  from naming assumptions.
- Track rows and observations remain written by existing database methods.
- Validate after a run:

```sql
SELECT
  count(*) AS stored_observations,
  sum(observation_count) AS declared_observations
FROM objecttrack;
```

The two values must match. The validated full run had 1,240,464 for both.

### Operational command

```bash
python -m BackEnd.app.pipeline.tracking_resume \
  --start-video-id L26_V295 --workers 2
```

Choose a start boundary only after querying persisted tracks. Do not run a
second writer against videos currently being tracked.

## OCR

### Data scope

OCR is intentionally run for `source=official` frames only. This avoids
duplicating OCR for extracted frames while preserving all organizer keyframes.
It processed 177,321 official frames and produced 1,270,545 text rows.

### Optimizations implemented

- `ocr_parallel` partitions pending videos into disjoint round-robin shards
  and starts isolated OCR worker subprocesses. Each subprocess creates its own
  Paddle/VietOCR model state, avoiding unsafe model sharing.
- Worker parameters are configurable: frame chunk size, detection batch size,
  recognition batch size, input source, and precision.
- Each completed video is logged and persisted independently. Resume combines
  all supplied append-only logs and removes stale OCR rows only for pending
  videos before rerunning them.
- `PostgreManager.add_ocr_records()` inserts one video's OCR rows in a single
  transaction. It validates every frame foreign key and rejects duplicate
  `(frame_id, n)` records before the insert. This replaces per-row database
  transactions and materially reduces PostgreSQL overhead.
- `benchmark_ocr_precision` runs read-only FP32/FP16 comparisons and records
  timing, peak PyTorch memory, frame-level exact agreement, and text multiset
  Jaccard similarity.

### Precision result on the A6000 run

On 256 frames from `L25_V009`:

| Mode | Time | Frames/s | OCR rows | Peak PyTorch VRAM |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 136.46 s | 1.876 | 2,404 | 1.51 GB |
| FP16 | 157.10 s | 1.630 | 2,405 | 0.64 GB |

FP16 reduced memory but was slower on this exact stack. Its output comparison
was 96.094% exact frame match and 0.99213 text multiset Jaccard. Therefore the
production run remained FP32. Re-benchmark after changing Paddle, VietOCR,
CUDA, or GPU before changing this choice.

### Operational command

```bash
python -m BackEnd.app.pipeline.ocr_parallel \
  --workers 3 \
  --frame-source official \
  --precision fp32 \
  --frame-chunk-size 64 \
  --detection-batch-size 24 \
  --recognition-batch-size 192 \
  --resume-log artifacts/ocr_previous.log
```

Start conservatively, watch `nvidia-smi`, and raise batch sizes only after a
stable sample. More workers raise aggregate VRAM use because each loads OCR
models independently.

## Embedding and FAISS

### Decode and CLIP throughput

- The PyAV clip decoder chooses the nearest requested frame while streaming.
  It no longer materializes RGB arrays for every candidate frame in a decode
  interval; RGB conversion occurs only for selected timestamps and is cached
  when multiple requests select the same frame.
- Clip work is bounded with `max_clips_per_unit=64` by default. This avoids an
  unbounded decoded-frame map for a long video.
- Two bounded decode workers prefetch up to two work units while GPU CLIP
  encoding consumes the previous unit. Output order remains deterministic.
- CLIP image encoding uses the configured project batch size rather than an
  unrelated hard-coded default.

### Durable clip-to-shot aggregation

Shot embedding no longer depends on a large in-memory clip cache surviving for
the entire process. It proceeds per video:

```text
embed clips -> persist DB mappings + FAISS vectors -> reconstruct vectors if needed
            -> aggregate shots -> persist shot mappings + FAISS vectors -> release cache
```

`FAISS_Manager.reconstruct_clip_vectors()` returns persisted vectors in the
requested mapping order. Consequently a restarted process can finish shot
embeddings for previously persisted clips without recomputing CLIP vectors.

### Exclusive FAISS writer and final verification

`embedding_only` acquires an exclusive lock for the selected FAISS directory.
It processes all frame embeddings, then clip plus shot embeddings per video,
and finally verifies DB-to-FAISS IDs and counts for frame, clip, and shot
indexes. This prevents duplicate IDs and inconsistent index metadata caused by
concurrent writers.

### Operational command

```bash
python -m BackEnd.app.pipeline.embedding_only \
  --faiss-dir artifacts/faiss
```

For a planned OCR-to-embedding handoff, use
`scripts/handoff_embedding_after_ocr.sh`. It requires the old orchestrator to
be stopped, waits for OCR worker completion, copies OCR result state to a
durable location, verifies the expected video count, and then starts the
exclusive embedding writer.

## Scheduling on a 48 GB GPU

The practical schedule used for the completed run was:

```text
tracking workers  -> complete and validate
OCR workers       -> complete and validate
embedding writer  -> complete and validate DB/FAISS
```

Some overlap is possible, but it is not automatically faster:

- OCR consumes most VRAM and benefits from multiple workers.
- Tracking and clip embedding may be limited by CPU video decode rather than
  GPU utilization.
- FAISS embedding writes must remain exclusive.

The handoff tooling supports starting embedding after verified OCR completion
while a stopped parent orchestrator remains stopped. Do not use it to create a
second concurrent FAISS writer.

## Cloudflare image URLs

The uploader result is a TSV of `(frame_id, public_url)`. The final database
update writes URLs to `frame.img_url` only for official frames. In the
completed upload, all 177,321 official frames had a non-null Cloudflare URL.

If a post-upload `curl` validation fails after a successful transaction, first
check the database count before treating it as data loss:

```sql
SELECT
  count(*) AS official_frames,
  count(*) FILTER (WHERE img_url LIKE 'https://YOUR_PUBLIC_DOMAIN/%') AS public_urls
FROM frame
WHERE source = 'official';
```

The observed failure was a Bash TSV parsing bug in the post-commit verifier;
it attempted to resolve a frame ID as a host. The upload manifest had 177,321
successes, zero failures, and the committed URL count was 177,321.

## Backup and recovery

Before destructive DB or FAISS work:

1. Stop writers cleanly whenever possible.
2. Copy FAISS files and `faiss_index.json` together.
3. Export database state with `pg_dump --no-owner --no-privileges`.
4. Generate SHA-256 checksums and verify them after transfer.
5. Restore to a fresh database first, then check entity counts and mapping
   counts before pointing the application at it.

Generated dumps, FAISS indexes, logs, and downloaded artifacts are deliberately
ignored by Git under `artifacts/`. Store them in durable object storage or a
separate backup location; source control should contain code, migrations,
scripts, and this operational documentation only.

## Regression coverage

Focused tests cover streaming frame selection, FAISS vector reconstruction
ordering, OCR transaction behavior, OCR sharding/resume parsing, OCR precision
comparison, and embedding handoff validation. Run them with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
PYTHONPATH=. pytest -q \
  BackEnd/tests/embedding/test_clip_embedding_service.py \
  BackEnd/tests/embedding/test_work_planner.py \
  BackEnd/tests/embedding/test_gpu_batching_and_precision.py \
  BackEnd/tests/embedding/test_streaming_decoder_selection.py \
  BackEnd/tests/pipeline/test_embedding_pipeline.py \
  BackEnd/tests/pipeline/test_embedding_only.py \
  BackEnd/tests/pipeline/test_ocr_pipeline.py \
  BackEnd/tests/pipeline/test_ocr_parallel.py \
  BackEnd/tests/pipeline/test_benchmark_ocr_precision.py \
  BackEnd/tests/database/test_faiss_manager.py \
  BackEnd/tests/ocr/test_ocr_service.py
```

Use a GPU benchmark separately for throughput conclusions. Unit tests validate
control flow and data contracts; they do not establish production performance.
