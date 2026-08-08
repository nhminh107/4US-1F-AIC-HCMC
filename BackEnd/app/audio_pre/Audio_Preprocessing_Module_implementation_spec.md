# Audio Preprocessing Module

> **Status:** Implementation-ready specification\
> **Scope:** Video → normalized shot-level audio → speech detection →
> `AudioSegment`\
> **Primary consumer:** ASR module\
> **Module does not perform transcription.**

------------------------------------------------------------------------

## 1. Vai trò trong Pipeline

Audio Preprocessing Module nằm giữa các module xử lý video/shot và ASR.

``` text
Video
 ├── [Visual] Shot Detection → Keyframe Extraction → Object Detection → ...
 │
 └── [Audio]
       │
       ▼
   Audio Preprocessing
       │
       ├── Audio Extraction
       ├── Shot-based Segmentation
       ├── Normalization
       └── VAD
       │
       ▼
   list[AudioSegment]
       │
       ▼
      ASR
       │
       ▼
TranscriptSegmentResult
       │
       ▼
Metadata Fusion
```

### Input contract

``` python
VideoMetadata
list[ShotMetadata]
```

-   `VideoMetadata` cung cấp `video_path` và `video_id`.
-   `ShotMetadata` cung cấp `shot_id`, `shot_index`, `start_ms`,
    `end_ms`.
-   Temporal boundary của shot là responsibility của module tạo
    `ShotMetadata`, không phải của Audio Preprocessing.

### Processing order

``` text
Audio Extraction
    ↓
Shot-based Segmentation
    ↓
Normalization
    ↓
VAD
    ↓
AudioSegment construction
    ↓
JSON export/cache
```

### Output contract

Public output:

``` python
list[AudioSegment]
```

Mỗi `AudioSegment.audio_path` trỏ tới một normalized WAV file sẵn sàng
cho ASR.

Module này **không transcribe**, **không language-detect**, và **không
speaker-diarize**.

------------------------------------------------------------------------

## 2. Quyết định thiết kế

  Quyết định             Contract
  ---------------------- -------------------------------------------
  Input                  `VideoMetadata + list[ShotMetadata]`
  Primary output         `list[AudioSegment]`
  Output artifact        WAV files trên disk
  Segmentation           1 shot → 1 audio segment
  Boundary               Exact `ShotMetadata.start_ms/end_ms`
  Padding                `0 ms`
  Overlap                None
  Container              WAV
  Codec                  PCM
  Bit depth              16-bit
  Sample rate            **16,000 Hz --- HARD REQUIREMENT**
  Channels               **Mono (1 channel) --- HARD REQUIREMENT**
  Volume normalization   Peak normalization
  Peak target            **1.0 (0 dBFS)**
  Silent signal          Không amplify
  VAD                    `webrtcvad`
  Transcription          Không thuộc module này

### Temporal invariant

``` text
AudioSegment.start_ms == ShotMetadata.start_ms
AudioSegment.end_ms   == ShotMetadata.end_ms
```

Module **không clamp, sửa, hoặc suy diễn lại** shot boundaries.

------------------------------------------------------------------------

## 3. Cấu trúc module

``` text
audio_preprocessing/
├── __init__.py
├── extractor.py          # Extract full-video audio
├── vad.py                # Speech detection
├── normalizer.py         # Format conversion + peak normalization
├── schemas.py            # AudioSegment
├── exporter.py           # JSON/cache export
├── utils.py              # Shared helpers
├── run_preprocessing.py  # Batch entrypoint
├── output/
│   └── .gitkeep
└── README.md
```

### Responsibility

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `extractor.py`                      Extract full audio track from
                                      source video into temporary WAV

  `normalizer.py`                     Resample, mono conversion, peak
                                      normalization, 16-bit PCM output

  `vad.py`                            Detect speech and return `bool`

  `schemas.py`                        Define immutable `AudioSegment`

  `exporter.py`                       Serialize successful `AudioSegment`
                                      list to JSON

  `utils.py`                          Path generation, duration,
                                      validation, cleanup helpers

  `run_preprocessing.py`              Batch CLI / orchestration
                                      entrypoint

  `__init__.py`                       Public package exports only

  `README.md`                         Module documentation
  -----------------------------------------------------------------------

### Responsibility boundary

-   `extractor.py` does **not** process `ShotMetadata`.
-   Segmentation does **not** normalize audio.
-   `normalizer.py` does **not** perform VAD.
-   `vad.py` does **not** cut audio.
-   `AudioSegment.has_speech` is a speech result, not an audio-existence
    flag.

------------------------------------------------------------------------

## 4. Schema nội bộ

``` python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioSegment:
    segment_id: str
    video_id: str
    shot_id: str
    start_ms: int
    end_ms: int
    audio_path: Path
    sample_rate: int
    has_speech: bool
    language_hint: str | None
```

### Field contract

  Field             Contract
  ----------------- ---------------------------------------------------
  `segment_id`      Deterministic: `{video_id}_shot{shot_index:04d}`
  `video_id`        Copied from `VideoMetadata.video_id`
  `shot_id`         Copied from `ShotMetadata.shot_id`
  `start_ms`        Exact value from `ShotMetadata.start_ms`
  `end_ms`          Exact value from `ShotMetadata.end_ms`
  `audio_path`      Absolute path to an existing normalized WAV
  `sample_rate`     Always `16000`
  `has_speech`      Result of VAD only
  `language_hint`   Optional pass-through hint; no language detection

### `has_speech` semantics

`has_speech` **does not mean that audio exists**.

``` text
audio signal exists + speech exists
    → has_speech = True

audio signal exists + no speech
    → has_speech = False

silence
    → has_speech = False
```

Music, environmental noise, or other non-speech audio may exist while
`has_speech=False`.

`AudioSegment` only represents a successfully processed shot, so
`audio_path` must never point to a missing or failed artifact.

------------------------------------------------------------------------

## 5. Pipeline chi tiết

### 5.1 Audio Extraction --- `extractor.py`

Input:

``` python
VideoMetadata
```

Output:

``` python
Path  # temporary full-video WAV
```

Processing:

``` text
video_path
    ↓
check audio stream
    ↓
ffmpeg
    ↓
{video_id}_raw.wav
```

The extractor:

-   extracts the full audio track once;
-   does not clip by shot;
-   creates a temporary WAV;
-   does not perform VAD;
-   does not perform shot segmentation.

#### No audio stream

If the video contains no audio stream:

``` text
warning log
return []
```

No fake `AudioSegment` is created.

#### Extraction failure

If an audio stream exists but FFmpeg extraction fails:

``` text
raise exception
```

Do not convert extraction errors into `[]`.

------------------------------------------------------------------------

### 5.2 Raw WAV reuse

`{video_id}_raw.wav` is an intermediate artifact.

If it already exists, validate it before reuse.

Required validation:

``` text
✓ file exists
✓ file is readable
✓ valid WAV
✓ contains audio samples
✓ duration can be determined
```

If valid:

``` text
reuse existing raw WAV
```

If invalid:

``` text
discard/rewrite invalid artifact
re-extract from source video
```

The raw WAV does **not** need to already be 16 kHz, mono, or 16-bit.
Those are normalized-output requirements.

------------------------------------------------------------------------

### 5.3 Shot validation

If the video has an audio stream, validate the complete
`list[ShotMetadata]` before starting shot segmentation.

Required checks for every shot:

``` text
start_ms >= 0
end_ms > start_ms
start_ms < audio_duration
end_ms <= audio_duration
```

If any shot is invalid:

``` text
raise exception
```

Do not partially process a video with invalid temporal input.

### Boundary ownership

The audio module does not repair invalid shot boundaries.

If a shot exceeds the actual audio duration, the shot is invalid and
preprocessing fails. The module responsible for producing `ShotMetadata`
is responsible for producing correct temporal boundaries.

### No clamping

Never perform:

``` python
end_ms = min(end_ms, audio_duration)
```

The original `ShotMetadata` values must remain unchanged.

------------------------------------------------------------------------

### 5.4 Shot-based Segmentation

For every valid shot:

``` text
raw full-video WAV
        +
start_ms / end_ms
        ↓
intermediate shot WAV
```

Rules:

-   exactly one audio segment per shot;
-   exact `start_ms/end_ms`;
-   `0 ms` padding;
-   no overlap added by this module;
-   segmentation output is intermediate/raw audio;
-   normalization happens in the next step.

The module trusts FFmpeg's successful temporal extraction and does
**not** perform an additional duration verification after segmentation.

#### Shot-level segmentation failure

A valid shot may still fail during FFmpeg segmentation.

Behavior:

``` text
segmentation failure
    ↓
delete partial/corrupted shot WAV if it exists
    ↓
log error
    ↓
skip shot
    ↓
continue remaining shots
```

This is **best-effort at shot-processing level**.

Example:

``` text
100 shots
 ├── 97 successful
 └── 3 failed

return 97 AudioSegment objects
```

Every failed shot must be logged with at least:

``` text
video_id
shot_id
start_ms
end_ms
error
```

------------------------------------------------------------------------

### 5.5 Normalization --- `normalizer.py`

Input:

``` text
intermediate shot WAV
```

Output:

``` text
normalized WAV:
    WAV
    PCM
    16-bit
    16 kHz
    mono
```

Exact processing order:

``` text
Intermediate shot WAV
        │
        ▼
1. Resample → 16,000 Hz
        │
        ▼
2. Convert/downmix → Mono
        │
        ▼
3. Convert to working floating-point representation
        │
        ▼
4. Peak normalization → target 1.0
        │
        ▼
5. Quantize/convert → 16-bit PCM
        │
        ▼
Normalized WAV
```

### Peak normalization

Let:

``` python
peak = max(abs(samples))
```

Then:

``` python
if peak > 0:
    samples = samples / peak
else:
    samples = samples
```

Target:

``` text
max(abs(samples)) = 1.0
```

for non-silent audio before int16 quantization.

No:

-   RMS normalization;
-   LUFS normalization;
-   loudness normalization;
-   dynamic compression;
-   `-1 dBFS` headroom.

### Silent signal

A signal is considered silent for normalization purposes when:

``` python
peak == 0
```

Do not amplify it.

This definition is only for normalization. It is **not** the definition
of speech.

### Quantization

Before converting to int16:

``` text
samples are floating point in [-1.0, 1.0]
```

The implementation must ensure samples do not exceed the valid PCM range
before conversion.

The resulting WAV must contain signed 16-bit PCM samples.

------------------------------------------------------------------------

### 5.6 VAD --- `vad.py`

Input:

``` text
normalized 16 kHz / mono / 16-bit PCM WAV
```

Output:

``` python
bool
```

Use:

``` text
webrtcvad
```

VAD is responsible only for detecting speech.

It must not:

-   trim silence;
-   split speech into smaller segments;
-   modify the WAV;
-   change shot boundaries.

### VAD input contract

Because `webrtcvad` requires:

``` text
sample rate = 16000 Hz
sample width = 16-bit
channels = mono
```

VAD must run only after normalization.

### Frame processing

Use valid WebRTC VAD frame durations:

``` text
10 ms
20 ms
30 ms
```

The implementation should use **30 ms frames** for this module to keep
processing simple and lightweight.

The final incomplete frame, if shorter than 30 ms, is ignored rather
than padded.

### VAD aggressiveness

Use:

``` python
webrtcvad.Vad(2)
```

as the default mode.

The value is an implementation constant for this module, not exposed
through `preprocess_video()`.

### `has_speech`

A segment is marked:

``` text
has_speech = True
```

when at least one complete 30 ms frame is classified as speech.

Otherwise:

``` text
has_speech = False
```

VAD does not change the segment audio file.

------------------------------------------------------------------------

### 5.7 AudioSegment Builder

For every successfully normalized and VAD-processed shot:

``` python
AudioSegment(
    segment_id=f"{video.video_id}_shot{shot.shot_index:04d}",
    video_id=video.video_id,
    shot_id=shot.shot_id,
    start_ms=shot.start_ms,
    end_ms=shot.end_ms,
    audio_path=normalized_wav.resolve(),
    sample_rate=16000,
    has_speech=has_speech,
    language_hint=language_hint,
)
```

The builder must use the original shot metadata for temporal fields.

------------------------------------------------------------------------

### 5.8 Exporter --- `exporter.py`

After processing:

``` text
list[AudioSegment]
        ↓
audio_segments.json
```

The JSON file is a cache/debug artifact.

The in-memory return value remains:

``` python
list[AudioSegment]
```

JSON serialization rules:

-   `Path` is serialized as a string;
-   all paths are absolute;
-   field names match `AudioSegment`;
-   list order follows the input shot order for successfully processed
    shots;
-   failed shots are absent from the list.

Example:

``` json
{
  "video_id": "video_001",
  "segments": [
    {
      "segment_id": "video_001_shot0001",
      "video_id": "video_001",
      "shot_id": "shot_0001",
      "start_ms": 0,
      "end_ms": 5200,
      "audio_path": "/abs/path/output/video_001/video_001_shot0001.wav",
      "sample_rate": 16000,
      "has_speech": true,
      "language_hint": "vi"
    }
  ]
}
```

------------------------------------------------------------------------

## 6. Public Interface

``` python
def preprocess_video(
    video: VideoMetadata,
    shots: list[ShotMetadata],
    output_dir: Path,
    language_hint: str | None = None,
) -> list[AudioSegment]:
    """
    Process all valid audio shots for one video.

    Returns only successfully processed AudioSegment objects.
    """
```

### `preprocess_video()` contract

``` text
No audio stream
    → warning
    → return []

Invalid shot metadata
    → raise exception

FFmpeg full-audio extraction failure
    → raise exception

Individual valid-shot processing failure
    → cleanup failed artifact
    → log error
    → skip shot
    → continue

Successful processing
    → AudioSegment
```

------------------------------------------------------------------------

### `preprocess_shot()`

``` python
def preprocess_shot(
    video: VideoMetadata,
    shot: ShotMetadata,
    raw_audio_path: Path,
    output_dir: Path,
    language_hint: str | None = None,
) -> AudioSegment:
    """
    Process one already-validated shot using an existing raw WAV.
    """
```

`preprocess_shot()` is intended for targeted reprocessing/debugging.

It must:

-   use the supplied `raw_audio_path`;
-   preserve `shot.start_ms/end_ms`;
-   segment exactly the shot boundary;
-   normalize the segment;
-   run VAD;
-   construct and return `AudioSegment`.

It must not alter `ShotMetadata`.

------------------------------------------------------------------------

## 7. Output File Naming

``` text
output/
└── {video_id}/
    ├── {video_id}_shot0001.wav
    ├── {video_id}_shot0002.wav
    ├── ...
    └── audio_segments.json
```

The raw intermediate file:

``` text
{video_id}_raw.wav
```

may temporarily exist during processing but **must be deleted after the
segmentation phase finishes**.

It is not considered a persistent module output.

### Naming rule

``` python
f"{video_id}_shot{shot_index:04d}.wav"
```

Example:

``` text
video001_shot0000.wav
video001_shot0001.wav
video001_shot0002.wav
```

`shot_index` comes from `ShotMetadata.shot_index`.

------------------------------------------------------------------------

## 8. Raw WAV Lifecycle

``` text
raw WAV exists?
    │
    ├── NO
    │    ↓
    │  extract
    │
    └── YES
         ↓
       validate
         │
      ┌──┴──┐
      │     │
    valid invalid
      │     │
    reuse  re-extract
      │     │
      └──┬──┘
         ↓
    segment shots
         ↓
    segmentation phase ends
         ↓
    DELETE raw WAV
```

The raw WAV is deleted even when some individual shots fail.

Example:

``` text
100 shots
97 successful
3 failed

97 normalized WAVs remain
raw WAV is deleted
3 failed WAV artifacts are deleted
```

If the process crashes before cleanup, a subsequent run may reuse the
raw WAV only after it passes the required validation.

------------------------------------------------------------------------

## 9. Error Handling

### Error classification

  -----------------------------------------------------------------------
  Condition                           Behavior
  ----------------------------------- -----------------------------------
  No audio stream                     Warning + return `[]`

  Existing raw WAV valid              Reuse

  Existing raw WAV invalid            Re-extract

  FFmpeg full-audio extraction        Raise exception
  failure                             

  Invalid `start_ms` / `end_ms`       Raise exception

  Shot boundary exceeds audio         Raise exception
  duration                            

  Individual valid-shot segmentation  Delete partial artifact + log +
  failure                             skip

  Individual normalization failure    Delete failed artifact + log + skip

  Individual VAD failure              Delete failed/invalid output
                                      artifact + log + skip

  JSON export failure                 Raise exception
  -----------------------------------------------------------------------

### Principle

Do not silently convert infrastructure/processing errors into "no
speech".

These states are distinct:

``` text
No audio
    ≠
No speech
    ≠
Processing failure
```

------------------------------------------------------------------------

## 10. Idempotency and Reprocessing

The module should be safe to run repeatedly for the same video/output
directory.

For raw extraction:

``` text
valid raw WAV exists
    → reuse
```

For normalized shot outputs:

``` text
existing normalized WAV
    → do not blindly trust it as a successful result
```

A shot should be considered successfully processed only when its output
artifact is valid and its `AudioSegment` can be constructed.

For deterministic reprocessing, failed/partial shot artifacts must be
removed before retrying.

`segment_id` and output filenames are deterministic, so rerunning the
same video does not create random duplicate filenames.

------------------------------------------------------------------------

## 11. Language Hint

`language_hint` is optional metadata passed through to ASR.

Examples:

``` text
"vi"
"en"
None
```

Rules:

-   preprocessing does not perform language detection;
-   preprocessing does not infer language from audio;
-   `language_hint` is copied into every generated `AudioSegment`;
-   `None` means no hint was supplied;
-   semantic language validation is outside this module's
    responsibility.

------------------------------------------------------------------------

## 12. CLI / Batch Entrypoint

`run_preprocessing.py` is the batch entrypoint.

The CLI should support at minimum:

``` text
python -m audio_preprocessing.run_preprocessing \
    --video <video_path> \
    --shots <shots_json> \
    --output-dir <output_dir> \
    [--language <language_hint>]
```

Responsibilities:

1.  Load `VideoMetadata`.
2.  Load `ShotMetadata` list.
3.  Call `preprocess_video()`.
4.  Report processing summary.
5.  Return non-zero exit code when `preprocess_video()` raises a fatal
    exception.

Suggested summary:

``` text
Video: video001
Shots requested: 100
Shots processed: 97
Shots skipped: 3
Speech segments: 41
Output directory: ...
```

The CLI is an orchestration layer and must not duplicate preprocessing
logic.

------------------------------------------------------------------------

## 13. Logging

Use Python's standard `logging` module.

Recommended levels:

``` text
INFO
    extraction started/completed
    raw WAV reused
    normalization completed
    preprocessing summary

WARNING
    video has no audio stream

ERROR
    individual shot processing failure
    cleanup failure
    unexpected processing error
```

Shot-level errors should include:

``` text
video_id
shot_id
shot_index
start_ms
end_ms
operation
error
```

Do not log full audio contents or large binary data.

------------------------------------------------------------------------

## 14. Dependencies

  -----------------------------------------------------------------------
  Package                 Purpose                 Requirement
  ----------------------- ----------------------- -----------------------
  `ffmpeg-python`         FFmpeg invocation       Requires system
                                                  `ffmpeg` binary

  `pydub`                 Audio loading,          Uses FFmpeg/libav
                          clipping, resampling,   
                          channel conversion      

  `webrtcvad`             Speech detection        CPU-only, lightweight
  -----------------------------------------------------------------------

Suggested versions:

``` text
ffmpeg-python>=0.2.0
pydub>=0.25.1
webrtcvad>=2.0.10
```

System requirement:

``` text
ffmpeg
```

must be installed and available on `PATH`.

`webrtcvad` requires:

``` text
16 kHz
16-bit PCM
mono
```

which is exactly the normalized VAD input contract.

------------------------------------------------------------------------

## 15. Testing Requirements

Tests must cover at least the following.

### 15.1 Extraction

``` text
✓ video with valid audio → raw WAV created
✓ video without audio → warning + []
✓ FFmpeg extraction failure → exception
✓ valid existing raw WAV → reused
✓ invalid existing raw WAV → re-extracted
```

### 15.2 Shot validation

``` text
✓ start_ms == 0
✓ end_ms > start_ms
✓ negative start_ms → exception
✓ end_ms <= start_ms → exception
✓ start_ms >= audio_duration → exception
✓ end_ms > audio_duration → exception
```

### 15.3 Segmentation

``` text
✓ one shot → one intermediate WAV
✓ exact shot boundaries passed to segmentation
✓ no padding
✓ no overlap
✓ segmentation failure → partial artifact removed
✓ segmentation failure → remaining shots continue
```

### 15.4 Normalization

``` text
✓ 44.1 kHz → 16 kHz
✓ 48 kHz → 16 kHz
✓ stereo → mono
✓ multi-channel → mono
✓ output is 16-bit PCM
✓ non-silent signal peak → 1.0 before quantization
✓ silent signal is not amplified
✓ no clipping outside int16 range
```

### 15.5 VAD

``` text
✓ valid normalized WAV accepted
✓ speech-containing segment → has_speech=True
✓ non-speech audio → has_speech=False
✓ silence → has_speech=False
✓ VAD does not modify WAV
✓ incomplete final frame is ignored
```

### 15.6 Schema

``` text
✓ segment_id deterministic
✓ temporal fields copied exactly
✓ audio_path absolute
✓ audio_path exists for successful segment
✓ sample_rate == 16000
✓ language_hint propagated
```

### 15.7 Export

``` text
✓ JSON created
✓ Path serialized as string
✓ field names stable
✓ successful segments preserved in shot order
✓ failed shots absent
```

------------------------------------------------------------------------

## 16. Acceptance Criteria

The module is considered complete only when all of the following are
true:

### Input

-   [ ] Accepts `VideoMetadata`.
-   [ ] Accepts `list[ShotMetadata]`.
-   [ ] Uses `ShotMetadata.start_ms/end_ms` as the temporal source of
    truth.

### Extraction

-   [ ] Extracts full-video audio only once when necessary.
-   [ ] Reuses a valid existing raw WAV.
-   [ ] Re-extracts an invalid raw WAV.
-   [ ] Returns `[]` for videos without an audio stream.
-   [ ] Raises on actual FFmpeg extraction failure.

### Segmentation

-   [ ] Produces at most one successful audio artifact per shot.
-   [ ] Uses exact shot boundaries.
-   [ ] Adds zero padding.
-   [ ] Does not clamp timestamps.
-   [ ] Rejects invalid shot boundaries before segmentation.
-   [ ] Skips individual processing failures.
-   [ ] Cleans failed partial WAV files.

### Normalization

Every successful output WAV is:

``` text
WAV
PCM
16-bit
16 kHz
mono
peak normalized
target peak = 1.0 for non-silent signal
```

### VAD

-   [ ] Runs only after normalization.
-   [ ] Uses WebRTC VAD.
-   [ ] Uses 30 ms frames.
-   [ ] Uses VAD mode 2.
-   [ ] Returns a boolean speech result.
-   [ ] Does not modify or trim audio.

### Schema

-   [ ] Returns `list[AudioSegment]`.
-   [ ] `segment_id` is deterministic.
-   [ ] `start_ms/end_ms` exactly match `ShotMetadata`.
-   [ ] `audio_path` is absolute and points to an existing normalized
    WAV.
-   [ ] `sample_rate == 16000`.
-   [ ] `has_speech` represents speech detection only.
-   [ ] `language_hint` is passed through unchanged.

### Storage

-   [ ] Raw full-video WAV is temporary.
-   [ ] Raw WAV is deleted after segmentation phase.
-   [ ] Failed partial shot artifacts are deleted.
-   [ ] Successful normalized WAVs remain.
-   [ ] `audio_segments.json` is generated after processing.

### Integration

-   [ ] `preprocess_video()` is the primary public entrypoint.
-   [ ] ASR can consume `AudioSegment.audio_path` directly.
-   [ ] Module does not perform transcription.
-   [ ] Module does not perform language detection.
-   [ ] Module does not perform speaker diarization.

------------------------------------------------------------------------

## 17. Non-Responsibilities

This module does **not**:

-   transcribe speech;
-   detect language;
-   perform speaker diarization;
-   perform text retrieval;
-   perform video-text retrieval;
-   modify visual shot boundaries;
-   extract keyframes;
-   perform object detection;
-   perform OCR;
-   perform query-side audio processing.

Those responsibilities belong to other modules in the larger Video-Text
Retrieval system.

------------------------------------------------------------------------

## 18. Implementation Invariants

The following invariants MUST hold:

``` text
1. Every successful AudioSegment has an existing normalized WAV.

2. Every normalized WAV is:
       WAV + PCM + 16-bit + 16 kHz + mono.

3. AudioSegment.start_ms/end_ms exactly equal ShotMetadata values.

4. No padding or temporal clamping is introduced.

5. has_speech means speech detected by VAD,
   not audio existence.

6. No audio stream → [].

7. Fatal input/extraction errors → exception.

8. Individual valid-shot processing failures →
   cleanup + log + skip.

9. Raw full-video WAV is temporary and deleted after segmentation.

10. segment_id and output filenames are deterministic.

11. Audio preprocessing never performs transcription.

12. Audio preprocessing never repairs invalid ShotMetadata.
```

------------------------------------------------------------------------

## 19. Reference End-to-End Algorithm

``` python
def preprocess_video(video, shots, output_dir, language_hint=None):
    # 1. Check whether source contains an audio stream.
    if not extractor.has_audio_stream(video.video_path):
        logger.warning("Video has no audio stream: %s", video.video_id)
        return []

    # 2. Extract or reuse full-video raw WAV.
    raw_wav = extractor.get_or_extract_raw_audio(
        video.video_path,
        video.video_id,
        output_dir,
    )

    try:
        # 3. Validate all shot boundaries.
        audio_duration_ms = extractor.get_duration_ms(raw_wav)
        validate_shots(shots, audio_duration_ms)

        results = []

        # 4. Process each shot independently.
        for shot in shots:
            try:
                intermediate_wav = segment_shot(
                    raw_wav,
                    shot.start_ms,
                    shot.end_ms,
                    output_dir,
                    video.video_id,
                    shot.shot_index,
                )

                normalized_wav = normalizer.normalize(intermediate_wav)

                has_speech = vad.detect(normalized_wav)

                segment = AudioSegment(
                    segment_id=f"{video.video_id}_shot{shot.shot_index:04d}",
                    video_id=video.video_id,
                    shot_id=shot.shot_id,
                    start_ms=shot.start_ms,
                    end_ms=shot.end_ms,
                    audio_path=normalized_wav.resolve(),
                    sample_rate=16000,
                    has_speech=has_speech,
                    language_hint=language_hint,
                )

                results.append(segment)

            except Exception as exc:
                cleanup_failed_shot_artifacts(...)
                logger.error(...)
                continue

        # 5. Export successful results.
        exporter.save_json(results, output_dir)

        return results

    finally:
        # Raw WAV is an intermediate artifact.
        cleanup_raw_wav(raw_wav)
```

The exact helper names may differ in implementation, but the
**observable behavior and contracts above must remain unchanged**.

------------------------------------------------------------------------

## 20. Handoff to Coding Agent

The coding agent implementing this module MUST treat this document as
the implementation contract.

The agent may choose internal helper names, class organization, and
low-level FFmpeg invocation details, but MUST NOT change:

``` text
Input contract
Output contract
Temporal semantics
Normalization format
VAD semantics
Error classification
Artifact lifecycle
AudioSegment schema
```

If an implementation detail is not specified here, prefer the simplest
implementation consistent with the existing project architecture and
dependencies rather than introducing a new framework or subsystem.
