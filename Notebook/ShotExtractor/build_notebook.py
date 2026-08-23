"""Build the standalone Kaggle Shot Extractor notebook from repository source."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("shot_extractor_kaggle.ipynb")
MODEL_SOURCE = (ROOT / "BackEnd/app/shot_extractor/transnetv2_model.py").read_text(encoding="utf-8")


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Shot Extractor — TransNetV2 (Kaggle batch preprocessing)\n\nNotebook này chỉ đọc video/bảng input và chỉ ghi vào `/kaggle/working`. Nó **không** kết nối PostgreSQL, Cloudflare/R2/S3 hoặc upload dữ liệu.\n\nAttach một Kaggle Dataset có `videos.csv` (hoặc JSONL/Parquet đã cấu hình), video `.mp4`, và checkpoint PyTorch đã convert `models/transnetv2-pytorch-weights.pth`. Kết quả là các row `shot` để DBA import có chủ đích vào PostgreSQL. Nếu segmentation thay đổi, notebook không xóa shot cũ vì chúng có thể đã được tham chiếu bởi frame/clip/caption/tracking.\n"""),
    code(r'''# Preflight: Kaggle normally includes these dependencies; do not reinstall PyTorch/CUDA here.
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import requests
import torch

for executable in ("ffmpeg", "ffprobe"):
    result = subprocess.run([executable, "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{executable} is required and was not usable: {result.stderr[:300]}")
    print(result.stdout.splitlines()[0])
print("Python packages: numpy", np.__version__, "pandas", pd.__version__, "torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "CUDA:", torch.version.cuda)
'''),
    code(r'''# Configuration. Change only this cell for a Kaggle batch/range.
INPUT_DIR = Path("/kaggle/input/shot-extractor-input")
OUTPUT_DIR = Path("/kaggle/working/shot-extractor-output")
VIDEOS_TABLE = INPUT_DIR / "videos.csv"  # .csv, .jsonl, or .parquet
WEIGHTS_PATH = INPUT_DIR / "models" / "transnetv2-pytorch-weights.pth"

VIDEO_IDS: list[str] | None = None
START_OFFSET = 0
MAX_VIDEOS: int | None = None
REQUIRE_GPU = True
THRESHOLD = 0.5
MIN_SHOT_DURATION_MS = 500
WINDOW_BATCH_SIZE = 8
DECODE_MODE = "auto"  # auto | memory | memmap
MAX_DECODE_RAM_MB = 768
USE_AMP = False  # FP32 is the default required for backend parity.
FORCE_REPROCESS = False
CONTINUE_ON_ERROR = False
RUN_REAL_PIPELINE = False
RUN_ID = "default"
NOTEBOOK_LOGIC_VERSION = "shot-extractor-kaggle-v1"
ALLOWED_VIDEO_SUFFIXES = {".mp4"}

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("REQUIRE_GPU=true but CUDA is unavailable. Enable a Kaggle GPU before processing.")
if DECODE_MODE not in {"auto", "memory", "memmap"}:
    raise ValueError("DECODE_MODE must be auto, memory, or memmap")
if not 0 <= THRESHOLD <= 1 or MIN_SHOT_DURATION_MS < 0 or WINDOW_BATCH_SIZE < 1:
    raise ValueError("Invalid detector configuration")
if not INPUT_DIR.is_dir() or not VIDEOS_TABLE.is_file():
    raise FileNotFoundError(f"Attach input dataset then set INPUT_DIR/VIDEOS_TABLE; missing {VIDEOS_TABLE}")
if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError("Compatible converted checkpoint is missing: " + str(WEIGHTS_PATH))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for relative in ("per_video", "logs", ".cache/downloads", ".tmp"):
    (OUTPUT_DIR / relative).mkdir(parents=True, exist_ok=True)
'''),
    code(r'''# Input table loading, safe source display, validation, and deterministic selection.
OUTPUT_COLUMNS = ["shot_id", "video_id", "shot_index", "start_ms", "end_ms", "start_frame_idx", "end_frame_idx"]

def read_video_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv": return pd.read_csv(path, dtype={"video_id": "string"})
    if suffix == ".jsonl": return pd.read_json(path, lines=True, dtype={"video_id": "string"})
    if suffix == ".parquet": return pd.read_parquet(path)
    raise ValueError("VIDEOS_TABLE must have .csv, .jsonl, or .parquet suffix")

def safe_source_label(value: str | None) -> str | None:
    if not value: return None
    parsed = urlparse(value)
    return parsed.path if parsed.scheme in {"http", "https"} else Path(value).name

def is_valid_url(value: Any) -> bool:
    if not isinstance(value, str): return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

def local_video_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip(): return None
    path = Path(value.strip())
    return path if path.is_absolute() else INPUT_DIR / path

def validate_and_select_videos(table: pd.DataFrame) -> list[dict[str, Any]]:
    if "video_id" not in table.columns or not ({"video_path", "video_url"} & set(table.columns)):
        raise ValueError("Input requires video_id and at least one of video_path/video_url")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in table.to_dict("records"):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id or len(video_id) > 15: raise ValueError(f"Invalid video_id: {video_id!r}")
        if video_id in seen: raise ValueError(f"Duplicate video_id: {video_id}")
        seen.add(video_id)
        local = local_video_path(row.get("video_path"))
        url = str(row.get("video_url") or "").strip() or None
        if local is not None and local.is_file(): source = {"kind": "local", "path": local}
        elif url and is_valid_url(url): source = {"kind": "url", "url": url}
        else: raise ValueError(f"{video_id}: no existing video_path or valid HTTP(S) video_url")
        suffix = (source.get("path", Path(urlparse(source.get("url", "")).path)).suffix.lower())
        if suffix not in ALLOWED_VIDEO_SUFFIXES: raise ValueError(f"{video_id}: only {sorted(ALLOWED_VIDEO_SUFFIXES)} are allowed")
        records.append({"video_id": video_id, "source": source})
    records.sort(key=lambda item: item["video_id"])
    if VIDEO_IDS is not None:
        wanted = set(VIDEO_IDS); unknown = wanted - {x["video_id"] for x in records}
        if unknown: raise ValueError(f"VIDEO_IDS not found: {sorted(unknown)}")
        records = [x for x in records if x["video_id"] in wanted]
    records = records[START_OFFSET:] if START_OFFSET else records
    if MAX_VIDEOS is not None: records = records[:MAX_VIDEOS]
    print(f"Selected {len(records)} videos; IDs preview: {[x['video_id'] for x in records[:10]]}")
    return records

VIDEO_RECORDS = validate_and_select_videos(read_video_table(VIDEOS_TABLE))
'''),
    markdown("""## TransNetV2 architecture and backend-compatible helpers\n\nCell bên dưới chứa đầy đủ kiến trúc PyTorch vendor từ project (tên layer/shape giữ nguyên checkpoint), rồi thêm decoder FFmpeg, sliding window và logic boundary tương thích backend. Không dùng OpenCV hay nội suy FPS.\n"""),
    code(MODEL_SOURCE + r'''

INPUT_WIDTH, INPUT_HEIGHT = 48, 27
FRAME_BYTES = INPUT_WIDTH * INPUT_HEIGHT * 3
WINDOW_SIZE, WINDOW_HOP, WINDOW_CONTEXT = 100, 50, 25

@dataclass(frozen=True)
class ShotMetadata:
    shot_id: str
    video_id: str
    shot_index: int
    start_ms: int
    end_ms: int
    start_frame_idx: int
    end_frame_idx: int

def parse_frame_rate(raw: str) -> float | None:
    numerator, separator, denominator = raw.partition("/")
    try: fps = float(numerator) / (float(denominator) if separator else 1.0)
    except (ValueError, ZeroDivisionError): return None
    return fps if fps > 0 else None

def probe_fps(video_path: Path) -> float:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "default=noprint_wrappers=1", str(video_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode: raise RuntimeError(f"ffprobe failed for {video_path.name}: {result.stderr[:500]}")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    fps = parse_frame_rate(fields.get("avg_frame_rate", "")) or parse_frame_rate(fields.get("r_frame_rate", ""))
    if fps is None: raise RuntimeError(f"Could not determine positive fps for {video_path.name}")
    return fps

def probe_frame_count(video_path: Path) -> int:
    command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    try: count = int(result.stdout.strip())
    except ValueError: count = 0
    if result.returncode or count < 1: raise RuntimeError(f"Could not count frames for {video_path.name} before decode")
    return count

def decode_frames(video_path: Path, raw_path: Path, mode: str) -> tuple[np.ndarray, Path | None, float]:
    started = time.monotonic()
    command = ["ffmpeg", "-v", "error", "-i", str(video_path), "-vf", "scale=48:27", "-pix_fmt", "rgb24", "-f", "rawvideo"]
    if mode == "memory":
        result = subprocess.run([*command, "pipe:1"], capture_output=True)
        if result.returncode: raise RuntimeError(f"ffmpeg decode failed for {video_path.name}: {result.stderr.decode(errors='replace')[:500]}")
        raw = result.stdout
        if not raw or len(raw) % FRAME_BYTES: raise RuntimeError(f"Invalid raw RGB byte count for {video_path.name}")
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 27, 48, 3), None, time.monotonic()-started
    with raw_path.open("wb") as stream:
        process = subprocess.Popen([*command, "pipe:1"], stdout=stream, stderr=subprocess.PIPE)
        _, stderr = process.communicate()
    byte_count = raw_path.stat().st_size if raw_path.exists() else 0
    if process.returncode or not byte_count or byte_count % FRAME_BYTES:
        raise RuntimeError(f"ffmpeg decode failed for {video_path.name}: {stderr.decode(errors='replace')[:500]}")
    frames = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(byte_count // FRAME_BYTES, 27, 48, 3))
    return frames, raw_path, time.monotonic()-started

def iter_windows(frames: np.ndarray) -> Iterator[np.ndarray]:
    total = len(frames)
    if total < 1: raise ValueError("Cannot predict zero frames")
    trailing_pad = WINDOW_CONTEXT + WINDOW_HOP - (total % WINDOW_HOP if total % WINDOW_HOP else WINDOW_HOP)
    padded = np.concatenate([np.repeat(frames[:1], WINDOW_CONTEXT, axis=0), frames, np.repeat(frames[-1:], trailing_pad, axis=0)])
    for pointer in range(0, len(padded) - WINDOW_SIZE + 1, WINDOW_HOP): yield padded[pointer:pointer + WINDOW_SIZE]

def predictions_to_scene_frames(predictions: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    if predictions.ndim != 1 or not len(predictions): raise ValueError("predictions must be non-empty 1-D")
    states = (predictions > threshold).astype(np.uint8); scenes = []; previous = 0; start = 0; state = 0
    for index, state in enumerate(states):
        state = int(state)
        if previous == 1 and state == 0: start = index
        if previous == 0 and state == 1 and index != 0: scenes.append((start, index))
        previous = state
    if state == 0: scenes.append((start, len(predictions)-1))
    return scenes or [(0, len(predictions)-1)]

def merge_short_shots(raw: list[list[int]], minimum_ms: int) -> list[list[int]]:
    if len(raw) <= 1: return raw
    merged: list[list[int]] = []
    for shot in raw:
        if shot[3] - shot[2] < minimum_ms and merged:
            merged[-1][1], merged[-1][3] = shot[1], shot[3]
        else: merged.append(list(shot))
    if len(merged) > 1 and merged[0][3] - merged[0][2] < minimum_ms:
        first = merged.pop(0); merged[0][0], merged[0][2] = first[0], first[2]
    return merged

def validate_shots(shots: list[ShotMetadata], video_id: str) -> None:
    previous_end_ms = previous_end_frame = -1
    for index, shot in enumerate(shots):
        if shot.shot_index != index or shot.video_id != video_id or len(shot.shot_id) > 15: raise ValueError(f"{video_id}: invalid shot identity / varchar(15) overflow")
        if shot.start_ms < previous_end_ms or shot.end_ms <= shot.start_ms or shot.start_frame_idx <= previous_end_frame or shot.end_frame_idx < shot.start_frame_idx: raise ValueError(f"{video_id}: invalid order/overlap at {shot.shot_id}")
        previous_end_ms, previous_end_frame = shot.end_ms, shot.end_frame_idx

def scene_frames_to_shots(scenes: list[tuple[int, int]], video_id: str, fps: float, minimum_ms: int) -> list[ShotMetadata]:
    if fps <= 0 or not scenes: raise ValueError(f"{video_id}: fps must be positive and scenes non-empty")
    raw: list[list[int]] = []; prior = -1
    for start, end in scenes:
        if start <= prior or end < start: raise ValueError(f"{video_id}: invalid scenes")
        start_ms, end_ms = round(start / fps * 1000), round((end + 1) / fps * 1000)
        raw.append([start, end, start_ms, max(end_ms, start_ms + 1)]); prior = end
    shots = [ShotMetadata(f"{video_id}_S{i:03d}", video_id, i, s_ms, e_ms, start, end) for i, (start, end, s_ms, e_ms) in enumerate(merge_short_shots(raw, minimum_ms))]
    validate_shots(shots, video_id); return shots
'''),
    code(r'''# Model loading and bounded GPU inference. OOM retries the same video from the beginning.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if REQUIRE_GPU and DEVICE.type != "cuda": raise RuntimeError("REQUIRE_GPU=true but CUDA unavailable")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    free, total = torch.cuda.mem_get_info()
    print(f"GPU={torch.cuda.get_device_name(0)} PyTorch={torch.__version__} CUDA={torch.version.cuda} VRAM free/total={free//2**20}/{total//2**20} MiB")
if USE_AMP: print("WARNING: AMP is enabled; A/B compare against FP32 before official use.")

def load_model() -> TransNetV2:
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model = TransNetV2(); model.load_state_dict(state, strict=True); model.eval(); model.to(DEVICE)
    return model

MODEL = load_model()

def predict_frames(model: Any, frames: np.ndarray, requested_batch_size: int) -> tuple[np.ndarray, int, float]:
    batch_size = requested_batch_size
    while True:
        chunks: list[np.ndarray] = []; started = time.monotonic()
        try:
            batch: list[np.ndarray] = []
            with torch.inference_mode():
                for window in iter_windows(frames):
                    batch.append(window)
                    if len(batch) == batch_size:
                        tensor = torch.from_numpy(np.stack(batch)).to(DEVICE, non_blocking=DEVICE.type == "cuda")
                        context = torch.autocast(device_type="cuda", enabled=USE_AMP) if DEVICE.type == "cuda" else nullcontext()
                        with context: logits, _ = model(tensor)
                        chunks.append(torch.sigmoid(logits)[:, 25:75, 0].float().cpu().numpy()); del tensor, logits; batch.clear()
                if batch:
                    tensor = torch.from_numpy(np.stack(batch)).to(DEVICE, non_blocking=DEVICE.type == "cuda")
                    context = torch.autocast(device_type="cuda", enabled=USE_AMP) if DEVICE.type == "cuda" else nullcontext()
                    with context: logits, _ = model(tensor)
                    chunks.append(torch.sigmoid(logits)[:, 25:75, 0].float().cpu().numpy()); del tensor, logits
            return np.concatenate(chunks).reshape(-1)[:len(frames)], batch_size, time.monotonic()-started
        except torch.cuda.OutOfMemoryError as error:
            del chunks; gc.collect()
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
            if batch_size == 1: raise RuntimeError("CUDA OOM even at WINDOW_BATCH_SIZE=1") from error
            batch_size = max(1, batch_size // 2); print(f"CUDA OOM; retrying same video with window batch size {batch_size}")
'''),
    code(r'''# Resume fingerprints, streaming download, atomic writes, per-video runner, and aggregate exports.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

WEIGHTS_SHA256 = sha256_file(WEIGHTS_PATH)
CONFIG_FINGERPRINT = hashlib.sha256(json.dumps({"logic": NOTEBOOK_LOGIC_VERSION, "threshold": THRESHOLD, "min_ms": MIN_SHOT_DURATION_MS, "weights": WEIGHTS_SHA256, "amp": USE_AMP}, sort_keys=True).encode()).hexdigest()

def safe_url_fingerprint(url: str) -> str:
    parsed = urlparse(url); return hashlib.sha256(urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", "")).encode()).hexdigest()

def source_fingerprint(source: dict[str, Any]) -> str:
    if source["kind"] == "local":
        stat = source["path"].stat(); return hashlib.sha256(f"local:{source['path']}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    return safe_url_fingerprint(source["url"])

def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(temporary, path)

def append_error(video_id: str, stage: str, error: BaseException) -> None:
    message = str(error).replace("\n", " ")[:500]
    with (OUTPUT_DIR / "logs/errors.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps({"video_id": video_id, "stage": stage, "exception_type": type(error).__name__, "message": message}) + "\n")

def load_checkpoint() -> dict[str, Any]:
    path = OUTPUT_DIR / "checkpoint.json"
    try: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"done": {}}
    except json.JSONDecodeError: return {"done": {}}

def resolve_video(record: dict[str, Any]) -> Path:
    source, video_id = record["source"], record["video_id"]
    if source["kind"] == "local": return source["path"]
    target = OUTPUT_DIR / ".cache/downloads" / f"{video_id}.mp4"; part = target.with_suffix(".mp4.part"); cache_info = target.with_suffix(".json")
    # URL is deliberately never logged; cache validity is tied to its redacted fingerprint.
    fingerprint = source_fingerprint(source)
    try:
        if target.is_file() and json.loads(cache_info.read_text(encoding="utf-8")).get("source_fingerprint") == fingerprint: return target
    except (OSError, json.JSONDecodeError): pass
    for attempt in range(3):
        try:
            with requests.get(source["url"], stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                with part.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk: stream.write(chunk)
            os.replace(part, target); atomic_json(cache_info, {"source_fingerprint": fingerprint}); return target
        except requests.RequestException:
            part.unlink(missing_ok=True)
            if attempt == 2: raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")

def valid_per_video(path: Path, fingerprint: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8")); shots = [ShotMetadata(**row) for row in payload["shots"]]
        validate_shots(shots, payload["video_id"])
        return payload.get("source_fingerprint") == fingerprint and payload.get("config_fingerprint") == CONFIG_FINGERPRINT
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError): return False

def run_one(record: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    video_id, source = record["video_id"], record["source"]; fingerprint = source_fingerprint(source); per_video = OUTPUT_DIR / "per_video" / f"{video_id}.json"
    if not FORCE_REPROCESS and checkpoint["done"].get(video_id) == {"source": fingerprint, "config": CONFIG_FINGERPRINT} and valid_per_video(per_video, fingerprint): return "skipped", {"video_id": video_id}
    temporary_raw: Path | None = None; frames: np.ndarray | None = None
    try:
        video_path = resolve_video(record); fps = probe_fps(video_path)
        # Raw RGB estimate is exact once ffprobe has counted decoded frames.
        estimated = probe_frame_count(video_path) * FRAME_BYTES
        mode = DECODE_MODE if DECODE_MODE != "auto" else ("memory" if estimated <= MAX_DECODE_RAM_MB * 1024**2 else "memmap")
        temporary_raw = OUTPUT_DIR / ".tmp" / f"{video_id}.rgb"
        frames, raw_path, decode_seconds = decode_frames(video_path, temporary_raw, mode)
        predictions, effective_batch, inference_seconds = predict_frames(MODEL, frames, WINDOW_BATCH_SIZE)
        shots = scene_frames_to_shots(predictions_to_scene_frames(predictions, THRESHOLD), video_id, fps, MIN_SHOT_DURATION_MS)
        payload = {"video_id": video_id, "source": {"kind": source["kind"], "basename": video_path.name}, "source_fingerprint": fingerprint, "config_fingerprint": CONFIG_FINGERPRINT, "fps": fps, "frame_count": len(frames), "elapsed_seconds": round(decode_seconds + inference_seconds, 3), "decode_seconds": round(decode_seconds, 3), "inference_seconds": round(inference_seconds, 3), "effective_batch_size": effective_batch, "detector": {"threshold": THRESHOLD, "min_shot_duration_ms": MIN_SHOT_DURATION_MS, "use_amp": USE_AMP}, "shots": [asdict(shot) for shot in shots]}
        atomic_json(per_video, payload); checkpoint["done"][video_id] = {"source": fingerprint, "config": CONFIG_FINGERPRINT}; atomic_json(OUTPUT_DIR / "checkpoint.json", checkpoint)
        return "processed", payload
    finally:
        if isinstance(frames, np.memmap): frames._mmap.close()
        del frames
        if temporary_raw is not None: temporary_raw.unlink(missing_ok=True)
        gc.collect()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

def sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"

def rebuild_exports() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((OUTPUT_DIR / "per_video").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); rows.extend(payload["shots"])
    rows.sort(key=lambda row: (row["video_id"], row["shot_index"]))
    with (OUTPUT_DIR / "shots.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS); writer.writeheader(); writer.writerows(rows)
    with (OUTPUT_DIR / "shots.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps({key: row[key] for key in OUTPUT_COLUMNS}, ensure_ascii=False) + "\n")
    columns = ", ".join(OUTPUT_COLUMNS); updates = ",\n  ".join(f"{key} = EXCLUDED.{key}" for key in OUTPUT_COLUMNS[1:])
    statements = ["BEGIN;"]
    for row in rows:
        values = ", ".join(sql_literal(row[key]) if key in {"shot_id", "video_id"} else str(int(row[key])) for key in OUTPUT_COLUMNS)
        statements.append(f"INSERT INTO shot (\n  {columns}\n) VALUES ({values})\nON CONFLICT (shot_id) DO UPDATE SET\n  {updates};")
    statements.append("COMMIT;")
    (OUTPUT_DIR / "insert_shots.sql").write_text("\n\n".join(statements) + "\n", encoding="utf-8")
    return rows

def run_batch(records: list[dict[str, Any]]) -> pd.DataFrame:
    checkpoint, summary, started = load_checkpoint(), [], time.monotonic()
    for record in records:
        try:
            status, detail = run_one(record, checkpoint); detail["status"] = status; summary.append(detail)
        except Exception as error:
            append_error(record["video_id"], "per_video", error); summary.append({"video_id": record["video_id"], "status": "failed", "error": type(error).__name__})
            if not CONTINUE_ON_ERROR: rebuild_exports(); raise RuntimeError(f"Failed video {record['video_id']}; see logs/errors.jsonl") from error
    rows = rebuild_exports(); manifest = {"run_id": RUN_ID, "logic_version": NOTEBOOK_LOGIC_VERSION, "config_fingerprint": CONFIG_FINGERPRINT, "weights_sha256": WEIGHTS_SHA256, "use_amp": USE_AMP, "wall_seconds": round(time.monotonic()-started, 3), "summary": summary, "rows": len(rows)}; atomic_json(OUTPUT_DIR / "run_manifest.json", manifest)
    result = pd.DataFrame(summary); print(result["status"].value_counts().to_dict(), "wall seconds", manifest["wall_seconds"]); return result

atomic_json(OUTPUT_DIR / "config.json", {"logic_version": NOTEBOOK_LOGIC_VERSION, "input_table": VIDEOS_TABLE.name, "weights_sha256": WEIGHTS_SHA256, "threshold": THRESHOLD, "min_shot_duration_ms": MIN_SHOT_DURATION_MS, "window_batch_size": WINDOW_BATCH_SIZE, "decode_mode": DECODE_MODE, "require_gpu": REQUIRE_GPU, "use_amp": USE_AMP})
'''),
    code(r'''# Mock/smoke tests — run this cell before enabling the real pipeline. No real video/weights required.
import tempfile

def run_mock_tests() -> None:
    assert predictions_to_scene_frames(np.zeros(10, np.float32)) == [(0, 9)]
    assert predictions_to_scene_frames(np.array([0, 0, 0, 1, 1, 1, 1, 0, 0], np.float32)) == [(0, 3), (7, 8)]
    assert predictions_to_scene_frames(np.array([0, .5, 0], np.float32)) == [(0, 2)]
    assert predictions_to_scene_frames(np.ones(10, np.float32)) == [(0, 9)]
    assert len(scene_frames_to_shots([(0, 4), (5, 49)], "L21_V001", 25., 500)) == 1
    assert len(scene_frames_to_shots([(0, 199), (200, 204)], "L21_V001", 25., 500)) == 1
    expected = scene_frames_to_shots([(0, 24)], "L21_V001", 25., 500)[0]
    assert (expected.start_ms, expected.end_ms) == (0, 1000)
    for count in (1, 49, 50, 51, 100, 137, 250):
        frames = np.zeros((count, 27, 48, 3), dtype=np.uint8); windows = list(iter_windows(frames))
        assert all(window.shape == (100, 27, 48, 3) for window in windows) and len(windows) * 50 >= count
    # End-to-end export shape with a mocked decoder/model prediction is checked without touching GPU/network.
    shots = scene_frames_to_shots(predictions_to_scene_frames(np.array([0]*30 + [1,1] + [0]*40, np.float32)), "L21_V001", 25., 500)
    assert [shot.shot_id for shot in shots] == [f"L21_V001_S{i:03d}" for i in range(len(shots))]
    validate_shots(shots, "L21_V001")
    # OOM retry: model fails once at batch 4 then succeeds at batch 2; batch-1 failure is explicit.
    class MockOOMModel:
        def __init__(self, always=False): self.always = always
        def __call__(self, tensor):
            if self.always or tensor.shape[0] > 2: raise torch.cuda.OutOfMemoryError("mock")
            return torch.zeros((tensor.shape[0], 100, 1)), {}
    old_device = globals()["DEVICE"]
    try:
        globals()["DEVICE"] = torch.device("cpu")
        pred, effective, _ = predict_frames(MockOOMModel(), np.zeros((137,27,48,3), np.uint8), 4); assert len(pred) == 137 and effective == 2
        try: predict_frames(MockOOMModel(always=True), np.zeros((1,27,48,3), np.uint8), 1)
        except RuntimeError as error: assert "BATCH_SIZE=1" in str(error)
        else: raise AssertionError("OOM at batch 1 did not fail clearly")
    finally: globals()["DEVICE"] = old_device
    # Resume: a valid matching per-video payload skips before resolve/decode/inference;
    # changing source/config forces the runner past that guard (mocked resolver observes it).
    with tempfile.TemporaryDirectory() as temp_dir:
        temporary_output = Path(temp_dir); (temporary_output / "per_video").mkdir(); (temporary_output / "logs").mkdir(); (temporary_output / ".cache/downloads").mkdir(parents=True); (temporary_output / ".tmp").mkdir()
        source_file = temporary_output / "source.mp4"; source_file.write_bytes(b"mock-source")
        record = {"video_id": "L21_V001", "source": {"kind": "local", "path": source_file}}
        source_fp = source_fingerprint(record["source"])
        payload = {"video_id": "L21_V001", "source_fingerprint": source_fp, "config_fingerprint": CONFIG_FINGERPRINT, "shots": [asdict(expected)]}
        per_video = temporary_output / "per_video/L21_V001.json"; atomic_json(per_video, payload)
        checkpoint = {"done": {"L21_V001": {"source": source_fp, "config": CONFIG_FINGERPRINT}}}
        old_output, old_force, old_resolve = globals()["OUTPUT_DIR"], globals()["FORCE_REPROCESS"], globals()["resolve_video"]
        calls = []
        try:
            globals()["OUTPUT_DIR"], globals()["FORCE_REPROCESS"] = temporary_output, False
            status, _ = run_one(record, checkpoint); assert status == "skipped"
            assert len(rebuild_exports()) == 1  # rerun rebuilds, never appends duplicate rows.
            exported_csv = pd.read_csv(temporary_output / "shots.csv"); exported_jsonl = [json.loads(line) for line in (temporary_output / "shots.jsonl").read_text().splitlines()]
            exported_sql = (temporary_output / "insert_shots.sql").read_text(encoding="utf-8")
            assert list(exported_csv.columns) == OUTPUT_COLUMNS and len(exported_jsonl) == 1
            assert exported_jsonl[0] == {key: getattr(expected, key) for key in OUTPUT_COLUMNS}
            assert "BEGIN;" in exported_sql and "COMMIT;" in exported_sql and "ON CONFLICT (shot_id)" in exported_sql
            source_file.write_bytes(b"changed-source")
            globals()["resolve_video"] = lambda _record: (calls.append("resolve"), (_ for _ in ()).throw(RuntimeError("reprocess observed")))[1]
            try: run_one(record, checkpoint)
            except RuntimeError as error: assert str(error) == "reprocess observed" and calls
            else: raise AssertionError("Changed source incorrectly skipped")
            payload["config_fingerprint"] = "old"; atomic_json(per_video, payload); source_file.write_bytes(b"mock-source")
            checkpoint["done"]["L21_V001"] = {"source": source_fingerprint(record["source"]), "config": CONFIG_FINGERPRINT}
            calls.clear()
            try: run_one(record, checkpoint)
            except RuntimeError as error: assert str(error) == "reprocess observed" and calls
            else: raise AssertionError("Changed config incorrectly skipped")
        finally: globals()["OUTPUT_DIR"], globals()["FORCE_REPROCESS"], globals()["resolve_video"] = old_output, old_force, old_resolve
    print("Mock tests passed: boundary, merge/time, sliding windows, output IDs/order, OOM retry, and resume/idempotency guards.")

run_mock_tests()
'''),
    code(r'''# Real run is opt-in. Set RUN_REAL_PIPELINE=True in the configuration cell, then run this cell.
if RUN_REAL_PIPELINE:
    batch_summary = run_batch(VIDEO_RECORDS)
    display(batch_summary)
    preview = pd.read_csv(OUTPUT_DIR / "shots.csv").head(10)
    display(preview)
    print("Download this directory from Kaggle Output:", OUTPUT_DIR)
    print("Import insert_shots.sql only after DBA reviews existing referenced shot rows.")
else:
    print("Real pipeline is disabled. Set RUN_REAL_PIPELINE=True in the configuration cell after mock tests pass.")
'''),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
