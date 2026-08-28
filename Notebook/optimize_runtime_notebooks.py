"""Apply runtime-throughput and packaging improvements to generated notebooks.

The notebooks remain standalone.  This migrator deliberately does not change their
selected model/artifact contract or environment-specific root paths.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def find_cell(notebook: dict, marker: str) -> dict:
    matches = [cell for cell in notebook["cells"] if marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Expected source fragment was not found: {old[:100]!r}")
    return text.replace(old, new, 1)


def add_zip_cell(
    notebook: dict,
    *,
    variable: str,
    prefix: str,
    after_marker: str | None = None,
) -> None:
    marker = f"# Package validated {prefix} artifacts"
    existing = next((cell for cell in notebook["cells"] if marker in source(cell)), None)
    zip_source = f'''{marker} into one downloadable ZIP.
zip_path = Path(shutil.make_archive(str({variable}), "zip", root_dir={variable}))
required = [path for path in {variable}.rglob("*") if path.suffix in {{".sql", ".faiss"}}]
if not required:
    raise RuntimeError("ZIP packaging refused: validated SQL/FAISS files were not found")
print(f"Download ZIP: {{zip_path}} ({{zip_path.stat().st_size / 1024**2:.1f}} MiB)")
'''
    zip_cell = existing or {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": zip_source.splitlines(keepends=True),
        }
    set_source(zip_cell, zip_source)
    if existing is not None:
        notebook["cells"].remove(existing)
    if after_marker is None:
        notebook["cells"].append(zip_cell)
        return
    target_index = next(
        index for index, cell in enumerate(notebook["cells"])
        if after_marker in source(cell)
    )
    notebook["cells"].insert(target_index + 1, zip_cell)


def patch_frame() -> None:
    path = ROOT / "FrameEmbedding/frame_embedding_siglip_kaggle.ipynb"
    notebook = load(path)

    config = find_cell(notebook, "IMAGE_BATCH_SIZE = 128")
    value = source(config)
    value = replace_once(
        value,
        "IMAGE_BATCH_SIZE = 128\nMIN_BATCH_SIZE = 8\nNUM_WORKERS = min(4, os.cpu_count() or 1)",
        "IMAGE_BATCH_SIZE = 128\nMIN_BATCH_SIZE = 8\nAUTO_TUNE_BATCH = True\nAUTO_TUNE_BATCH_CANDIDATES = (64, 128, 192, 256, 384, 512)\nNUM_WORKERS = min(8, os.cpu_count() or 1)\nPREFETCH_FACTOR = 4",
    )
    set_source(config, value)

    preflight = find_cell(notebook, "# Imports and runtime setup")
    value = source(preflight)
    value = value.replace(
        '''if not KEYFRAME_FILE.is_file():
    KEYFRAME_FILE = INPUT_DIR / "keyframes.csv"
if not KEYFRAME_FILE.is_file():
    raise FileNotFoundError(f"Neither keyframe.csv nor keyframes.csv exists under {INPUT_DIR}")''',
        '''if not KEYFRAME_FILE.is_file():
    KEYFRAME_FILE = INPUT_DIR / "keyframes.csv"
if not KEYFRAME_FILE.is_file():
    print("Configured keyframe CSV is absent; input discovery will search mounted datasets.")''',
        1,
    )
    set_source(preflight, value)

    input_cell = find_cell(notebook, "# Input validation and deterministic path resolution")
    value = source(input_cell)
    discovery = '''def discover_keyframe_file() -> Path:
    if KEYFRAME_FILE.is_file():
        return KEYFRAME_FILE
    roots = [INPUT_DIR, Path("/kaggle/input")]
    candidates = sorted({candidate for root in roots if root.is_dir() for name in ("keyframe.csv", "keyframes.csv") for candidate in root.rglob(name)})
    if not candidates:
        raise FileNotFoundError("Drop a dataset containing keyframe.csv/keyframes.csv into the notebook input")
    if len(candidates) > 1:
        print("Multiple keyframe CSV files found; using", candidates[0])
    return candidates[0]

KEYFRAME_FILE = discover_keyframe_file()
if not FRAME_ROOT.is_dir():
    FRAME_ROOT = KEYFRAME_FILE.parent / "keyframes"

'''
    if "def discover_keyframe_file" not in value:
        value = value.replace("REQUIRED_COLUMNS =", discovery + "REQUIRED_COLUMNS =", 1)
    value = value.replace(
        '''    source = Path(raw_path)
    candidates = [source] if source.is_absolute() else []
    candidates += [FRAME_ROOT / source, FRAME_ROOT / source.name]
    return next((candidate for candidate in candidates if candidate.is_file()), None)''',
        '''    source = Path(raw_path)
    candidates = [source] if source.is_absolute() else []
    candidates += [KEYFRAME_FILE.parent / source, INPUT_DIR / source, FRAME_ROOT / source, FRAME_ROOT / source.name]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)''',
        1,
    )
    set_source(input_cell, value)

    model_cell = find_cell(notebook, "def encode_images(images: list[Image.Image]")
    value = source(model_cell)
    start = value.index("def encode_images(images: list[Image.Image]")
    replacement = '''BATCH_TUNING_TRIALS: list[dict[str, float | int]] = []

def encode_exact(images: list[Image.Image]) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {
        name: value.pin_memory().to(DEVICE, non_blocking=True)
        for name, value in inputs.items()
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=COMPUTE_DTYPE):
        image_output = model.get_image_features(**inputs)
    output = image_output.pooler_output
    if output is None or not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise RuntimeError("SigLIP2 did not return a 2D image pooler_output tensor")
    output = torch.nn.functional.normalize(output.float(), p=2, dim=1)
    result = np.ascontiguousarray(output.cpu().numpy(), dtype=np.float32)
    if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("model returned non-finite or non-normalized embeddings")
    return result

def autotune_image_batch(images: list[Image.Image]) -> int:
    if not AUTO_TUNE_BATCH or not images:
        return min(IMAGE_BATCH_SIZE, max(MIN_BATCH_SIZE, len(images)))
    best_size, best_rate = MIN_BATCH_SIZE, 0.0
    BATCH_TUNING_TRIALS.clear()
    for candidate in AUTO_TUNE_BATCH_CANDIDATES:
        probe = images[:min(candidate, len(images))]
        if len(probe) < MIN_BATCH_SIZE:
            continue
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            encode_exact(probe)
            torch.cuda.synchronize(); elapsed = max(time.perf_counter() - started, 1e-6)
            rate = len(probe) / elapsed
            BATCH_TUNING_TRIALS.append({"batch_size": candidate, "probe_images": len(probe), "images_per_second": rate, "peak_vram_bytes": int(torch.cuda.max_memory_allocated())})
            if rate >= best_rate * 0.98:
                best_size, best_rate = candidate, rate
            if len(probe) < candidate:
                break
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not is_oom(exc):
                raise
            torch.cuda.empty_cache()
            break
    print("Batch autotune:", BATCH_TUNING_TRIALS, "selected=", best_size)
    return max(MIN_BATCH_SIZE, best_size)

def encode_images(images: list[Image.Image], initial_size: int) -> tuple[np.ndarray, int]:
    vectors, cursor, size = [], 0, initial_size
    while cursor < len(images):
        batch = images[cursor:cursor + size]
        try:
            vectors.append(encode_exact(batch)); cursor += len(batch)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not is_oom(exc) or size <= MIN_BATCH_SIZE: raise
            size = max(MIN_BATCH_SIZE, size // 2)
            torch.cuda.empty_cache()
    return np.ascontiguousarray(np.concatenate(vectors)), size
'''
    if "BATCH_TUNING_TRIALS" not in value:
        set_source(model_cell, value[:start] + replacement)

    run_cell = find_cell(notebook, "def run_embedding() -> Path:")
    value = source(run_cell)
    value = replace_once(
        value,
        "loader_args = dict(batch_size=IMAGE_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate)\n    if NUM_WORKERS: loader_args.update(persistent_workers=True, prefetch_factor=2)",
        "loader_batch_size = max(AUTO_TUNE_BATCH_CANDIDATES) if AUTO_TUNE_BATCH else IMAGE_BATCH_SIZE\n    loader_args = dict(batch_size=loader_batch_size, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate)\n    if NUM_WORKERS: loader_args.update(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)",
    )
    value = replace_once(
        value,
        "index, dimension, success, pending_vectors, pending_rows, final_batch = None, None, 0, [], [], IMAGE_BATCH_SIZE",
        "index, dimension, success, pending_vectors, pending_rows, final_batch = None, None, 0, [], [], None",
    )
    value = replace_once(
        value,
        "if not good: continue\n        vectors, final_batch = encode_images([image for _, image in good], final_batch)",
        "if not good: continue\n        images = [image for _, image in good]\n        if final_batch is None: final_batch = autotune_image_batch(images)\n        vectors, final_batch = encode_images(images, final_batch)",
    )
    value = replace_once(
        value,
        '"final_batch_size": final_batch, "images_per_second"',
        '"final_batch_size": final_batch, "batch_tuning_trials": BATCH_TUNING_TRIALS, "images_per_second"',
    )
    set_source(run_cell, value)

    add_zip_cell(
        notebook,
        variable="run_dir",
        prefix="Frame",
        after_marker="run_dir = run_embedding()",
    )
    save(path, notebook)


def patch_clip() -> None:
    path = ROOT / "ClipEmbedding/clip_embedding_siglip_kaggle.ipynb"
    notebook = load(path)

    config = find_cell(notebook, "MAX_CLIPS_PER_DECODE_UNIT = 16")
    value = source(config)
    value = replace_once(
        value,
        "IMAGE_BATCH_SIZE = 256\nMIN_BATCH_SIZE = 8\nDECODE_WORKERS = 2\nDECODE_TOLERANCE_MS = 500",
        "IMAGE_BATCH_SIZE = 256\nMIN_BATCH_SIZE = 8\nAUTO_TUNE_BATCH = True\nAUTO_TUNE_BATCH_CANDIDATES = (64, 128, 192, 256, 384, 512)\nDECODE_WORKERS = 1  # one bounded producer overlaps grouped PyAV decode with GPU inference\nDECODE_TOLERANCE_MS = 500\nDECODE_GROUP_GAP_MS = 2_000",
    )
    set_source(config, value)

    validation = find_cell(notebook, "def load_inputs()")
    value = source(validation)
    discovery = '''def discover_clip_input() -> None:
    global INPUT_DIR, VIDEOS_FILE, SHOT_FILE, CLIP_FILE, VIDEO_CACHE_DIR
    preferred = [CLIP_FILE] if CLIP_FILE.is_file() else []
    roots = [INPUT_DIR, Path("/kaggle/input")]
    candidates = preferred + sorted({candidate for root in roots if root.is_dir() for candidate in root.rglob("clipwindow.csv")})
    for clip_path in candidates:
        root = clip_path.parent
        videos = root / "videos.csv"
        shot = root / "shot.csv" if (root / "shot.csv").is_file() else root / "shots.csv"
        if videos.is_file() and shot.is_file():
            INPUT_DIR, VIDEOS_FILE, SHOT_FILE, CLIP_FILE = root, videos, shot, clip_path
            VIDEO_CACHE_DIR = OUTPUT_DIR / "video_cache"
            print("Using Clip input:", root)
            return
    raise FileNotFoundError("Drop videos.csv, shot.csv/shots.csv and clipwindow.csv into one input directory")

discover_clip_input()

'''
    if "def discover_clip_input" not in value:
        value = discovery + value
    set_source(validation, value)

    decoder = find_cell(notebook, "def decode_requests(path: Path")
    value = source(decoder)
    value = value.replace(
        '''    local=row.get("video_path","").strip()
    if local and Path(local).is_file(): return Path(local), False''',
        '''    local=row.get("video_path","").strip()
    if local:
        source=Path(local); candidates=[source] if source.is_absolute() else [VIDEOS_FILE.parent/source, INPUT_DIR/source]
        resolved=next((candidate.resolve() for candidate in candidates if candidate.is_file()),None)
        if resolved is not None: return resolved, False''',
        1,
    )
    start = value.index("def decode_requests(path: Path")
    replacement = '''def group_timestamps(requested: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for timestamp in sorted(set(requested)):
        if not groups or timestamp - groups[-1][-1] > DECODE_GROUP_GAP_MS:
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)
    return groups

def decode_requests(path: Path, requested: list[int]) -> dict[int, tuple[int, Image.Image] | None]:
    """Seek once per nearby timestamp group instead of once per requested frame."""
    output: dict[int, tuple[int, Image.Image] | None] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; time_base = float(stream.time_base)
        for group in group_timestamps(requested):
            start_ms = max(0, group[0] - DECODE_TOLERANCE_MS)
            end_ms = group[-1] + DECODE_TOLERANCE_MS
            container.seek(max(0, int((start_ms / 1000) / time_base)), stream=stream, any_frame=False, backward=True)
            candidates: list[tuple[int, Image.Image]] = []
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                actual_ms = round(float(frame.pts * stream.time_base) * 1000)
                if actual_ms > end_ms:
                    break
                if actual_ms >= start_ms:
                    candidates.append((actual_ms, frame.to_image().convert("RGB")))
            for timestamp in group:
                output[timestamp] = min(candidates, key=lambda item: abs(item[0] - timestamp)) if candidates else None
    return output

def decode_unit(path: Path, unit: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[int, tuple[int, Image.Image] | None], float]:
    requested = {clip["clip_id"]: requested_times(clip) for clip in unit}
    flat = [timestamp for timestamps in requested.values() for timestamp in timestamps]
    started = time.perf_counter(); decoded = decode_requests(path, flat)
    return unit, requested, decoded, time.perf_counter() - started
'''
    if "def group_timestamps" not in value:
        set_source(decoder, value[:start] + replacement)
    else:
        value = value.replace(
            "timestamp - groups[-1][-1] > DECODE_GROUP_GAP_MS",
            "timestamp - groups[-1][0] > DECODE_GROUP_GAP_MS",
            1,
        )
        set_source(decoder, value)

    writer_cell = find_cell(notebook, "def config_hash() -> str:")
    value = source(writer_cell)
    value = value.replace(
        '"FRAMES_PER_CLIP","DECODE_TOLERANCE_MS","INDEX_VERSION","ROWS_PER_SHARD"',
        '"FRAMES_PER_CLIP","DECODE_TOLERANCE_MS","DECODE_GROUP_GAP_MS","INDEX_VERSION","ROWS_PER_SHARD"',
        1,
    )
    set_source(writer_cell, value)

    model_cell = find_cell(notebook, "def image_features(images: list[Image.Image]")
    value = source(model_cell)
    start = value.index("def image_features(images: list[Image.Image]")
    replacement = '''BATCH_TUNING_TRIALS: list[dict[str, float | int]] = []

def image_features_exact(images: list[Image.Image]) -> np.ndarray:
    batch = processor(images=images, return_tensors="pt")
    pixels = batch["pixel_values"].pin_memory().to(DEVICE, non_blocking=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=AMP_DTYPE):
        image_output = model.get_image_features(pixel_values=pixels)
    features = image_output.pooler_output
    if features is None or not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise RuntimeError("SigLIP2 did not return a 2D image pooler_output tensor")
    features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
    return np.ascontiguousarray(features.cpu().numpy(), dtype=np.float32)

def autotune_image_batch(images: list[Image.Image]) -> int:
    if not AUTO_TUNE_BATCH or not images:
        return IMAGE_BATCH_SIZE
    best_size, best_rate = MIN_BATCH_SIZE, 0.0; BATCH_TUNING_TRIALS.clear()
    for candidate in AUTO_TUNE_BATCH_CANDIDATES:
        probe = images[:min(candidate, len(images))]
        if len(probe) < MIN_BATCH_SIZE: continue
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            image_features_exact(probe)
            torch.cuda.synchronize(); elapsed = max(time.perf_counter() - started, 1e-6); rate = len(probe) / elapsed
            BATCH_TUNING_TRIALS.append({"batch_size": candidate, "probe_images": len(probe), "images_per_second": rate, "peak_vram_bytes": int(torch.cuda.max_memory_allocated())})
            if rate >= best_rate * 0.98: best_size, best_rate = candidate, rate
            if len(probe) < candidate: break
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower(): raise
            torch.cuda.empty_cache(); break
    print("Batch autotune:", BATCH_TUNING_TRIALS, "selected=", best_size)
    return max(MIN_BATCH_SIZE, best_size)

def image_features(images: list[Image.Image], batch_size: int) -> tuple[np.ndarray, int]:
    chunks=[]; pos=0; current_size=batch_size
    while pos < len(images):
        current=min(current_size,len(images)-pos)
        try:
            chunks.append(image_features_exact(images[pos:pos+current])); pos += current
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() or current <= MIN_BATCH_SIZE: raise
            current_size=max(MIN_BATCH_SIZE,current//2); torch.cuda.empty_cache()
    return np.ascontiguousarray(np.concatenate(chunks)), current_size
'''
    if "BATCH_TUNING_TRIALS" not in value:
        set_source(model_cell, value[:start] + replacement)

    run_marker = "# Main run. A bounded producer" if any("# Main run. A bounded producer" in source(cell) for cell in notebook["cells"]) else "# Main run. Decode units are bounded"
    run_cell = find_cell(notebook, run_marker)
    new_run = r'''# Main run. A bounded producer decodes the next unit while the GPU encodes the current unit.
def append_failure(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def run() -> Path:
    clips, videos = load_inputs(); selected = sorted({item["video_id"] for item in clips})[VIDEO_START:VIDEO_END]
    run_id = RUN_ID or time.strftime("run_%Y%m%d_%H%M%S"); root = OUTPUT_DIR / run_id; root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint.json"
    state = {"input_hash": input_hash(), "config_hash": config_hash(), "completed_video_ids": [], "next_faiss_id": 1, "shards": [], "metrics": {"requested_n": 0, "decoded_n": 0, "success": 0, "failed": 0, "decode_s": 0.0, "encode_s": 0.0, "peak": 0}}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if (state["input_hash"] != input_hash() or state["config_hash"] != config_hash()) and not ALLOW_CHECKPOINT_OVERRIDE:
            raise RuntimeError("checkpoint input/config hash mismatch; set explicit override only after review")
    metrics = state.setdefault("metrics", {}); failures = root / "failures.jsonl"
    requested_n=int(metrics.get("requested_n",0)); decoded_n=int(metrics.get("decoded_n",0)); success=int(metrics.get("success",0)); failed=int(metrics.get("failed",0)); decode_s=float(metrics.get("decode_s",0)); encode_s=float(metrics.get("encode_s",0)); peak=int(metrics.get("peak",0))
    all_vec: list[np.ndarray] = []; all_meta: list[dict[str, Any]] = []; shard=len(state["shards"]); final_batch: int | None = None
    for video_id in selected:
        if video_id in state["completed_video_ids"]: continue
        group=[clip for clip in clips if clip["video_id"] == video_id]; video=None; cached=False
        try:
            video,cached=resolve_video(videos[video_id],video_id)
            units=[group[offset:offset+MAX_CLIPS_PER_DECODE_UNIT] for offset in range(0,len(group),MAX_CLIPS_PER_DECODE_UNIT)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                future = pool.submit(decode_unit, video, units[0]) if units else None
                for unit_index in range(len(units)):
                    assert future is not None
                    unit, req, decoded, unit_decode_s = future.result(); decode_s += unit_decode_s
                    future = pool.submit(decode_unit, video, units[unit_index+1]) if unit_index+1 < len(units) else None
                    flat=[timestamp for timestamps in req.values() for timestamp in timestamps]; requested_n += len(flat); decoded_n += sum(item is not None for item in decoded.values())
                    valid={timestamp:item for timestamp,item in decoded.items() if item is not None}; images=[item[1] for item in valid.values()]; features={}
                    if images:
                        if final_batch is None: final_batch = autotune_image_batch(images)
                        started=time.perf_counter(); matrix,final_batch=image_features(images,final_batch); encode_s += time.perf_counter()-started; features=dict(zip(valid,matrix))
                    for clip in unit:
                        samples=[]; actual=[]
                        for timestamp in req[clip["clip_id"]]:
                            item=decoded.get(timestamp)
                            if item and clip["start_ms"] <= item[0] < clip["end_ms"] and abs(item[0]-timestamp) <= DECODE_TOLERANCE_MS:
                                samples.append(features[timestamp]); actual.append(item[0])
                        if not samples:
                            append_failure(failures,{"stage":"decode_failed","clip_id":clip["clip_id"],"video_id":video_id}); failed += 1; continue
                        vector=np.mean(np.stack(samples),axis=0).astype(np.float32); vector /= np.linalg.norm(vector)
                        all_vec.append(vector); all_meta.append({"faiss_id":state["next_faiss_id"],"clip_id":clip["clip_id"],"shot_id":clip["shot_id"],"video_id":video_id,"start_ms":clip["start_ms"],"end_ms":clip["end_ms"],"sampled_timestamps_ms":req[clip["clip_id"]],"actual_timestamps_ms":actual,"valid_sample_count":len(samples),"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"dimension":len(vector),"normalized":True,"vector_shard":shard,"vector_row":len(all_vec)-1}); state["next_faiss_id"] += 1; success += 1
                    if len(all_vec) >= ROWS_PER_SHARD:
                        write_shard(root,shard,all_vec,all_meta); state["shards"].append(shard); shard += 1; all_vec=[]; all_meta=[]
                    del decoded,valid,images,features; gc.collect(); peak=max(peak,torch.cuda.max_memory_allocated())
            if all_vec:
                write_shard(root,shard,all_vec,all_meta); state["shards"].append(shard); shard += 1; all_vec=[]; all_meta=[]
            state["completed_video_ids"].append(video_id)
            state["metrics"]={"requested_n":requested_n,"decoded_n":decoded_n,"success":success,"failed":failed,"decode_s":decode_s,"encode_s":encode_s,"peak":peak,"final_batch":final_batch,"batch_tuning_trials":BATCH_TUNING_TRIALS}; atomic_json(checkpoint,state)
            if cached: video.unlink(missing_ok=True)
        except Exception as exc:
            append_failure(failures,{"stage":"video","video_id":video_id,"error":str(exc),"traceback":traceback.format_exc()}); failed += len(group)
            state["metrics"]={"requested_n":requested_n,"decoded_n":decoded_n,"success":success,"failed":failed,"decode_s":decode_s,"encode_s":encode_s,"peak":peak,"final_batch":final_batch,"batch_tuning_trials":BATCH_TUNING_TRIALS}; atomic_json(checkpoint,state)
            if cached and video is not None: video.unlink(missing_ok=True)
    if final_batch is None: final_batch = int(state.get("metrics",{}).get("final_batch") or IMAGE_BATCH_SIZE)
    finalize(root,requested_n,decoded_n,success,failed,decode_s,encode_s,peak,final_batch); return root
'''
    set_source(run_cell, new_run)

    add_zip_cell(
        notebook,
        variable="artifact_dir",
        prefix="Clip",
        after_marker="artifact_dir = run()",
    )
    save(path, notebook)


def patch_ocr() -> None:
    path = ROOT / "OCR/ocr_kaggle.ipynb"
    notebook = load(path)

    config = find_cell(notebook, "DETECTION_BATCH_SIZE = 64")
    value = source(config)
    value = replace_once(
        value,
        "MIN_RECOGNITION_BATCH_SIZE = 8\nFRAME_PROCESS_CHUNK_SIZE = 128",
        "MIN_RECOGNITION_BATCH_SIZE = 8\nAUTO_TUNE_BATCH = True\nDETECTION_BATCH_CANDIDATES = (16, 32, 64, 96, 128)\nRECOGNITION_BATCH_CANDIDATES = (64, 128, 256, 384, 512)\nFRAME_PROCESS_CHUNK_SIZE = 128",
    )
    set_source(config, value)
    value = source(config)
    value = value.replace(
        r'''if not SHOT_FILE.is_file() or not KEYFRAME_FILE.is_file():
    raise FileNotFoundError(
        f"Need shot/keyframe CSV:\n"
        f"SHOT_FILE={SHOT_FILE}\n"
        f"KEYFRAME_FILE={KEYFRAME_FILE}"
    )''',
        '''if not SHOT_FILE.is_file() or not KEYFRAME_FILE.is_file():
    print("Configured OCR CSV files are absent; input discovery will search mounted datasets.")''',
        1,
    )
    set_source(config, value)

    input_cell = find_cell(notebook, "def load_input()")
    value = source(input_cell)
    discovery = '''def discover_ocr_inputs() -> None:
    global INPUT_DIR, SHOT_FILE, KEYFRAME_FILE, FRAME_ROOT
    if SHOT_FILE.is_file() and KEYFRAME_FILE.is_file(): return
    roots = [INPUT_DIR, Path("/kaggle/input")]
    keyframes = sorted({candidate for root in roots if root.is_dir() for name in ("keyframe.csv", "keyframes.csv") for candidate in root.rglob(name)})
    for keyframe in keyframes:
        directory = keyframe.parent
        shot = directory / "shot.csv" if (directory / "shot.csv").is_file() else directory / "shots.csv"
        if shot.is_file():
            INPUT_DIR, SHOT_FILE, KEYFRAME_FILE, FRAME_ROOT = directory, shot, keyframe, directory / "keyframes"
            print("Using OCR input:", directory); return
    raise FileNotFoundError("Drop shot.csv/shots.csv and keyframe.csv/keyframes.csv into one input directory")

discover_ocr_inputs()

'''
    if "def discover_ocr_inputs" not in value:
        value = discovery + value
    value = value.replace(
        '''    path = Path(raw).expanduser(); choices = ([path] if path.is_absolute() else []) + [FRAME_ROOT / path, FRAME_ROOT / path.name]
    return next((item.resolve() for item in choices if item.is_file()), None)''',
        '''    path = Path(raw).expanduser(); choices = ([path] if path.is_absolute() else []) + [KEYFRAME_FILE.parent / path, INPUT_DIR / path, FRAME_ROOT / path, FRAME_ROOT / path.name]
    return next((item.resolve() for item in choices if item.is_file()), None)''',
        1,
    )
    set_source(input_cell, value)

    engine_cell = find_cell(notebook, "def infer_chunk(engine: OCREngine")
    value = source(engine_cell)
    if "def autotune(call" not in value:
        value = replace_once(
            value,
            "def adaptive(call,values: list[Any],initial: int,minimum: int) -> tuple[list[Any],int]:",
            '''def autotune(call, values: list[Any], candidates: tuple[int, ...], fallback: int) -> int:
    if not AUTO_TUNE_BATCH or not values: return fallback
    best_batch, best_rate = fallback, 0.0; trials=[]
    for candidate in candidates:
        probe=values[:min(candidate,len(values))]
        if not probe: continue
        try:
            started=time.perf_counter(); call(probe,candidate); elapsed=max(time.perf_counter()-started,1e-6); rate=len(probe)/elapsed
            trials.append({"batch_size":candidate,"items":len(probe),"items_per_second":round(rate,3)})
            if rate>=best_rate*0.98: best_batch,best_rate=candidate,rate
            if len(probe)<candidate: break
        except RuntimeError as error:
            if not is_oom(error): raise
            gc.collect(); torch.cuda.empty_cache(); break
    print("OCR batch autotune:",trials,"selected=",best_batch); return best_batch

def reading_order(regions: list[tuple[np.ndarray,float]]) -> list[tuple[np.ndarray,float]]:
    if not regions: return []
    heights=[max(np.linalg.norm(polygon[3]-polygon[0]),np.linalg.norm(polygon[2]-polygon[1])) for polygon,_ in regions]
    tolerance=max(8.0,float(np.median(heights))*0.5)
    return sorted(regions,key=lambda item:(round(float(item[0][:,1].mean())/tolerance),float(item[0][:,0].mean())))

def adaptive(call,values: list[Any],initial: int,minimum: int) -> tuple[list[Any],int]:''',
        )
        value = replace_once(
            value,
            "detected,det_batch=adaptive(engine.detect,images,DETECTION_BATCH_SIZE,MIN_DETECTION_BATCH_SIZE); crops,metadata=[],[]",
            "det_initial=getattr(engine,'det_batch',None) or autotune(engine.detect,images,DETECTION_BATCH_CANDIDATES,DETECTION_BATCH_SIZE)\n    engine.det_batch=det_initial\n    detected,det_batch=adaptive(engine.detect,images,det_initial,MIN_DETECTION_BATCH_SIZE); crops,metadata=[],[]",
        )
        value = replace_once(
            value,
            "for polygon,_ in sorted([(p,s) for p,s in regions if s>=DETECTION_BOX_THRESHOLD],key=lambda item:float(item[0][:,1].mean())):",
            "for polygon,_ in reading_order([(polygon4(p,w,h),s) for p,s in regions if s>=DETECTION_BOX_THRESHOLD]):",
        )
        value = replace_once(
            value,
            "recognized,rec_batch=adaptive(engine.recognize,crops,RECOGNITION_BATCH_SIZE,MIN_RECOGNITION_BATCH_SIZE); rows=[]; count=defaultdict(int)",
            "rec_initial=getattr(engine,'rec_batch',None) or autotune(engine.recognize,crops,RECOGNITION_BATCH_CANDIDATES,RECOGNITION_BATCH_SIZE)\n    engine.rec_batch=rec_initial\n    recognized,rec_batch=adaptive(engine.recognize,crops,rec_initial,MIN_RECOGNITION_BATCH_SIZE); rows=[]; count=defaultdict(int)",
        )
    set_source(engine_cell, value)

    add_zip_cell(notebook, variable="output", prefix="OCR")
    save(path, notebook)


def patch_shot() -> None:
    path = ROOT / "ShotEmbedding/shot_embedding_siglip_kaggle.ipynb"
    notebook = load(path)

    setup = find_cell(notebook, "CLIP_ARTIFACT_ROOT = INPUT_DIR")
    value = source(setup)
    discovery = '''def discover_shot_input() -> None:
    global INPUT_DIR, SHOT_FILE, CLIP_ARTIFACT_ROOT
    if SHOT_FILE.is_file() and (CLIP_ARTIFACT_ROOT / "manifest.json").is_file(): return
    roots = [INPUT_DIR, Path("/kaggle/input")]
    manifests = sorted({candidate for root in roots if root.is_dir() for candidate in root.rglob("manifest.json")})
    for manifest in manifests:
        try: payload=json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): continue
        if str(payload.get("entity_type")) != "clip": continue
        candidates=[manifest.parent/"shot.csv",manifest.parent/"shots.csv",manifest.parent.parent/"shot.csv",manifest.parent.parent/"shots.csv"]
        shot=next((candidate for candidate in candidates if candidate.is_file()),None)
        if shot is not None:
            INPUT_DIR,SHOT_FILE,CLIP_ARTIFACT_ROOT=shot.parent,shot,manifest.parent; print("Using Shot input:",manifest.parent); return
    raise FileNotFoundError("Drop a Clip artifact folder plus shot.csv/shots.csv into the notebook input")

discover_shot_input()

'''
    marker = "if not SHOT_FILE.is_file():"
    if "def discover_shot_input" not in value:
        value = value.replace(marker, discovery + marker, 1)
    set_source(setup, value)

    process = find_cell(notebook, "# Process one Shot at a time")
    value = source(process)
    value = value.replace(
        "if sum(len(pd.read_parquet(path)) for path in output_metadata_paths) != len(output_metadata):\n\n    raise RuntimeError('Shot metadata shard count mismatch')",
        "if len(output_vector_paths) != len(output_metadata_paths):\n\n    raise RuntimeError('Shot vector/metadata shard count mismatch')",
        1,
    )
    set_source(process, value)

    add_zip_cell(notebook, variable="final_dir", prefix="Shot")
    save(path, notebook)


def main() -> None:
    patch_frame()
    patch_clip()
    patch_ocr()
    patch_shot()
    print("Optimized runtime paths, batching, prefetch, and ZIP packaging for four notebooks.")


if __name__ == "__main__":
    main()
