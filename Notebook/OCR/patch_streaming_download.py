"""Replace OCR's all-at-once URL download stage with bounded prefetching."""
from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("ocr_kaggle.ipynb")

DOWNLOAD_CELL = r'''# [4/6] Bounded URL download helpers. Only one OCR chunk is prefetched at a time.
_size_lock = threading.Lock()
WORKING_BYTES = sum(path.stat().st_size for path in OUTPUT_DIR.rglob("*") if path.is_file()) if OUTPUT_DIR.exists() else 0
IMAGE_URL_BY_FRAME = FRAMES.set_index("frame_id")["image_url"] if "image_url" in FRAMES else pd.Series(dtype="string")
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Kaggle-Colab-OCR/1.0)",
    "Accept": "image/avif,image/webp,image/jpeg,image/*,*/*;q=0.8",
}

def reserve_size(amount: int) -> None:
    global WORKING_BYTES
    with _size_lock:
        if WORKING_BYTES + amount > MAX_WORKING_BYTES:
            raise RuntimeError("Kaggle working output would exceed MAX_WORKING_BYTES")
        WORKING_BYTES += amount

def remove_cached(path: Path) -> None:
    global WORKING_BYTES
    if path.is_file():
        size = path.stat().st_size
        path.unlink()
        with _size_lock:
            WORKING_BYTES = max(0, WORKING_BYTES - size)

def download_one(row: dict[str, Any]) -> Path:
    url = optional(row.get("image_url"))
    if url is None or not url.startswith(("http://", "https://")):
        raise ValueError("image_url is required for a missing local image")
    suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower() or ".jpg"
    target = IMAGE_CACHE_DIR / f"{row['frame_id']}{suffix}"
    if target.is_file() and target.stat().st_size > 0:
        return target
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        part = target.with_suffix(target.suffix + ".part")
        try:
            request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, part.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    reserve_size(len(chunk))
                    handle.write(chunk)
            if part.stat().st_size == 0:
                raise RuntimeError("empty image download")
            part.replace(target)
            return target
        except Exception:
            remove_cached(part)
            if attempt == DOWNLOAD_RETRIES:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")

def prepare_chunk(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve local paths and download only this chunk's missing images concurrently."""
    pending, ready, failures = [], [], []
    for record in records:
        row = dict(record)
        if optional(row.get("resolved_path")):
            ready.append(row)
            continue
        row["image_url"] = IMAGE_URL_BY_FRAME.get(row["frame_id"])
        pending.append(row)
    if not pending:
        return ready, failures
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_one, row): row for row in pending}
        for future in as_completed(futures):
            row = futures[future]
            try:
                row["resolved_path"] = str(future.result())
                ready.append(row)
            except Exception as error:
                failures.append({"frame_id": row["frame_id"], "reason": "image_download_failed", "error": str(error)})
    ready.sort(key=lambda row: (row["video_id"], row["timestamp_ms"], row["frame_idx"], row["frame_id"]))
    return ready, failures

print("[4/6] URL images will be prefetched one OCR chunk ahead; no full candidate download is performed.")
'''


def patch() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("# [4/6] Download only unresolved selected candidates"):
            cell["source"] = DOWNLOAD_CELL.splitlines(keepends=True)
        if source.startswith("# [5/6] GPU OCR with per-chunk checkpoint"):
            old_checkpoint_io = '''def append_jsonl(path: Path,items: Iterable[dict[str,Any]]) -> None:
    with path.open("a",encoding="utf-8") as handle:
        for item in items: handle.write(json.dumps(item,ensure_ascii=False)+"\\\\n")
def read_jsonl(path: Path) -> list[dict[str,Any]]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.is_file() else []'''
            new_checkpoint_io = '''def append_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write(chr(10))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read normal JSONL and legacy checkpoint files that used literal backslash-n separators."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    decoder, cursor, items, legacy_separator = json.JSONDecoder(), 0, [], chr(92) + "n"
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if text.startswith(legacy_separator, cursor):
            cursor += len(legacy_separator)
            continue
        if cursor >= len(text):
            break
        item, cursor = decoder.raw_decode(text, cursor)
        items.append(item)
    return items'''
            source = source.replace(old_checkpoint_io, new_checkpoint_io)
            source = source.replace("len(OCR_CANDIDATES)/elapsed", "len(CANDIDATES)/elapsed")
            before, marker, _ = source.partition("if RUN_PIPELINE:")
            if not marker:
                raise RuntimeError("OCR run block was not found")
            run_block = r'''if RUN_PIPELINE:
    started = time.monotonic()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    completed = {item["frame_id"] for item in read_jsonl(CHECKPOINT_DIR / "completed.jsonl")}
    pending = CANDIDATES.loc[~CANDIDATES.frame_id.isin(completed)].copy()
    chunk_starts = iter(range(0, len(pending), FRAME_PROCESS_CHUNK_SIZE))
    final_det, final_rec = DETECTION_BATCH_SIZE, RECOGNITION_BATCH_SIZE
    print(f"[5/6] Running OCR: already OCR'd={len(completed):,}; remaining={len(pending):,}; download workers={MAX_DOWNLOAD_WORKERS}")
    engine = OCREngine()
    try:
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            def submit_next():
                start = next(chunk_starts, None)
                if start is None:
                    return None
                records = pending.iloc[start:start + FRAME_PROCESS_CHUNK_SIZE].to_dict("records")
                return start, prefetch.submit(prepare_chunk, records)

            current = submit_next()
            with tqdm(total=len(pending), desc="OCR candidates", unit="image") as progress:
                while current is not None:
                    start, future = current
                    records, download_failures = future.result()
                    current = submit_next()  # Downloads overlap GPU OCR for this chunk.
                    progress.update(FRAME_PROCESS_CHUNK_SIZE if start + FRAME_PROCESS_CHUNK_SIZE <= len(pending) else len(pending) - start)
                    append_jsonl(CHECKPOINT_DIR / "failures.jsonl", download_failures)
                    if not records:
                        continue
                    try:
                        rows, chunk_failures, successful, final_det, final_rec = infer_chunk(engine, records)
                        # Commit result rows and completion state before cache removal.
                        append_jsonl(CHECKPOINT_DIR / "raw.jsonl", rows)
                        append_jsonl(CHECKPOINT_DIR / "failures.jsonl", chunk_failures)
                        append_jsonl(CHECKPOINT_DIR / "completed.jsonl", ({"frame_id": frame_id} for frame_id in successful))
                        for item in records:
                            path = Path(item["resolved_path"])
                            if IMAGE_CACHE_DIR in path.parents:
                                remove_cached(path)
                    except Exception as error:
                        if not CONTINUE_ON_ERROR:
                            raise
                        append_jsonl(CHECKPOINT_DIR / "failures.jsonl", [{"reason": "chunk_failed", "frame_ids": [item["frame_id"] for item in records], "error": str(error), "traceback": traceback.format_exc(limit=3)}])
        print("[6/6] Writing outputs")
        failures = PLAN_FAILURES + read_jsonl(CHECKPOINT_DIR / "failures.jsonl")
        output = write_output(read_jsonl(CHECKPOINT_DIR / "raw.jsonl"), failures, started, final_det, final_rec)
        print("Completed. Kaggle output:", output)
        display(pd.read_csv(output / "ocr.csv").head())
    finally:
        engine.close()
else:
    print("Dry plan only. Set RUN_PIPELINE=True to execute GPU OCR.")
'''
            cell["source"] = (before + run_block).splitlines(keepends=True)
        source = "".join(cell.get("source", []))
        if source.startswith("# Configuration. Keep RUN_ID unchanged to resume"):
            source = source.replace('INPUT_DIR = Path("/content")', 'INPUT_DIR = Path("/home/nhminh/AI_Project/4US-1F-AIC-HCMC/Notebook/test_data")')
            source = source.replace('OUTPUT_DIR = Path("/content")', 'OUTPUT_DIR = Path("/tmp/ocr_local_output")')
            source = source.replace('OUTPUT_DIR = Path("/content/ocr_output")', 'OUTPUT_DIR = Path("/tmp/ocr_local_output")')
            source = source.replace('DETECTION_BATCH_SIZE = 32', 'DETECTION_BATCH_SIZE = 64')
            source = source.replace('RECOGNITION_BATCH_SIZE = 128', 'RECOGNITION_BATCH_SIZE = 256')
            source = source.replace('FRAME_PROCESS_CHUNK_SIZE = 64', 'FRAME_PROCESS_CHUNK_SIZE = 128')
            source = source.replace('MAX_DOWNLOAD_WORKERS = 6', 'MAX_DOWNLOAD_WORKERS = 8')
            cell["source"] = source.splitlines(keepends=True)
        source = "".join(cell.get("source", []))
        if source.startswith("# [4/6] Bounded URL download helpers") and "DOWNLOAD_HEADERS" not in source:
            source = source.replace(
                'IMAGE_URL_BY_FRAME = FRAMES.set_index("frame_id")["image_url"] if "image_url" in FRAMES else pd.Series(dtype="string")',
                'IMAGE_URL_BY_FRAME = FRAMES.set_index("frame_id")["image_url"] if "image_url" in FRAMES else pd.Series(dtype="string")\nDOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Kaggle-Colab-OCR/1.0)", "Accept": "image/avif,image/webp,image/jpeg,image/*,*/*;q=0.8"}',
            ).replace(
                'with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, part.open("wb") as handle:',
                'request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)\n            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, part.open("wb") as handle:',
            )
            cell["source"] = source.splitlines(keepends=True)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


patch()
