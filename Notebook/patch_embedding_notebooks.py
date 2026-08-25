"""Apply consistent per-artifact FAISS ID state to embedding notebooks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent


def replace_cell(path: Path, old: str, new: str) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if new in "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]):
        return
    replaced = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if old in source:
            cell["source"] = source.replace(old, new).splitlines(keepends=True)
            replaced = True
    if not replaced:
        raise RuntimeError(f"Expected source was not found in {path}: {old[:80]!r}")
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


frame = ROOT / "FrameEmbedding/frame_embedding_siglip_kaggle.ipynb"
replace_cell(
    frame,
    'videos = sorted({row["video_id"] for row in valid})[VIDEO_START:VIDEO_END]\n    selected = [row for row in valid if row["video_id"] in set(videos)]',
    'videos = set(sorted({row["video_id"] for row in valid})[VIDEO_START:VIDEO_END])\n    selected = [row for row in valid if row["video_id"] in videos]',
)

# Repair notebooks produced by the first non-idempotent version of this migrator.
notebook = json.loads(frame.read_text(encoding="utf-8"))
duplicate = 'checkpoint = temp_dir / "checkpoint.json"\n    state = {"next_faiss_id": 1, "state": "in_progress"}\n    atomic_json(checkpoint, state)\n    checkpoint = temp_dir / "checkpoint.json"\n    state = {"next_faiss_id": 1, "state": "in_progress"}\n    atomic_json(checkpoint, state)'
single = 'checkpoint = temp_dir / "checkpoint.json"\n    state = {"next_faiss_id": 1, "state": "in_progress"}\n    atomic_json(checkpoint, state)'
for cell in notebook["cells"]:
    cell["source"] = "".join(cell.get("source", [])).replace(duplicate, single).splitlines(keepends=True)
frame.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
replace_cell(
    frame,
    'checkpoint = temp_dir / "checkpoint.json"\n    state = {"next_faiss_id": 1, "state": "in_progress"}\n    atomic_json(checkpoint, state)\n    checkpoint = temp_dir / "checkpoint.json"\n    state = {"next_faiss_id": 1, "state": "in_progress"}\n    atomic_json(checkpoint, state)',
    'checkpoint = temp_dir / "checkpoint.json"\n    state = {"next_faiss_id": 1, "state": "in_progress"}\n    atomic_json(checkpoint, state)',
)
replace_cell(
    frame,
    'temp_dir.mkdir(parents=True); (temp_dir / "vectors").mkdir(); (temp_dir / "metadata").mkdir()\n    failures = temp_dir / "failures.jsonl"; failures.touch()',
    'temp_dir.mkdir(parents=True); (temp_dir / "vectors").mkdir(); (temp_dir / "metadata").mkdir()\n    failures = temp_dir / "failures.jsonl"; failures.touch()\n    checkpoint = temp_dir / "checkpoint.json"\n    state = {"next_faiss_id": 1, "state": "in_progress"}\n    atomic_json(checkpoint, state)',
)
replace_cell(
    frame,
    'ids = np.arange(success + 1, success + len(good) + 1, dtype=np.int64); index.add_with_ids(vectors, ids)',
    'ids = np.arange(state["next_faiss_id"], state["next_faiss_id"] + len(good), dtype=np.int64); index.add_with_ids(vectors, ids)\n        state["next_faiss_id"] += len(good)',
)
replace_cell(
    frame,
    'pending_vectors, pending_rows = [], []; shard_number += 1',
    'pending_vectors, pending_rows = [], []; shard_number += 1\n        atomic_json(checkpoint, {**state, "last_committed_shard": shard_number - 1})',
)

clip = ROOT / "ClipEmbedding/clip_embedding_siglip_kaggle.ipynb"
replace_cell(clip, '"next_faiss_id":0', '"next_faiss_id":1')

shot = ROOT / "ShotEmbedding/shot_embedding_siglip_kaggle.ipynb"
replace_cell(
    shot,
    "faiss_id = 1; shard_index = 0; clips_consumed = 0; completed_videos: list[str] = []",
    "faiss_id = int(json.loads(checkpoint_path.read_text(encoding='utf-8'))['next_faiss_id']); shard_index = 0; clips_consumed = 0; completed_videos: list[str] = []",
)
replace_cell(
    shot,
    "{'embedding_id', 'entity_id', 'shot_id', 'video_id', 'start_ms', 'end_ms', 'vector_shard', 'vector_row', 'status', 'model_revision', 'dimension', 'normalized'}",
    "{'faiss_id', 'clip_id', 'shot_id', 'video_id', 'start_ms', 'end_ms', 'vector_shard', 'vector_row', 'model_revision', 'dimension', 'normalized'}",
)
replace_cell(shot, "clip_metadata = clip_metadata[clip_metadata.status.astype(str).eq('success')].copy()\n", "clip_metadata = clip_metadata.copy()\n")
replace_cell(shot, "str(row['embedding_id'])", "str(row['faiss_id'])")
replace_cell(shot, "str(row['entity_id'])", "str(row['clip_id'])")
replace_cell(shot, "str(row['embedding_id']))", "str(row['faiss_id']))")
replace_cell(shot, "row.entity_id", "row.clip_id")
replace_cell(
    shot,
    "mmap_shards = {path: np.load(resolve_artifact_path(path), mmap_mode='r') for path in vector_paths}",
    "mmap_shards = {str(index): np.load(resolve_artifact_path(path), mmap_mode='r') for index, path in enumerate(vector_paths)}",
)


def add_streaming_clip_finalize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    marker = "# Streaming finalization: one shard at a time; IndexFlat still owns the final index vectors."
    if any(marker in "".join(cell.get("source", [])) for cell in notebook["cells"]):
        return
    source = '''# Streaming finalization: one shard at a time; IndexFlat still owns the final index vectors.
def finalize(root: Path, requested_n: int, decoded_n: int, success: int, failed: int, decode_s: float, encode_s: float, peak: int, batch: int) -> None:
    vector_paths = sorted((root / "vectors").glob("part-*.npy")); metadata_paths = sorted((root / "metadata").glob("part-*.parquet"))
    if len(vector_paths) != len(metadata_paths): raise RuntimeError("vector/metadata shard count mismatch")
    index, dim, total, seen_ids, seen_clips = None, None, 0, set(), set()
    with (root / "clip_embedding_mapping.csv").open("w", newline="", encoding="utf-8") as mapping, (root / "insert_clip_embedding_records.sql").open("w", encoding="utf-8") as sql:
        writer = csv.DictWriter(mapping, fieldnames=["faiss_id", "index_version", "clip_id", "model_name"]); writer.writeheader()
        sql.write("-- Full rebuild artifact only; do not append to an existing production FAISS ID space.\\n")
        for vector_path, metadata_path in zip(vector_paths, metadata_paths):
            matrix = np.ascontiguousarray(np.load(vector_path, mmap_mode="r"), dtype=np.float32)
            metadata = pq.read_table(metadata_path).to_pylist()
            if len(matrix) != len(metadata) or matrix.ndim != 2 or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4): raise RuntimeError(f"invalid shard {vector_path.name}")
            if dim is None: dim = matrix.shape[1]; index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
            if matrix.shape[1] != dim: raise RuntimeError("embedding dimensions differ across shards")
            ids = np.asarray([row["faiss_id"] for row in metadata], dtype=np.int64)
            clips = [row["clip_id"] for row in metadata]
            if len(set(ids)) != len(ids) or len(set(clips)) != len(clips) or seen_ids.intersection(ids) or seen_clips.intersection(clips): raise RuntimeError("duplicate FAISS ID or clip_id")
            index.add_with_ids(matrix, ids); seen_ids.update(map(int, ids)); seen_clips.update(clips); total += len(metadata)
            for row in metadata:
                writer.writerow({"faiss_id": row["faiss_id"], "index_version": INDEX_VERSION, "clip_id": row["clip_id"], "model_name": MODEL_NAME})
                clip_id = str(row["clip_id"]).replace("'", "''")
                sql.write(f"INSERT INTO clipembeddingrecord (faiss_id, index_version, clip_id, model_name) VALUES ({row['faiss_id']}, {INDEX_VERSION}, '{clip_id}', '{MODEL_NAME}') ON CONFLICT (clip_id, index_version) DO NOTHING;\\n")
    if index is None: raise RuntimeError("no successful clip embeddings")
    faiss.write_index(index, str(root / "clip.faiss")); assert index.ntotal == total
    checks = {str(item.relative_to(root)): sha256_file(item) for item in root.rglob("*") if item.is_file() and item.name not in {"manifest.json", "summary.json"}}
    manifest = {"entity_type": "clip", "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "normalized": True, "sampling_version": "uniform_midpoint_16_v1", "pooling_version": "masked_mean_v1", "success_count": total, "failure_count": failed, "dimension": dim, "vector_shards": [str(item.relative_to(root)) for item in vector_paths], "metadata_shards": [str(item.relative_to(root)) for item in metadata_paths], "checksums": checks, "faiss_checksum": checks.get("clip.faiss")}
    atomic_json(root / "manifest.json", manifest)
    summary = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "model_revision": MODEL_REVISION, "dimension": dim, "clips_success": total, "clips_failed": failed, "unique_decoded_frames": decoded_n, "requested_frames": requested_n, "decode_seconds": decode_s, "encode_seconds": encode_s, "peak_vram_bytes": peak, "final_batch_size": batch}
    atomic_json(root / "summary.json", summary); print("Download Output:", root)
'''
    notebook["cells"].insert(-1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)})
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


add_streaming_clip_finalize(clip)
replace_cell(
    shot,
    "checkpoint_path.write_text(json.dumps({'config_hash': config_hash, 'completed_video_ids': completed_videos, 'next_faiss_id': faiss_id, 'state': 'in_progress'}, indent=2))",
    "checkpoint_tmp = checkpoint_path.with_suffix('.tmp'); checkpoint_tmp.write_text(json.dumps({'config_hash': config_hash, 'completed_video_ids': completed_videos, 'next_faiss_id': faiss_id, 'state': 'in_progress'}, indent=2), encoding='utf-8'); checkpoint_tmp.replace(checkpoint_path)",
)
