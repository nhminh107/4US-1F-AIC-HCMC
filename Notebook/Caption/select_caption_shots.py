"""Select a reproducible, caption-worthy subset of shots from local indexes.

The selector is deliberately read-only with respect to PostgreSQL and the
existing FAISS artifacts.  It writes a Kaggle-compatible shot CSV plus an
audit table that explains every selection decision.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable

import faiss
import numpy as np
from sqlalchemy import and_, func, select

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BackEnd.app.database.models import (
    ClipEmbeddingRecord,
    ClipWindow,
    Frame,
    FrameEmbeddingRecord,
    OCR,
    Shot,
    ShotEmbeddingRecord,
    TranscriptSegment,
)
from BackEnd.app.database.postgre_db import PostgreManager


SHOT_COLUMNS = (
    "shot_id",
    "video_id",
    "shot_index",
    "start_ms",
    "end_ms",
    "start_frame_idx",
    "end_frame_idx",
)
WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    shots_csv: Path
    faiss_dir: Path
    output_dir: Path
    target: int
    index_version: int
    model_name: str
    model_version: str
    pooling_method: str
    batch_size: int
    seed: int


@dataclass(frozen=True, slots=True)
class ShotRecord:
    shot_id: str
    video_id: str
    shot_index: int
    start_ms: int
    end_ms: int
    start_frame_idx: str
    end_frame_idx: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def parse_args() -> SelectorConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots-csv",
        type=Path,
        default=REPO_ROOT / "Notebook/content/shots.csv",
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=REPO_ROOT / "BackEnd/app/database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Notebook/content/caption_selection",
    )
    parser.add_argument("--target", type=int, default=18_000)
    parser.add_argument("--index-version", type=int, default=0)
    parser.add_argument("--model-name", default="clip-ViT-B-32")
    parser.add_argument("--model-version", default="0")
    parser.add_argument("--pooling-method", default="mean")
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=20260827)
    arguments = parser.parse_args()
    if not 15_000 <= arguments.target <= 20_000:
        parser.error("--target must be in [15000, 20000].")
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    return SelectorConfig(
        shots_csv=arguments.shots_csv.resolve(),
        faiss_dir=arguments.faiss_dir.resolve(),
        output_dir=arguments.output_dir.resolve(),
        target=arguments.target,
        index_version=arguments.index_version,
        model_name=arguments.model_name,
        model_version=arguments.model_version,
        pooling_method=arguments.pooling_method,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
    )


def load_shots(path: Path) -> list[ShotRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Shot CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(SHOT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Shot CSV is missing columns: {sorted(missing)}")
        shots: list[ShotRecord] = []
        seen_ids: set[str] = set()
        for row in reader:
            shot = ShotRecord(
                shot_id=str(row["shot_id"]).strip(),
                video_id=str(row["video_id"]).strip(),
                shot_index=int(row["shot_index"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                start_frame_idx=str(row["start_frame_idx"]),
                end_frame_idx=str(row["end_frame_idx"]),
            )
            if not shot.shot_id or shot.shot_id in seen_ids:
                raise ValueError(f"Duplicate or blank shot_id: {shot.shot_id!r}")
            if shot.end_ms <= shot.start_ms:
                raise ValueError(f"Invalid shot interval: {shot.shot_id}")
            seen_ids.add(shot.shot_id)
            shots.append(shot)
    if not shots:
        raise ValueError("Shot CSV has no rows.")
    return shots


def read_index(faiss_dir: Path, kind: str, expected_dimension: int | None = None) -> faiss.Index:
    path = faiss_dir / f"{kind}.faiss"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FAISS artifact: {path}")
    index = faiss.read_index(str(path))
    if not isinstance(index, faiss.IndexIDMap2):
        raise ValueError(f"{kind}.faiss must use IndexIDMap2.")
    if expected_dimension is not None and index.d != expected_dimension:
        raise ValueError(
            f"{kind}.faiss has dimension {index.d}, expected {expected_dimension}."
        )
    return index


def faiss_ids(index: faiss.Index) -> set[int]:
    return set(faiss.vector_to_array(index.id_map).astype(np.int64, copy=False).tolist())


def reconstruct_batch(index: faiss.Index, ids: Iterable[int]) -> np.ndarray:
    values = np.asarray(list(ids), dtype=np.int64)
    if values.size == 0:
        return np.empty((0, index.d), dtype=np.float32)
    try:
        vectors = index.reconstruct_batch(values)
    except (AttributeError, RuntimeError, TypeError):
        vectors = np.asarray([index.reconstruct(int(item)) for item in values], dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)
    return vectors


def reconstruct_all(index: faiss.Index, ids: list[int], batch_size: int) -> np.ndarray:
    batches = [
        reconstruct_batch(index, ids[start : start + batch_size])
        for start in range(0, len(ids), batch_size)
    ]
    return np.vstack(batches) if batches else np.empty((0, index.d), dtype=np.float32)


def percentile_rank(values: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    result = np.zeros(values.shape[0], dtype=np.float32)
    mask = np.isfinite(values)
    if eligible is not None:
        mask &= eligible
    positions = np.flatnonzero(mask)
    if positions.size <= 1:
        return result
    ordered = positions[np.argsort(values[positions], kind="stable")]
    result[ordered] = np.linspace(0.0, 1.0, ordered.size, dtype=np.float32)
    return result


def shot_lookup_by_video(shots: list[ShotRecord]) -> dict[str, tuple[list[int], list[str], list[int]]]:
    grouped: dict[str, list[ShotRecord]] = {}
    for shot in shots:
        grouped.setdefault(shot.video_id, []).append(shot)
    lookup: dict[str, tuple[list[int], list[str], list[int]]] = {}
    for video_id, values in grouped.items():
        values.sort(key=lambda item: (item.start_ms, item.shot_index, item.shot_id))
        lookup[video_id] = (
            [item.start_ms for item in values],
            [item.shot_id for item in values],
            [item.end_ms for item in values],
        )
    return lookup


def temporal_shot_id(
    video_lookup: dict[str, tuple[list[int], list[str], list[int]]],
    video_id: str,
    timestamp_ms: int,
) -> str | None:
    item = video_lookup.get(video_id)
    if item is None:
        return None
    starts, shot_ids, ends = item
    position = bisect_right(starts, timestamp_ms) - 1
    if position < 0 or timestamp_ms >= ends[position]:
        return None
    return shot_ids[position]


def load_shot_mapping(
    manager: PostgreManager,
    config: SelectorConfig,
    shot_index: faiss.Index,
    expected_shot_ids: set[str],
) -> list[tuple[int, str]]:
    statement = (
        select(ShotEmbeddingRecord.faiss_id, ShotEmbeddingRecord.shot_id)
        .where(
            ShotEmbeddingRecord.index_version == config.index_version,
            ShotEmbeddingRecord.model_name == config.model_name,
            ShotEmbeddingRecord.model_version == config.model_version,
            ShotEmbeddingRecord.pooling_method == config.pooling_method,
        )
        .order_by(ShotEmbeddingRecord.faiss_id)
    )
    with manager.session_factory() as session:
        mappings = [(int(faiss_id), str(shot_id)) for faiss_id, shot_id in session.execute(statement)]
    mapped_ids = {shot_id for _, shot_id in mappings}
    mapped_faiss_ids = {faiss_id for faiss_id, _ in mappings}
    if mapped_ids != expected_shot_ids:
        raise RuntimeError(
            "Shot embedding mapping does not match shots CSV: "
            f"missing={len(expected_shot_ids - mapped_ids)}, extra={len(mapped_ids - expected_shot_ids)}"
        )
    if mapped_faiss_ids != faiss_ids(shot_index):
        raise RuntimeError("Shot embedding mapping does not match the FAISS ID map.")
    if len(mappings) != len(mapped_ids):
        raise RuntimeError("Shot embedding mapping has duplicate shot IDs.")
    return mappings


def compute_neighbors(
    vectors: np.ndarray,
    ids: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find deterministic anchor neighbours without changing persisted FAISS data.

    A compact anchor index avoids materialising a second full 96k-vector FAISS
    index beside the production frame index, which is unnecessarily memory
    intensive on a local workstation.
    """
    count, dimension = vectors.shape
    anchor_count = min(4_096, count)
    generator = np.random.default_rng(seed)
    anchor_positions = np.sort(generator.choice(count, size=anchor_count, replace=False))
    temporary = faiss.IndexFlatIP(dimension)
    temporary.add(vectors[anchor_positions])

    similarities = np.full(count, -1.0, dtype=np.float32)
    neighbour_ids = np.full(count, -1, dtype=np.int64)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        distances, found_positions = temporary.search(vectors[start:stop], min(8, anchor_count))
        for offset, current_id in enumerate(ids[start:stop]):
            for distance, found_position in zip(
                distances[offset], found_positions[offset], strict=True
            ):
                found_id = ids[anchor_positions[found_position]] if found_position >= 0 else -1
                if found_id >= 0 and found_id != current_id:
                    similarities[start + offset] = float(distance)
                    neighbour_ids[start + offset] = int(found_id)
                    break
    return similarities, neighbour_ids


def visual_signatures(vectors: np.ndarray, seed: int) -> np.ndarray:
    """Create compact random-hyperplane buckets for strict duplicate checks."""
    generator = np.random.default_rng(seed + 1)
    planes = generator.standard_normal((vectors.shape[1], 24), dtype=np.float32)
    bits = (vectors @ planes >= 0.0).astype(np.uint32)
    weights = (1 << np.arange(24, dtype=np.uint32)).reshape(1, -1)
    return (bits * weights).sum(axis=1, dtype=np.uint32)


def clip_change_by_shot(
    manager: PostgreManager,
    config: SelectorConfig,
    clip_index: faiss.Index,
    batch_size: int,
) -> tuple[dict[str, int], dict[str, float]]:
    statement = (
        select(ClipEmbeddingRecord.faiss_id, ClipWindow.shot_id, ClipWindow.start_ms)
        .join(ClipWindow, ClipEmbeddingRecord.clip_id == ClipWindow.clip_id)
        .where(
            ClipEmbeddingRecord.index_version == config.index_version,
            ClipEmbeddingRecord.model_name == config.model_name,
        )
        .order_by(ClipWindow.shot_id, ClipWindow.start_ms, ClipEmbeddingRecord.faiss_id)
    )
    counts: dict[str, int] = {}
    change_sums: dict[str, float] = {}
    change_counts: dict[str, int] = {}
    previous_shot_id: str | None = None
    previous_vector: np.ndarray | None = None
    with manager.session_factory() as session:
        result = session.execute(statement).yield_per(batch_size)
        pending = list(result.fetchmany(batch_size))
        while pending:
            vectors = reconstruct_batch(clip_index, [int(row[0]) for row in pending])
            for row, vector in zip(pending, vectors, strict=True):
                shot_id = str(row[1])
                counts[shot_id] = counts.get(shot_id, 0) + 1
                if shot_id == previous_shot_id and previous_vector is not None:
                    change = max(0.0, 1.0 - float(np.dot(previous_vector, vector)))
                    change_sums[shot_id] = change_sums.get(shot_id, 0.0) + change
                    change_counts[shot_id] = change_counts.get(shot_id, 0) + 1
                previous_shot_id = shot_id
                previous_vector = vector
            pending = list(result.fetchmany(batch_size))
    changes = {
        shot_id: change_sums.get(shot_id, 0.0) / change_counts.get(shot_id, 1)
        for shot_id in counts
    }
    return counts, changes


def frame_change_by_shot(
    manager: PostgreManager,
    config: SelectorConfig,
    frame_index: faiss.Index,
    video_lookup: dict[str, tuple[list[int], list[str], list[int]]],
    known_shot_ids: set[str],
    batch_size: int,
) -> tuple[dict[str, int], dict[str, float]]:
    statement = (
        select(
            FrameEmbeddingRecord.faiss_id,
            Frame.video_id,
            Frame.shot_id,
            Frame.timestamp_ms,
        )
        .join(Frame, FrameEmbeddingRecord.frame_id == Frame.frame_id)
        .where(
            FrameEmbeddingRecord.index_version == config.index_version,
            FrameEmbeddingRecord.model_name == config.model_name,
        )
        .order_by(Frame.video_id, Frame.timestamp_ms, FrameEmbeddingRecord.faiss_id)
    )
    counts: dict[str, int] = {}
    change_sums: dict[str, float] = {}
    change_counts: dict[str, int] = {}
    previous_shot_id: str | None = None
    previous_vector: np.ndarray | None = None
    with manager.session_factory() as session:
        result = session.execute(statement).yield_per(batch_size)
        pending = list(result.fetchmany(batch_size))
        while pending:
            vectors = reconstruct_batch(frame_index, [int(row[0]) for row in pending])
            for row, vector in zip(pending, vectors, strict=True):
                video_id = str(row[1])
                direct_shot_id = str(row[2]) if row[2] is not None else None
                shot_id = direct_shot_id if direct_shot_id in known_shot_ids else temporal_shot_id(
                    video_lookup, video_id, int(row[3])
                )
                if shot_id is None:
                    previous_shot_id = None
                    previous_vector = None
                    continue
                counts[shot_id] = counts.get(shot_id, 0) + 1
                if shot_id == previous_shot_id and previous_vector is not None:
                    change = max(0.0, 1.0 - float(np.dot(previous_vector, vector)))
                    change_sums[shot_id] = change_sums.get(shot_id, 0.0) + change
                    change_counts[shot_id] = change_counts.get(shot_id, 0) + 1
                previous_shot_id = shot_id
                previous_vector = vector
            pending = list(result.fetchmany(batch_size))
    changes = {
        shot_id: change_sums.get(shot_id, 0.0) / change_counts.get(shot_id, 1)
        for shot_id in counts
    }
    return counts, changes


def asr_by_shot(manager: PostgreManager) -> tuple[dict[str, float], dict[str, int]]:
    statement = (
        select(
            Shot.shot_id,
            Shot.start_ms,
            Shot.end_ms,
            TranscriptSegment.start_ms,
            TranscriptSegment.end_ms,
            TranscriptSegment.text,
        )
        .join(
            TranscriptSegment,
            and_(
                TranscriptSegment.video_id == Shot.video_id,
                TranscriptSegment.start_ms < Shot.end_ms,
                TranscriptSegment.end_ms > Shot.start_ms,
            ),
        )
    )
    intervals: dict[str, list[tuple[int, int]]] = {}
    words: dict[str, int] = {}
    durations: dict[str, int] = {}
    with manager.session_factory() as session:
        for shot_id, shot_start, shot_end, segment_start, segment_end, text in session.execute(statement).yield_per(5_000):
            key = str(shot_id)
            start = max(int(shot_start), int(segment_start))
            end = min(int(shot_end), int(segment_end))
            if end <= start:
                continue
            intervals.setdefault(key, []).append((start, end))
            words[key] = words.get(key, 0) + len(WORD_RE.findall(str(text)))
            durations[key] = int(shot_end) - int(shot_start)
    coverage: dict[str, float] = {}
    for shot_id, ranges in intervals.items():
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        covered_ms = sum(end - start for start, end in merged)
        coverage[shot_id] = min(1.0, covered_ms / durations[shot_id])
    return coverage, words


def ocr_by_shot(manager: PostgreManager) -> tuple[dict[str, int], dict[str, int]]:
    statement = (
        select(
            Shot.shot_id,
            func.count(func.distinct(Frame.frame_id)),
            func.count(func.distinct(func.lower(OCR.text))),
        )
        .join(
            Frame,
            and_(
                Frame.video_id == Shot.video_id,
                Frame.timestamp_ms >= Shot.start_ms,
                Frame.timestamp_ms < Shot.end_ms,
            ),
        )
        .join(OCR, OCR.frame_id == Frame.frame_id)
        .group_by(Shot.shot_id)
    )
    with manager.session_factory() as session:
        rows = list(session.execute(statement))
    return (
        {str(shot_id): int(frame_count) for shot_id, frame_count, _ in rows},
        {str(shot_id): int(text_count) for shot_id, _, text_count in rows},
    )


def optional_quotas(target: int, mandatory_count: int) -> tuple[int, int, int]:
    short_quota = round(target * (1_750 / 18_000))
    micro_quota = round(target * (200 / 18_000))
    medium_quota = target - mandatory_count - short_quota - micro_quota
    if medium_quota <= 0:
        raise ValueError("Target is too small after mandatory and short-shot quotas.")
    return medium_quota, short_quota, micro_quota


def select_ranked(
    candidates: list[int],
    quota: int,
    scores: np.ndarray,
    shot_ids: list[str],
    video_ids: list[str],
    nearest_ids: np.ndarray,
    nearest_similarity: np.ndarray,
    faiss_id_to_row: dict[int, int],
    text_coverage: np.ndarray,
    vectors: np.ndarray,
    signatures: np.ndarray,
    selected_rows: set[int],
) -> tuple[list[int], set[int]]:
    ranked = sorted(candidates, key=lambda row: (-float(scores[row]), shot_ids[row]))
    candidate_count_by_video: dict[str, int] = {}
    for row in candidates:
        candidate_count_by_video[video_ids[row]] = candidate_count_by_video.get(video_ids[row], 0) + 1
    soft_cap_by_video = {
        video_id: max(15, math.ceil(count * 0.80))
        for video_id, count in candidate_count_by_video.items()
    }
    selected_count_by_video: dict[str, int] = {}
    selected_by_signature: dict[int, list[int]] = {}
    for selected_row in selected_rows:
        selected_by_signature.setdefault(int(signatures[selected_row]), []).append(selected_row)
    duplicate_rows: set[int] = set()
    accepted: list[int] = []

    def is_duplicate(row: int) -> bool:
        neighbour_row = faiss_id_to_row.get(int(nearest_ids[row]))
        if (
            neighbour_row in selected_rows
            and nearest_similarity[row] >= 0.985
            and abs(float(text_coverage[row] - text_coverage[neighbour_row])) <= 0.25
        ):
            return True
        for selected_row in selected_by_signature.get(int(signatures[row]), []):
            if (
                float(np.dot(vectors[row], vectors[selected_row])) >= 0.985
                and abs(float(text_coverage[row] - text_coverage[selected_row])) <= 0.25
            ):
                return True
        return False

    def try_select(respect_cap: bool) -> None:
        for row in ranked:
            if len(accepted) >= quota or row in selected_rows:
                continue
            if is_duplicate(row):
                duplicate_rows.add(row)
                continue
            video_id = video_ids[row]
            if respect_cap and selected_count_by_video.get(video_id, 0) >= soft_cap_by_video[video_id]:
                continue
            selected_rows.add(row)
            accepted.append(row)
            selected_by_signature.setdefault(int(signatures[row]), []).append(row)
            selected_count_by_video[video_id] = selected_count_by_video.get(video_id, 0) + 1

    try_select(respect_cap=True)
    if len(accepted) < quota:
        try_select(respect_cap=False)
    if len(accepted) < quota:
        raise RuntimeError(f"Could only select {len(accepted)} of {quota} requested candidates.")
    return accepted, duplicate_rows


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = parse_args()
    print("[1/7] Loading shot CSV and FAISS indexes", flush=True)
    shots = load_shots(config.shots_csv)
    shot_by_id = {shot.shot_id: shot for shot in shots}
    shot_index = read_index(config.faiss_dir, "shot")
    manager = PostgreManager()
    mappings = load_shot_mapping(manager, config, shot_index, set(shot_by_id))
    faiss_id_by_shot = {shot_id: faiss_id for faiss_id, shot_id in mappings}

    shot_ids = [shot.shot_id for shot in shots]
    video_ids = [shot.video_id for shot in shots]
    ordered_faiss_ids = [faiss_id_by_shot[shot_id] for shot_id in shot_ids]
    print("[2/7] Reconstructing shot vectors and finding visual neighbours", flush=True)
    vectors = reconstruct_all(shot_index, ordered_faiss_ids, config.batch_size)
    shot_dimension = int(shot_index.d)
    shot_vector_count = int(shot_index.ntotal)
    del shot_index
    gc.collect()
    nearest_similarity, nearest_ids = compute_neighbors(
        vectors, np.asarray(ordered_faiss_ids, dtype=np.int64), config.batch_size, config.seed
    )
    signatures = visual_signatures(vectors, config.seed)
    faiss_id_to_row = {faiss_id: row for row, faiss_id in enumerate(ordered_faiss_ids)}

    video_lookup = shot_lookup_by_video(shots)
    print("[3/7] Aggregating clip and frame visual-change signals", flush=True)
    clip_index = read_index(config.faiss_dir, "clip", shot_dimension)
    clip_vector_count = int(clip_index.ntotal)
    clip_counts, clip_changes = clip_change_by_shot(manager, config, clip_index, config.batch_size)
    del clip_index
    gc.collect()
    frame_index = read_index(config.faiss_dir, "frame", shot_dimension)
    frame_vector_count = int(frame_index.ntotal)
    frame_counts, frame_changes = frame_change_by_shot(
        manager,
        config,
        frame_index,
        video_lookup,
        set(shot_ids),
        config.batch_size,
    )
    del frame_index
    gc.collect()
    print("[4/7] Aggregating ASR and OCR coverage", flush=True)
    asr_coverage_by_id, asr_words_by_id = asr_by_shot(manager)
    ocr_frames_by_id, ocr_texts_by_id = ocr_by_shot(manager)

    count = len(shots)
    durations = np.asarray([shot.duration_ms / 1_000.0 for shot in shots], dtype=np.float32)
    clip_count = np.asarray([clip_counts.get(shot_id, 0) for shot_id in shot_ids], dtype=np.int32)
    frame_count = np.asarray([frame_counts.get(shot_id, 0) for shot_id in shot_ids], dtype=np.int32)
    clip_change = np.asarray([clip_changes.get(shot_id, 0.0) for shot_id in shot_ids], dtype=np.float32)
    frame_change = np.asarray([frame_changes.get(shot_id, 0.0) for shot_id in shot_ids], dtype=np.float32)
    visual_change_raw = np.maximum(clip_change, frame_change)
    valid_change = (clip_count >= 2) | (frame_count >= 2)
    visual_change = percentile_rank(visual_change_raw, valid_change)
    novelty = percentile_rank(1.0 - nearest_similarity, nearest_similarity >= 0.0)

    neighbour_contrast_raw = np.zeros(count, dtype=np.float32)
    rows_by_video: dict[str, list[int]] = {}
    for row, shot in enumerate(shots):
        rows_by_video.setdefault(shot.video_id, []).append(row)
    for rows in rows_by_video.values():
        rows.sort(key=lambda row: (shots[row].shot_index, shots[row].start_ms, shot_ids[row]))
        for position, row in enumerate(rows):
            similarities: list[float] = []
            if position > 0:
                similarities.append(float(np.dot(vectors[row], vectors[rows[position - 1]])))
            if position + 1 < len(rows):
                similarities.append(float(np.dot(vectors[row], vectors[rows[position + 1]])))
            if similarities:
                neighbour_contrast_raw[row] = max(0.0, 1.0 - max(similarities))
    neighbour_contrast = percentile_rank(neighbour_contrast_raw)
    duration_score = np.clip(np.log1p(durations) / math.log1p(60.0), 0.0, 1.0)

    asr_overlap = np.asarray([asr_coverage_by_id.get(shot_id, 0.0) for shot_id in shot_ids], dtype=np.float32)
    asr_words = np.asarray([asr_words_by_id.get(shot_id, 0) for shot_id in shot_ids], dtype=np.int32)
    ocr_frames = np.asarray([ocr_frames_by_id.get(shot_id, 0) for shot_id in shot_ids], dtype=np.int32)
    ocr_texts = np.asarray([ocr_texts_by_id.get(shot_id, 0) for shot_id in shot_ids], dtype=np.int32)
    text_coverage = np.clip(
        0.65 * asr_overlap
        + 0.15 * np.minimum(asr_words / 30.0, 1.0)
        + 0.10 * np.minimum(ocr_frames / 3.0, 1.0)
        + 0.10 * np.minimum(ocr_texts / 8.0, 1.0),
        0.0,
        1.0,
    )
    base_score = (
        0.35 * visual_change
        + 0.30 * novelty
        + 0.20 * neighbour_contrast
        + 0.15 * duration_score
    )
    static_and_redundant = (visual_change < 0.35) & (novelty < 0.35)
    selection_score = base_score - 0.15 * text_coverage * static_and_redundant

    print("[5/7] Applying tier quotas and duplicate-aware ranking", flush=True)
    mandatory_rows = [row for row, duration in enumerate(durations) if duration >= 10.0]
    if len(mandatory_rows) > config.target:
        raise RuntimeError("Mandatory >=10 second tier exceeds the requested target.")
    medium_quota, short_quota, micro_quota = optional_quotas(config.target, len(mandatory_rows))
    medium_rows = [row for row, duration in enumerate(durations) if 5.0 <= duration < 10.0]
    short_rows = [row for row, duration in enumerate(durations) if 2.0 <= duration < 5.0]
    micro_rows = [row for row, duration in enumerate(durations) if duration < 2.0]
    for name, candidates, quota in (
        ("medium", medium_rows, medium_quota),
        ("short", short_rows, short_quota),
        ("micro", micro_rows, micro_quota),
    ):
        if len(candidates) < quota:
            raise RuntimeError(f"{name} tier has only {len(candidates)} candidates for quota {quota}.")

    selected_rows = set(mandatory_rows)
    selected_tier = np.full(count, "not_selected", dtype=object)
    selected_tier[mandatory_rows] = "A_long_ge_10s"
    medium_selected, medium_duplicates = select_ranked(
        medium_rows,
        medium_quota,
        selection_score,
        shot_ids,
        video_ids,
        nearest_ids,
        nearest_similarity,
        faiss_id_to_row,
        text_coverage,
        vectors,
        signatures,
        selected_rows,
    )
    selected_tier[medium_selected] = "B_medium_5_to_10s"
    short_selected, short_duplicates = select_ranked(
        short_rows,
        short_quota,
        selection_score,
        shot_ids,
        video_ids,
        nearest_ids,
        nearest_similarity,
        faiss_id_to_row,
        text_coverage,
        vectors,
        signatures,
        selected_rows,
    )
    selected_tier[short_selected] = "C_short_2_to_5s"
    micro_selected, micro_duplicates = select_ranked(
        micro_rows,
        micro_quota,
        selection_score,
        shot_ids,
        video_ids,
        nearest_ids,
        nearest_similarity,
        faiss_id_to_row,
        text_coverage,
        vectors,
        signatures,
        selected_rows,
    )
    selected_tier[micro_selected] = "D_micro_lt_2s"
    if len(selected_rows) != config.target:
        raise RuntimeError(f"Selection count mismatch: {len(selected_rows)} != {config.target}")

    duplicate_rows = medium_duplicates | short_duplicates | micro_duplicates
    selected_reason: list[str] = []
    for row in range(count):
        if row in mandatory_rows:
            selected_reason.append("duration_ge_10s")
        elif row in selected_rows:
            reasons = ["ranked_complex_visual"]
            if visual_change[row] >= 0.75:
                reasons.append("high_visual_change")
            if novelty[row] >= 0.75:
                reasons.append("high_visual_novelty")
            if text_coverage[row] < 0.30:
                reasons.append("low_text_coverage")
            selected_reason.append("|".join(reasons))
        elif row in duplicate_rows:
            selected_reason.append("near_duplicate_of_selected_shot")
        else:
            selected_reason.append("below_tier_quota")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("[6/7] Writing Kaggle CSV and audit report", flush=True)
    selected_path = config.output_dir / "shots_for_caption.csv"
    audit_path = config.output_dir / "caption_selection_audit.csv"
    summary_path = config.output_dir / "selection_summary.json"
    selected_output_rows = [
        {
            "shot_id": shot.shot_id,
            "video_id": shot.video_id,
            "shot_index": shot.shot_index,
            "start_ms": shot.start_ms,
            "end_ms": shot.end_ms,
            "start_frame_idx": shot.start_frame_idx,
            "end_frame_idx": shot.end_frame_idx,
        }
        for row, shot in enumerate(shots)
        if row in selected_rows
    ]
    atomic_write_csv(selected_path, list(SHOT_COLUMNS), selected_output_rows)

    audit_fields = [
        *SHOT_COLUMNS,
        "selected",
        "selection_tier",
        "selection_reason",
        "duration_sec",
        "shot_faiss_id",
        "selection_score",
        "visual_change_score",
        "visual_novelty_score",
        "neighbor_contrast_score",
        "nearest_similarity",
        "nearest_shot_id",
        "clip_count",
        "frame_count",
        "asr_overlap_ratio",
        "asr_word_count",
        "ocr_frame_count",
        "ocr_distinct_text_count",
        "text_coverage_score",
    ]
    audit_rows: list[dict[str, object]] = []
    for row, shot in enumerate(shots):
        nearest_row = faiss_id_to_row.get(int(nearest_ids[row]))
        audit_rows.append(
            {
                "shot_id": shot.shot_id,
                "video_id": shot.video_id,
                "shot_index": shot.shot_index,
                "start_ms": shot.start_ms,
                "end_ms": shot.end_ms,
                "start_frame_idx": shot.start_frame_idx,
                "end_frame_idx": shot.end_frame_idx,
                "selected": row in selected_rows,
                "selection_tier": selected_tier[row],
                "selection_reason": selected_reason[row],
                "duration_sec": round(float(durations[row]), 3),
                "shot_faiss_id": ordered_faiss_ids[row],
                "selection_score": round(float(selection_score[row]), 6),
                "visual_change_score": round(float(visual_change[row]), 6),
                "visual_novelty_score": round(float(novelty[row]), 6),
                "neighbor_contrast_score": round(float(neighbour_contrast[row]), 6),
                "nearest_similarity": round(float(nearest_similarity[row]), 6),
                "nearest_shot_id": shot_ids[nearest_row] if nearest_row is not None else "",
                "clip_count": int(clip_count[row]),
                "frame_count": int(frame_count[row]),
                "asr_overlap_ratio": round(float(asr_overlap[row]), 6),
                "asr_word_count": int(asr_words[row]),
                "ocr_frame_count": int(ocr_frames[row]),
                "ocr_distinct_text_count": int(ocr_texts[row]),
                "text_coverage_score": round(float(text_coverage[row]), 6),
            }
        )
    atomic_write_csv(audit_path, audit_fields, audit_rows)

    summary = {
        "target": config.target,
        "selected": len(selected_output_rows),
        "input_shots": count,
        "input_shots_sha256": source_sha256(config.shots_csv),
        "faiss": {
            "directory": str(config.faiss_dir),
            "dimension": shot_dimension,
            "shot_vectors": shot_vector_count,
            "clip_vectors": clip_vector_count,
            "frame_vectors": frame_vector_count,
            "index_version": config.index_version,
            "model_name": config.model_name,
            "model_version": config.model_version,
            "pooling_method": config.pooling_method,
        },
        "selection": {
            "seed": config.seed,
            "tier_counts": {
                "A_long_ge_10s": len(mandatory_rows),
                "B_medium_5_to_10s": len(medium_selected),
                "C_short_2_to_5s": len(short_selected),
                "D_micro_lt_2s": len(micro_selected),
            },
            "duplicate_candidates_skipped": len(duplicate_rows),
            "asr_covered_shots": int(np.count_nonzero(asr_overlap > 0)),
            "ocr_covered_shots": int(np.count_nonzero(ocr_frames > 0)),
        },
    }
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_summary.replace(summary_path)
    print("[7/7] Selection completed", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
