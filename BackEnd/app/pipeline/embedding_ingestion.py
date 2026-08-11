"""Ingest organizer keyframe CLIP features into the shared FAISS index."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from BackEnd.CONFIG import (
    CLIP_DIMENSION,
    CLIP_MODEL,
    KEYFRAME_MAP_DIR,
    ORGANIZER_CLIP_FEATURE_DIR,
)
from BackEnd.app.contracts.pipeline import FrameEmbeddingMapping
from BackEnd.app.database.faiss_db import FAISS_Manager
from BackEnd.app.database.postgre_db import PostgreManager


@dataclass(frozen=True, slots=True)
class _FrameReference:
    """Minimal source identity required by ``FAISS_Manager.add_imgs``."""

    frame_id: str


def ingest_organizer_frame_embeddings(
    *,
    db: PostgreManager,
    faiss_manager: FAISS_Manager,
    feature_dir: Path = ORGANIZER_CLIP_FEATURE_DIR,
    map_dir: Path = KEYFRAME_MAP_DIR,
    batch_size: int = 8_192,
    video_ids: Iterable[str] | None = None,
) -> int:
    """Load official keyframe vectors and persist FAISS/DB mappings.

    Each matrix row ``i`` maps to organizer frame ``<video_id>_{i + 1:03d}``.
    The CSV map is validated before insertion so a malformed ordering cannot
    silently attach a vector to the wrong frame.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    inserted = 0
    for vectors, frame_references in iter_organizer_embedding_batches(
        feature_dir=feature_dir,
        map_dir=map_dir,
        batch_size=batch_size,
        video_ids=video_ids,
    ):
        inserted += _persist_embedding_batch(
            db=db,
            faiss_manager=faiss_manager,
            vectors=vectors,
            frame_references=frame_references,
        )
    return inserted


def iter_organizer_embedding_batches(
    *,
    feature_dir: Path,
    map_dir: Path,
    batch_size: int,
    video_ids: Iterable[str] | None = None,
) -> Iterator[tuple[np.ndarray, list[_FrameReference]]]:
    """Stream float32 vector batches without loading all 873 matrices into RAM."""

    resolved_feature_dir = Path(feature_dir).expanduser().resolve()
    resolved_map_dir = Path(map_dir).expanduser().resolve()
    if not resolved_feature_dir.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {resolved_feature_dir}")
    if not resolved_map_dir.is_dir():
        raise FileNotFoundError(f"Keyframe map directory does not exist: {resolved_map_dir}")

    selected_video_ids = set(video_ids) if video_ids is not None else None
    for feature_path in sorted(resolved_feature_dir.glob("*.npy")):
        video_id = feature_path.stem
        if selected_video_ids is not None and video_id not in selected_video_ids:
            continue

        matrix = np.load(feature_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[1] != CLIP_DIMENSION:
            raise ValueError(
                f"Feature matrix must have shape (K, {CLIP_DIMENSION}): "
                f"{feature_path} has {matrix.shape}."
            )
        if not np.issubdtype(matrix.dtype, np.floating):
            raise ValueError(f"Feature matrix must use a floating dtype: {feature_path}")
        frame_references = _load_frame_references(
            video_id=video_id,
            map_path=resolved_map_dir / f"{video_id}.csv",
            row_count=matrix.shape[0],
        )

        for start in range(0, len(frame_references), batch_size):
            stop = min(start + batch_size, len(frame_references))
            vectors = np.asarray(matrix[start:stop], dtype=np.float32)
            yield vectors, frame_references[start:stop]


def _load_frame_references(
    *,
    video_id: str,
    map_path: Path,
    row_count: int,
) -> list[_FrameReference]:
    if not map_path.is_file():
        raise FileNotFoundError(f"Missing keyframe map for {video_id}: {map_path}")

    with map_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != row_count:
        raise ValueError(
            f"Feature/map row count mismatch for {video_id}: "
            f"features={row_count}, map={len(rows)}."
        )

    references: list[_FrameReference] = []
    for row_index, row in enumerate(rows):
        try:
            keyframe_number = int(row["n"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid n at {map_path}, row {row_index + 2}.") from exc
        expected_number = row_index + 1
        if keyframe_number != expected_number:
            raise ValueError(
                f"Expected n={expected_number} at {map_path}, row {row_index + 2}; "
                f"got n={keyframe_number}."
            )
        references.append(_FrameReference(f"{video_id}_{keyframe_number:03d}"))
    return references


def _persist_embedding_batch(
    *,
    db: PostgreManager,
    faiss_manager: FAISS_Manager,
    vectors: np.ndarray,
    frame_references: list[_FrameReference],
) -> int:
    frame_ids = [reference.frame_id for reference in frame_references]
    existing = db.get_frame_embedding_mappings(
        frame_ids,
        index_version=faiss_manager.version,
        model_name=faiss_manager.model_name,
    )
    faiss_manager.validate_ids("frame", [mapping.faiss_id for mapping in existing])
    existing_frame_ids = {mapping.frame_id for mapping in existing}
    pending_positions = [
        index
        for index, frame_id in enumerate(frame_ids)
        if frame_id not in existing_frame_ids
    ]
    if not pending_positions:
        return 0

    pending_vectors = vectors[pending_positions]
    pending_references = [frame_references[index] for index in pending_positions]
    mappings, _, _ = faiss_manager.add_and_save(
        imgs=pending_vectors,
        imgs_model=pending_references,
    )
    try:
        db.add_frame_embedding_records(mappings)
    except Exception:
        faiss_manager.rollback(frame_mappings=mappings)
        raise
    return len(mappings)


def build_shared_faiss_manager(
    *,
    data_path: Path | None = None,
    index_version: int = 0,
) -> FAISS_Manager:
    """Create the shared frame index used by organizer and extracted frames."""

    return FAISS_Manager(
        img_dim=CLIP_DIMENSION,
        clip_dim=CLIP_DIMENSION,
        shot_dim=CLIP_DIMENSION,
        version=index_version,
        model_name=CLIP_MODEL,
        data_path=data_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest organizer clip-features-32 into the frame FAISS index."
    )
    parser.add_argument("--feature-dir", type=Path, default=ORGANIZER_CLIP_FEATURE_DIR)
    parser.add_argument("--map-dir", type=Path, default=KEYFRAME_MAP_DIR)
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--index-version", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    arguments = parser.parse_args()

    faiss_manager = build_shared_faiss_manager(
        data_path=arguments.faiss_dir,
        index_version=arguments.index_version,
    )
    inserted = ingest_organizer_frame_embeddings(
        db=PostgreManager(),
        faiss_manager=faiss_manager,
        feature_dir=arguments.feature_dir,
        map_dir=arguments.map_dir,
        batch_size=arguments.batch_size,
        video_ids=arguments.video_ids,
    )
    print(f"Inserted {inserted} organizer frame embedding mappings.")


__all__ = [
    "build_shared_faiss_manager",
    "ingest_organizer_frame_embeddings",
    "iter_organizer_embedding_batches",
]


if __name__ == "__main__":
    main()
