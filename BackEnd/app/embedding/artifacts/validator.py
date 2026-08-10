"""Validate embedding artifact manifests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from BackEnd.app.contracts.embedding import EmbeddingArtifactManifest
from BackEnd.CONFIG import EntityType
from BackEnd.app.embedding.artifacts.checksum import sha256_file


def validate_embedding_artifact(
    manifest: EmbeddingArtifactManifest,
    artifact_root: Path,
) -> dict[str, object]:
    """Validate shard checksums and vector row counts."""

    errors: list[str] = []
    for relative_path, checksum in manifest.checksums.items():
        path = artifact_root / relative_path
        if not path.is_file():
            errors.append(f"Missing artifact file: {relative_path}")
            continue
        actual = sha256_file(path)
        if actual != checksum:
            errors.append(f"Checksum mismatch for {relative_path}")

    vector_count = 0
    for shard in manifest.vector_shards:
        array = np.load(artifact_root / shard, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != manifest.dimension:
            errors.append(f"Invalid vector shape for {shard}: {array.shape}")
        vector_count += int(array.shape[0])
    if vector_count != manifest.success_count:
        errors.append(
            f"Vector rows {vector_count} != success_count {manifest.success_count}"
        )

    return {
        "valid": not errors,
        "errors": errors,
        "manifest": asdict(manifest),
    }


def load_manifest(path: Path) -> EmbeddingArtifactManifest:
    """Load a manifest JSON file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("entity_type"), EntityType):
        data["entity_type"] = EntityType(data["entity_type"])
    for field_name in ("vector_shards", "metadata_shards", "failure_shards"):
        data[field_name] = tuple(data.get(field_name) or ())
    return EmbeddingArtifactManifest(**data)
