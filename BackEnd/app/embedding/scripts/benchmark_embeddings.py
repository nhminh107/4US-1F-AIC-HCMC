"""Smoke benchmark for embedding artifacts without building FAISS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from BackEnd.app.embedding.artifacts.reader import load_vectors
from BackEnd.app.embedding.artifacts.validator import load_manifest, validate_embedding_artifact


def summarize_artifact(manifest_path: Path) -> dict[str, object]:
    """Load an artifact and return lightweight quality/system stats."""

    manifest = load_manifest(manifest_path)
    artifact_root = manifest_path.parent
    validation = validate_embedding_artifact(manifest, artifact_root)
    vectors = load_vectors(artifact_root, manifest)
    norms = np.linalg.norm(vectors, axis=1) if vectors.size else np.array([])
    return {
        "valid": validation["valid"],
        "errors": validation["errors"],
        "entity_type": manifest.entity_type.value,
        "embedding_space_id": manifest.embedding_space_id,
        "success_count": manifest.success_count,
        "failure_count": manifest.failure_count,
        "dimension": manifest.dimension,
        "mean_norm": None if norms.size == 0 else float(norms.mean()),
        "min_norm": None if norms.size == 0 else float(norms.min()),
        "max_norm": None if norms.size == 0 else float(norms.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a clip/shot embedding artifact.")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    summary = summarize_artifact(args.manifest)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

