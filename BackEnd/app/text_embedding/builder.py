"""Build an immutable FlatIP text-index bundle."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from uuid import uuid4

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from BackEnd.app.embedding.artifacts.checksum import sha256_file
from BackEnd.app.text_embedding.contracts import TextDocument, TextIndexManifest
from BackEnd.app.text_embedding.encoder import TextEncoder


def build_text_index(
    documents: list[TextDocument],
    encoder: TextEncoder,
    output_root: Path,
    *,
    build_id: str,
    batch_size: int = 128,
) -> Path:
    """Encode documents and atomically write a FAISS index with row mappings."""

    if not documents:
        raise ValueError("At least one non-empty text document is required.")
    if not build_id.strip() or batch_size <= 0:
        raise ValueError("build_id must be non-empty and batch_size must be positive.")
    source_types = {document.source_type for document in documents}
    if len(source_types) != 1:
        raise ValueError("A text index may contain only one source_type.")
    if len({document.entity_id for document in documents}) != len(documents):
        raise ValueError("Text documents contain duplicate entity_id values.")

    vectors = np.asarray(
        encoder.encode([document.text for document in documents], batch_size=batch_size),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(documents):
        raise ValueError("Encoder output must have shape (document_count, dimension).")
    if vectors.shape[1] <= 0 or not np.isfinite(vectors).all():
        raise ValueError("Encoder output must contain finite, non-empty vectors.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Encoder returned a zero vector.")
    vectors = np.ascontiguousarray(vectors / norms[:, None], dtype=np.float32)

    source_type = source_types.pop()
    final_root = Path(output_root) / source_type / build_id
    temporary_root = final_root.with_name(f".{final_root.name}.{uuid4().hex}.tmp")
    if final_root.exists():
        raise FileExistsError(f"Text index already exists: {final_root}")

    try:
        temporary_root.mkdir(parents=True)
        index_path = temporary_root / "index.faiss"
        mapping_path = temporary_root / "mapping.parquet"

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        faiss_ids = np.arange(1, len(documents) + 1, dtype=np.int64)
        index.add_with_ids(vectors, faiss_ids)
        faiss.write_index(index, str(index_path))

        mapping_rows = [
            {"faiss_id": int(faiss_id), **asdict(document)}
            for faiss_id, document in zip(faiss_ids, documents, strict=True)
        ]
        pq.write_table(pa.Table.from_pylist(mapping_rows), mapping_path)

        checksums = {
            index_path.name: sha256_file(index_path),
            mapping_path.name: sha256_file(mapping_path),
        }
        manifest = TextIndexManifest(
            artifact_version="dense-text-index@1.0.0",
            build_id=build_id,
            source_type=source_type,
            model_id=encoder.model_id,
            model_revision=encoder.model_revision,
            dimension=int(vectors.shape[1]),
            normalized=True,
            record_count=len(documents),
            index_file=index_path.name,
            mapping_file=mapping_path.name,
            checksums=checksums,
            created_at=datetime.now(UTC).isoformat(),
        )
        (temporary_root / "manifest.json").write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_root.rename(final_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
    return final_root


def validate_text_index(artifact_root: Path) -> dict[str, object]:
    """Validate checksums and the index-to-mapping row count."""

    root = Path(artifact_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for filename, expected in manifest["checksums"].items():
        path = root / filename
        if not path.is_file():
            errors.append(f"Missing file: {filename}")
        elif sha256_file(path) != expected:
            errors.append(f"Checksum mismatch: {filename}")

    if not errors:
        index = faiss.read_index(str(root / manifest["index_file"]))
        mapping_table = pq.read_table(root / manifest["mapping_file"])
        mapping_count = mapping_table.num_rows
        if index.ntotal != manifest["record_count"] or mapping_count != index.ntotal:
            errors.append("Index, mapping, and manifest counts do not match.")
        if index.d != manifest["dimension"]:
            errors.append("Index dimension does not match the manifest.")
        if not isinstance(index, faiss.IndexIDMap2):
            errors.append("Index must be an IndexIDMap2.")
        else:
            index_ids = faiss.vector_to_array(index.id_map).tolist()
            mapping_ids = mapping_table.column("faiss_id").to_pylist()
            if index_ids != mapping_ids:
                errors.append("FAISS IDs do not match mapping rows.")
    return {"valid": not errors, "errors": errors, "manifest": manifest}
