"""Read and write immutable FrameContext artifacts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from BackEnd.app.embedding.artifacts.checksum import sha256_file
from BackEnd.app.frame_context.contracts import FrameContextRecord


def write_frame_context_artifact(
    records: list[FrameContextRecord],
    output_root: Path,
    *,
    build_id: str,
) -> Path:
    """Atomically write records and a checksum manifest into a new build directory."""

    if not build_id.strip():
        raise ValueError("build_id must not be empty.")
    if len({record.frame_id for record in records}) != len(records):
        raise ValueError("FrameContext records contain duplicate frame_id values.")

    final_root = Path(output_root) / build_id
    temporary_root = final_root.with_name(f".{final_root.name}.{uuid4().hex}.tmp")
    if final_root.exists():
        raise FileExistsError(f"FrameContext artifact already exists: {final_root}")

    try:
        temporary_root.mkdir(parents=True)
        data_path = temporary_root / "frame_context_v1.parquet"
        rows = [asdict(record) for record in records]
        pq.write_table(pa.Table.from_pylist(rows), data_path)
        manifest = {
            "artifact_version": "frame-context-artifact@1.0.0",
            "build_id": build_id,
            "schema_version": records[0].schema_version if records else "frame-context@1.0.0",
            "record_count": len(records),
            "searchable_record_count": sum(bool(record.context_text) for record in records),
            "empty_record_count": sum(not record.context_text for record in records),
            "data_file": data_path.name,
            "checksums": {data_path.name: sha256_file(data_path)},
            "created_at": datetime.now(UTC).isoformat(),
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_root.rename(final_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
    return final_root


def read_frame_context_artifact(artifact_root: Path) -> list[FrameContextRecord]:
    """Load and checksum-validate a FrameContext artifact."""

    root = Path(artifact_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data_path = root / manifest["data_file"]
    expected_checksum = manifest["checksums"][manifest["data_file"]]
    if sha256_file(data_path) != expected_checksum:
        raise ValueError(f"Checksum mismatch: {data_path}")
    rows = pq.read_table(data_path).to_pylist()
    if len(rows) != manifest["record_count"]:
        raise ValueError("FrameContext row count does not match its manifest.")
    return [FrameContextRecord(**row) for row in rows]
