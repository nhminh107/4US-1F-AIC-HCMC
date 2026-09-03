"""Build Context or ASR dense text indexes."""

from __future__ import annotations

import argparse
from pathlib import Path

from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.text_embedding.builder import build_text_index, validate_text_index
from BackEnd.app.text_embedding.encoder import SentenceTransformerTextEncoder
from BackEnd.app.text_embedding.sources import asr_documents, context_documents


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dense text FAISS index.")
    parser.add_argument("source", choices=("context", "asr"))
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/text_indexes"))
    parser.add_argument("--model-id", default="BAAI/bge-m3")
    parser.add_argument("--model-revision")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-artifact", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manager = None
    if args.source == "context":
        if args.context_artifact is None:
            raise SystemExit("--context-artifact is required for source=context")
        documents = context_documents(args.context_artifact)
    else:
        manager = PostgreManager(database_url=args.database_url)
        documents = asr_documents(manager.engine, video_ids=args.video_ids)

    try:
        encoder = SentenceTransformerTextEncoder(
            args.model_id,
            model_revision=args.model_revision,
            device=args.device,
        )
        artifact_root = build_text_index(
            documents,
            encoder,
            args.output_root,
            build_id=args.build_id,
            batch_size=args.batch_size,
        )
        validation = validate_text_index(artifact_root)
        if not validation["valid"]:
            raise RuntimeError(f"Text index validation failed: {validation['errors']}")
    finally:
        if manager is not None:
            manager.engine.dispose()

    print(f"Wrote and validated {len(documents)} documents at {artifact_root}")


if __name__ == "__main__":
    main()
