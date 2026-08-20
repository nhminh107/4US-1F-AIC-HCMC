"""VQA orchestration over already-retrieved visual evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from BackEnd.app.contracts.vqa import (
    VQAEvidence,
    VQARequest,
    VQAResponse,
)


class VQAModelClient(Protocol):
    """Minimal interface implemented by the selected vision-language model."""

    model_name: str
    model_version: str

    def answer(self, *, question: str, image_paths: Sequence[Path], prompt: str) -> str:
        """Answer one question using only the supplied images."""


class VQAHandler:
    """Validate, deduplicate, and submit bounded retrieval evidence to a VLM."""

    prompt_version = "vqa-grounded-v1"

    def __init__(self, client: VQAModelClient) -> None:
        self.client = client

    def answer(self, request: VQARequest) -> VQAResponse:
        selected = self._select_evidence(request.evidence, request.max_evidence)
        image_paths = tuple(Path(item.image_path) for item in selected)
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "VQA evidence images do not exist: " + ", ".join(missing)
            )

        prompt = self._build_prompt(request.question, selected)
        answer = self.client.answer(
            question=request.question,
            image_paths=image_paths,
            prompt=prompt,
        )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("VQA model returned an empty answer.")

        return VQAResponse(
            answer=answer.strip(),
            evidence=selected,
            model_name=self.client.model_name,
            model_version=self.client.model_version,
            prompt_version=self.prompt_version,
        )

    @staticmethod
    def _select_evidence(
        evidence: tuple[VQAEvidence, ...],
        limit: int,
    ) -> tuple[VQAEvidence, ...]:
        selected: list[VQAEvidence] = []
        seen: set[tuple[str, int]] = set()
        for item in evidence:
            identity = (item.video_id, item.timestamp_ms)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(item)
            if len(selected) == limit:
                break
        return tuple(selected)

    @staticmethod
    def _build_prompt(question: str, evidence: tuple[VQAEvidence, ...]) -> str:
        manifest = "\n".join(
            f"Image {index}: video_id={item.video_id}, "
            f"timestamp_ms={item.timestamp_ms}, caption={item.caption or 'none'}"
            for index, item in enumerate(evidence, start=1)
        )
        return (
            "Answer using only visible evidence in the supplied images. "
            "Do not infer identities, actions, or events that are not visible. "
            "If the evidence is insufficient, say so explicitly.\n"
            f"Question: {question}\nEvidence manifest:\n{manifest}"
        )


__all__ = ["VQAHandler", "VQAModelClient"]
