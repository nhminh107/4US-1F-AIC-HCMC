from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from BackEnd.app.contracts.vqa import VQAEvidence, VQARequest
from BackEnd.app.service.vqa_handler import VQAHandler


class _VQAClientStub:
    model_name = "test-vlm"
    model_version = "1"

    def __init__(self) -> None:
        self.image_paths: tuple[Path, ...] = ()
        self.prompt = ""

    def answer(self, *, question: str, image_paths, prompt: str) -> str:
        self.image_paths = tuple(image_paths)
        self.prompt = prompt
        return "A red motorbike is visible."


class VQAHandlerTest(unittest.TestCase):
    def test_deduplicates_evidence_and_preserves_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.jpg"
            image.touch()
            evidence = VQAEvidence(
                image_path=str(image),
                video_id="L01_V001",
                timestamp_ms=1000,
                frame_id="F001",
            )
            client = _VQAClientStub()
            handler = VQAHandler(client)

            response = handler.answer(
                VQARequest(question="What vehicle is visible?", evidence=(evidence, evidence))
            )

            self.assertEqual(response.answer, "A red motorbike is visible.")
            self.assertEqual(response.evidence, (evidence,))
            self.assertEqual(client.image_paths, (image,))
            self.assertIn("using only visible evidence", client.prompt)

    def test_rejects_missing_evidence_image(self) -> None:
        handler = VQAHandler(_VQAClientStub())
        request = VQARequest(
            question="What is visible?",
            evidence=(
                VQAEvidence(
                    image_path="missing.jpg",
                    video_id="L01_V001",
                    timestamp_ms=0,
                ),
            ),
        )

        with self.assertRaises(FileNotFoundError):
            handler.answer(request)


if __name__ == "__main__":
    unittest.main()
