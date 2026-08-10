from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from BackEnd.app.object_detection.convert_aic25_objects_to_jsonl import convert_directory


class ConvertAIC25ObjectsToJsonlTests(unittest.TestCase):
    def test_converts_parallel_arrays_to_filtered_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_dir = temporary_path / "input"
            source_path = input_dir / "L21_V014" / "013.json"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(
                json.dumps(
                    {
                        "detection_scores": ["0.90", "0.24"],
                        "detection_class_names": ["/m/person", "/m/car"],
                        "detection_class_entities": ["Person", "Car"],
                        "detection_boxes": [
                            ["0.1", "0.2", "0.3", "0.4"],
                            ["0.5", "0.6", "0.7", "0.8"],
                        ],
                        "detection_class_labels": ["69", "571"],
                    }
                ),
                encoding="utf-8",
            )
            output_dir = temporary_path / "output"

            files, records = convert_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                confidence_threshold=0.25,
                overwrite=False,
            )

            self.assertEqual((files, records), (1, 1))
            output_path = output_dir / "L21_V014" / "013.jsonl"
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(
                json.loads(lines[0]),
                {
                    "video_id": "L21_V014",
                    "keyframe_index": 13,
                    "confidence": 0.9,
                    "class_mid": "/m/person",
                    "class_name": "Person",
                    "class_label": 69,
                    "bbox": {
                        "y_min": 0.1,
                        "x_min": 0.2,
                        "y_max": 0.3,
                        "x_max": 0.4,
                    },
                },
            )

    def test_rejects_mismatched_parallel_array_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_dir = temporary_path / "input"
            source_path = input_dir / "L21_V014" / "013.json"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(
                json.dumps(
                    {
                        "detection_scores": ["0.90"],
                        "detection_class_names": [],
                        "detection_class_entities": ["Person"],
                        "detection_boxes": [["0.1", "0.2", "0.3", "0.4"]],
                        "detection_class_labels": ["69"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
                convert_directory(
                    input_dir=input_dir,
                    output_dir=temporary_path / "output",
                    confidence_threshold=0.25,
                    overwrite=False,
                )
