"""Unit tests for the object detection module.

Run from project root:
    python3 -m unittest BackEnd.tests.object_detection.test_object_detection

This file intentionally avoids real OpenCV/YOLO inference. It guards the stable
contracts that can silently corrupt downstream DB/tracking data when changed:
bbox normalization, cNNN class ids, filtering, NMS, and ObjectDetectionResult.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

from BackEnd.app.contracts.pipeline import ClassMetadata, ObjectDetectionResult
from BackEnd.app.object_detection.class_mapper import _MAX_CLASS_INDEX, ClassMapper
from BackEnd.app.object_detection.exporter import detections_to_contracts
from BackEnd.app.object_detection.postprocess import (
    filter_by_class,
    filter_by_confidence,
    iou,
    nms,
)
from BackEnd.app.object_detection.schemas import BoundingBox, Detection


def make_detection(
    *,
    class_index: int = 0,
    class_id: str = "c000",
    class_name: str = "person",
    confidence: float = 0.9,
    x_min: float = 100,
    y_min: float = 50,
    x_max: float = 200,
    y_max: float = 150,
    frame_id: str | None = "frame_001",
) -> Detection:
    return Detection(
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        confidence=confidence,
        class_index=class_index,
        class_id=class_id,
        class_name=class_name,
        frame_id=frame_id,
        model_name="YOLO",
        model_version="yolov8n.pt",
    )


class TestBoundingBoxNormalized(unittest.TestCase):
    """Guard the bbox order footgun between internal xyxy and DB contract fields."""

    def test_normalized_returns_bounding_box(self) -> None:
        result = BoundingBox(0, 0, 100, 100).normalized(200, 200)
        self.assertIsInstance(result, BoundingBox)

    def test_field_order_is_xmin_xmax_ymin_ymax(self) -> None:
        bbox = BoundingBox(x_min=100, y_min=50, x_max=200, y_max=150)
        result = detections_to_contracts(
            [make_detection(x_min=100, y_min=50, x_max=200, y_max=150)],
            image_width=400,
            image_height=300,
        )[0]

        self.assertEqual(
            bbox.to_xyxy(),
            [100, 50, 200, 150],
            "Internal bbox order must remain [x_min, y_min, x_max, y_max].",
        )
        self.assertAlmostEqual(result.x_min, 100 / 400, msg="Position 0 must be x_min.")
        self.assertAlmostEqual(result.x_max, 200 / 400, msg="Position 1 must be x_max, not y_min.")
        self.assertAlmostEqual(result.y_min, 50 / 300, msg="Position 2 must be y_min, not x_max.")
        self.assertAlmostEqual(result.y_max, 150 / 300, msg="Position 3 must be y_max.")

    def test_normalized_values_within_unit_range(self) -> None:
        bbox = BoundingBox(0, 0, 640, 480).normalized(640, 480)
        for value in (bbox.x_min, bbox.x_max, bbox.y_min, bbox.y_max):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_normalized_x_max_greater_than_x_min(self) -> None:
        bbox = BoundingBox(10, 20, 300, 400).normalized(640, 480)
        self.assertGreater(bbox.x_max, bbox.x_min)

    def test_normalized_y_max_greater_than_y_min(self) -> None:
        bbox = BoundingBox(10, 20, 300, 400).normalized(640, 480)
        self.assertGreater(bbox.y_max, bbox.y_min)

    def test_normalized_raises_on_zero_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            BoundingBox(0, 0, 100, 100).normalized(0, 100)

    def test_normalized_raises_on_zero_height(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            BoundingBox(0, 0, 100, 100).normalized(100, 0)

    def test_objectdetectionresult_field_order_in_dataclass(self) -> None:
        field_names = [field.name for field in fields(ObjectDetectionResult)]
        self.assertLess(field_names.index("x_min"), field_names.index("x_max"))
        self.assertLess(field_names.index("x_max"), field_names.index("y_min"))
        self.assertLess(field_names.index("y_min"), field_names.index("y_max"))


class TestClassMapperFormat(unittest.TestCase):
    """Cover the full cNNN class id boundary."""

    def test_person_is_c000(self) -> None:
        self.assertEqual(ClassMapper.class_id_for_index(0), "c000")

    def test_car_is_c002(self) -> None:
        self.assertEqual(ClassMapper.class_id_for_index(2), "c002")

    def test_single_digit_is_zero_padded(self) -> None:
        self.assertEqual(ClassMapper.class_id_for_index(7), "c007")

    def test_double_digit_is_zero_padded(self) -> None:
        self.assertEqual(ClassMapper.class_id_for_index(42), "c042")

    def test_triple_digit(self) -> None:
        self.assertEqual(ClassMapper.class_id_for_index(999), "c999")

    def test_always_starts_with_c(self) -> None:
        for index in (0, 1, 9, 10, 99, 100, 999):
            self.assertTrue(ClassMapper.class_id_for_index(index).startswith("c"))

    def test_always_four_chars_total(self) -> None:
        for index in (0, 1, 9, 10, 99, 100, 999):
            self.assertEqual(len(ClassMapper.class_id_for_index(index)), 4)

    def test_negative_index_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ClassMapper.class_id_for_index(-1)

    def test_over_limit_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, str(_MAX_CLASS_INDEX)):
            ClassMapper.class_id_for_index(1000)

    def test_mapper_warns_on_large_class_set(self) -> None:
        big_names = {index: f"class_{index}" for index in range(1001)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ClassMapper(big_names)
        self.assertTrue(any("cNNN limit" in str(item.message) for item in caught))

    def test_coco_fallback_when_no_names_given(self) -> None:
        self.assertEqual(ClassMapper().class_name_for_index(0), "person")

    def test_class_name_from_index(self) -> None:
        mapper = ClassMapper({0: "person", 1: "bicycle", 2: "car"})
        self.assertEqual(mapper.class_name_for_index(1), "bicycle")

    def test_unknown_index_raises_key_error(self) -> None:
        mapper = ClassMapper({0: "person"})
        with self.assertRaises(KeyError):
            mapper.class_name_for_index(99)

    def test_to_class_metadata_class_id_and_name_match(self) -> None:
        mapper = ClassMapper({0: "person", 2: "car"})
        metadata = mapper.to_class_metadata()
        self.assertTrue(all(isinstance(item, ClassMetadata) for item in metadata))
        self.assertEqual(
            {item.class_id: item.class_name for item in metadata},
            {"c000": "person", "c002": "car"},
        )


class TestConfidenceFilter(unittest.TestCase):
    def test_keeps_detection_at_threshold(self) -> None:
        self.assertEqual(len(filter_by_confidence([make_detection(confidence=0.5)], threshold=0.5)), 1)

    def test_removes_detection_below_threshold(self) -> None:
        self.assertEqual(filter_by_confidence([make_detection(confidence=0.49)], threshold=0.5), [])

    def test_keeps_high_confidence(self) -> None:
        self.assertEqual(len(filter_by_confidence([make_detection(confidence=0.99)], threshold=0.5)), 1)

    def test_empty_input(self) -> None:
        self.assertEqual(filter_by_confidence([], threshold=0.5), [])

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            filter_by_confidence([], threshold=1.5)

    def test_mixed_batch(self) -> None:
        detections = [
            make_detection(confidence=0.8),
            make_detection(confidence=0.3),
            make_detection(confidence=0.6),
        ]
        result = filter_by_confidence(detections, threshold=0.5)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(detection.confidence >= 0.5 for detection in result))


class TestClassFilter(unittest.TestCase):
    def test_keeps_matching_class(self) -> None:
        self.assertEqual(len(filter_by_class([make_detection(class_id="c000")], class_ids={"c000"})), 1)

    def test_removes_non_matching_class(self) -> None:
        self.assertEqual(filter_by_class([make_detection(class_id="c002")], class_ids={"c000"}), [])

    def test_multi_class_filter(self) -> None:
        detections = [
            make_detection(class_id="c000"),
            make_detection(class_id="c002", class_index=2, class_name="car"),
            make_detection(class_id="c016", class_index=16, class_name="dog"),
        ]
        result = filter_by_class(detections, class_ids={"c000", "c016"})
        self.assertEqual([detection.class_id for detection in result], ["c000", "c016"])

    def test_empty_class_set_returns_nothing(self) -> None:
        self.assertEqual(filter_by_class([make_detection(class_id="c000")], class_ids=set()), [])

    def test_empty_detections(self) -> None:
        self.assertEqual(filter_by_class([], class_ids={"c000"}), [])


class TestIoU(unittest.TestCase):
    def test_identical_boxes_iou_is_one(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        self.assertAlmostEqual(iou(box, box), 1.0)

    def test_non_overlapping_boxes_iou_is_zero(self) -> None:
        self.assertAlmostEqual(iou(BoundingBox(0, 0, 50, 50), BoundingBox(100, 100, 200, 200)), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(iou(BoundingBox(0, 0, 100, 100), BoundingBox(50, 50, 150, 150)), 2500 / 17500)

    def test_contained_box(self) -> None:
        self.assertAlmostEqual(iou(BoundingBox(0, 0, 100, 100), BoundingBox(25, 25, 75, 75)), 2500 / 10000)

    def test_touching_edges_no_overlap(self) -> None:
        self.assertAlmostEqual(iou(BoundingBox(0, 0, 50, 100), BoundingBox(50, 0, 100, 100)), 0.0)


class TestNMS(unittest.TestCase):
    def test_single_detection_kept(self) -> None:
        detection = make_detection()
        self.assertEqual(nms([detection]), [detection])

    def test_no_suppression_when_no_overlap(self) -> None:
        first = make_detection(confidence=0.9, x_min=0, y_min=0, x_max=50, y_max=50)
        second = make_detection(confidence=0.8, x_min=200, y_min=200, x_max=300, y_max=300)
        self.assertEqual(len(nms([first, second], iou_threshold=0.5)), 2)

    def test_lower_confidence_suppressed_when_overlapping(self) -> None:
        high = make_detection(confidence=0.9, x_min=0, y_min=0, x_max=100, y_max=100)
        low = make_detection(confidence=0.5, x_min=0, y_min=0, x_max=100, y_max=100)
        self.assertEqual(nms([high, low], iou_threshold=0.5), [high])

    def test_different_classes_not_suppressed(self) -> None:
        person = make_detection(class_id="c000", class_index=0, class_name="person")
        dog = make_detection(class_id="c016", class_index=16, class_name="dog", confidence=0.85)
        self.assertEqual(len(nms([person, dog], iou_threshold=0.5)), 2)

    def test_empty_input(self) -> None:
        self.assertEqual(nms([]), [])

    def test_output_sorted_by_confidence_descending(self) -> None:
        detections = [
            make_detection(confidence=0.6, x_min=0, y_min=0, x_max=10, y_max=10),
            make_detection(confidence=0.9, x_min=100, y_min=100, x_max=110, y_max=110),
            make_detection(confidence=0.75, x_min=200, y_min=200, x_max=210, y_max=210),
        ]
        result = nms(detections, iou_threshold=0.5)
        confidences = [detection.confidence for detection in result]
        self.assertEqual(confidences, sorted(confidences, reverse=True))


class TestObjectDetectionResultContract(unittest.TestCase):
    def test_required_fields_present(self) -> None:
        field_names = {field.name for field in fields(ObjectDetectionResult)}
        required = {"frame_id", "class_id", "confidence", "x_min", "x_max", "y_min", "y_max"}
        self.assertTrue(required.issubset(field_names))

    def test_class_name_not_in_contract(self) -> None:
        field_names = {field.name for field in fields(ObjectDetectionResult)}
        self.assertNotIn(
            "class_name",
            field_names,
            "class_name should stay in ClassMetadata; downstream modules join by class_id.",
        )

    def test_timestamp_ms_not_in_contract(self) -> None:
        field_names = {field.name for field in fields(ObjectDetectionResult)}
        self.assertNotIn(
            "timestamp_ms",
            field_names,
            "timestamp_ms should stay in FrameMetadata; tracking joins by frame_id.",
        )

    def test_class_id_type_is_str(self) -> None:
        field_map = {field.name: field for field in fields(ObjectDetectionResult)}
        self.assertIn(field_map["class_id"].type, (str, "str"))

    def test_frozen_dataclass(self) -> None:
        result = ObjectDetectionResult(
            frame_id="f1",
            class_id="c000",
            confidence=0.9,
            x_min=0.1,
            x_max=0.5,
            y_min=0.2,
            y_max=0.8,
        )
        with self.assertRaises(FrozenInstanceError):
            result.confidence = 0.0  # type: ignore[misc]

    def test_full_construction_with_optional_fields(self) -> None:
        result = ObjectDetectionResult(
            frame_id="frame_042",
            class_id="c000",
            confidence=0.95,
            x_min=0.1,
            x_max=0.5,
            y_min=0.2,
            y_max=0.8,
            model_name="YOLO",
            model_version="yolov8n.pt",
            detection_id=1,
        )
        self.assertEqual(result.model_name, "YOLO")
        self.assertEqual(result.model_version, "yolov8n.pt")
        self.assertEqual(result.detection_id, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
