"""Class id and class name mapping for object detection."""

from __future__ import annotations

import warnings

from BackEnd.app.contracts.pipeline import ClassMetadata

_MAX_CLASS_INDEX = 999

COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _normalize_name(class_name: str) -> str:
    return " ".join(class_name.strip().lower().replace("_", " ").split())


class ClassMapper:
    """Bidirectional mapping between model class indices, ids, and names."""

    def __init__(self, names: dict[int, str] | list[str] | tuple[str, ...] | None = None):
        if names is None:
            names = COCO_CLASSES
        if isinstance(names, dict):
            self._index_to_name = {int(index): str(name) for index, name in names.items()}
        else:
            self._index_to_name = {index: str(name) for index, name in enumerate(names)}

        if self._index_to_name and max(self._index_to_name) > _MAX_CLASS_INDEX:
            warnings.warn(
                f"Class index exceeds cNNN limit ({_MAX_CLASS_INDEX}). "
                "Use a wider class_id strategy before persisting these classes.",
                UserWarning,
                stacklevel=2,
            )

        self._name_to_index = {
            _normalize_name(name): index for index, name in self._index_to_name.items()
        }

    @staticmethod
    def class_id_for_index(class_index: int) -> str:
        resolved_index = int(class_index)
        if resolved_index < 0:
            raise ValueError("class_index must be non-negative.")
        if resolved_index > _MAX_CLASS_INDEX:
            raise ValueError(f"class_index must be <= {_MAX_CLASS_INDEX}.")
        return f"c{resolved_index:03d}"

    def name_for_index(self, class_index: int) -> str:
        return self._index_to_name.get(int(class_index), f"class_{int(class_index)}")

    def class_name_for_index(self, class_index: int) -> str:
        resolved_index = int(class_index)
        if resolved_index not in self._index_to_name:
            raise KeyError(resolved_index)
        return self._index_to_name[resolved_index]

    def index_for_name(self, class_name: str) -> int | None:
        return self._name_to_index.get(_normalize_name(class_name))

    def index_for_class_id(self, class_id: str) -> int | None:
        if not class_id.startswith("c"):
            return None
        try:
            index = int(class_id[1:])
        except ValueError:
            return None
        return index if index in self._index_to_name else None

    def names(self) -> dict[int, str]:
        return dict(self._index_to_name)

    def to_metadata(self) -> list[ClassMetadata]:
        return [
            ClassMetadata(
                class_id=self.class_id_for_index(class_index),
                class_name=class_name,
            )
            for class_index, class_name in sorted(self._index_to_name.items())
        ]

    def to_class_metadata(self) -> list[ClassMetadata]:
        return self.to_metadata()
