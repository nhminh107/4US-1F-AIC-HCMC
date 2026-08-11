"""Canonical COCO 80 to Open Images mapping for YOLO tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from BackEnd.CONFIG import TRACKING_CLASS_MAPPING_VERSION


@dataclass(frozen=True, slots=True)
class CocoOpenImagesClass:
    """One validated YOLO COCO class and its canonical Open Images class."""

    coco_index: int
    coco_name: str
    class_id: str
    class_name: str


COCO_TO_OPENIMAGES: Mapping[int, CocoOpenImagesClass] = {
    0: CocoOpenImagesClass(0, "person", "/m/01g317", "Person"),
    1: CocoOpenImagesClass(1, "bicycle", "/m/0199g", "Bicycle"),
    2: CocoOpenImagesClass(2, "car", "/m/0k4j", "Car"),
    3: CocoOpenImagesClass(3, "motorcycle", "/m/04_sv", "Motorcycle"),
    4: CocoOpenImagesClass(4, "airplane", "/m/0cmf2", "Airplane"),
    5: CocoOpenImagesClass(5, "bus", "/m/01bjv", "Bus"),
    6: CocoOpenImagesClass(6, "train", "/m/07jdr", "Train"),
    7: CocoOpenImagesClass(7, "truck", "/m/07r04", "Truck"),
    8: CocoOpenImagesClass(8, "boat", "/m/019jd", "Boat"),
    9: CocoOpenImagesClass(9, "traffic light", "/m/015qff", "Traffic light"),
    10: CocoOpenImagesClass(10, "fire hydrant", "/m/01pns0", "Fire hydrant"),
    11: CocoOpenImagesClass(11, "stop sign", "/m/02pv19", "Stop sign"),
    12: CocoOpenImagesClass(12, "parking meter", "/m/015qbp", "Parking meter"),
    13: CocoOpenImagesClass(13, "bench", "/m/0cvnqh", "Bench"),
    14: CocoOpenImagesClass(14, "bird", "/m/015p6", "Bird"),
    15: CocoOpenImagesClass(15, "cat", "/m/01yrx", "Cat"),
    16: CocoOpenImagesClass(16, "dog", "/m/0bt9lr", "Dog"),
    17: CocoOpenImagesClass(17, "horse", "/m/03k3r", "Horse"),
    18: CocoOpenImagesClass(18, "sheep", "/m/07bgp", "Sheep"),
    19: CocoOpenImagesClass(19, "cow", "/m/01xq0k1", "Cattle"),
    20: CocoOpenImagesClass(20, "elephant", "/m/0bwd_0j", "Elephant"),
    21: CocoOpenImagesClass(21, "bear", "/m/01dws", "Bear"),
    22: CocoOpenImagesClass(22, "zebra", "/m/0898b", "Zebra"),
    23: CocoOpenImagesClass(23, "giraffe", "/m/03bk1", "Giraffe"),
    24: CocoOpenImagesClass(24, "backpack", "/m/01940j", "Backpack"),
    25: CocoOpenImagesClass(25, "umbrella", "/m/0hnnb", "Umbrella"),
    26: CocoOpenImagesClass(26, "handbag", "/m/080hkjn", "Handbag"),
    27: CocoOpenImagesClass(27, "tie", "/m/01rkbr", "Tie"),
    28: CocoOpenImagesClass(28, "suitcase", "/m/01s55n", "Suitcase"),
    29: CocoOpenImagesClass(29, "frisbee", "/m/02wmf", "Flying disc"),
    30: CocoOpenImagesClass(30, "skis", "/m/071p9", "Ski"),
    31: CocoOpenImagesClass(31, "snowboard", "/m/06__v", "Snowboard"),
    32: CocoOpenImagesClass(32, "sports ball", "/m/018xm", "Ball"),
    33: CocoOpenImagesClass(33, "kite", "/m/02zt3", "Kite"),
    34: CocoOpenImagesClass(34, "baseball bat", "/m/03g8mr", "Baseball bat"),
    35: CocoOpenImagesClass(35, "baseball glove", "/m/03grzl", "Baseball glove"),
    36: CocoOpenImagesClass(36, "skateboard", "/m/06_fw", "Skateboard"),
    37: CocoOpenImagesClass(37, "surfboard", "/m/019w40", "Surfboard"),
    38: CocoOpenImagesClass(38, "tennis racket", "/m/0h8my_4", "Tennis racket"),
    39: CocoOpenImagesClass(39, "bottle", "/m/04dr76w", "Bottle"),
    40: CocoOpenImagesClass(40, "wine glass", "/m/09tvcd", "Wine glass"),
    41: CocoOpenImagesClass(41, "cup", "/m/02p5f1q", "Coffee cup"),
    42: CocoOpenImagesClass(42, "fork", "/m/0dt3t", "Fork"),
    43: CocoOpenImagesClass(43, "knife", "/m/04ctx", "Knife"),
    44: CocoOpenImagesClass(44, "spoon", "/m/0cmx8", "Spoon"),
    45: CocoOpenImagesClass(45, "bowl", "/m/04kkgm", "Bowl"),
    46: CocoOpenImagesClass(46, "banana", "/m/09qck", "Banana"),
    47: CocoOpenImagesClass(47, "apple", "/m/014j1m", "Apple"),
    48: CocoOpenImagesClass(48, "sandwich", "/m/0l515", "Sandwich"),
    49: CocoOpenImagesClass(49, "orange", "/m/0cyhj_", "Orange"),
    50: CocoOpenImagesClass(50, "broccoli", "/m/0hkxq", "Broccoli"),
    51: CocoOpenImagesClass(51, "carrot", "/m/0fj52s", "Carrot"),
    52: CocoOpenImagesClass(52, "hot dog", "/m/01b9xk", "Hot dog"),
    53: CocoOpenImagesClass(53, "pizza", "/m/0663v", "Pizza"),
    54: CocoOpenImagesClass(54, "donut", "/m/0jy4k", "Doughnut"),
    55: CocoOpenImagesClass(55, "cake", "/m/0fszt", "Cake"),
    56: CocoOpenImagesClass(56, "chair", "/m/01mzpv", "Chair"),
    57: CocoOpenImagesClass(57, "couch", "/m/02crq1", "Couch"),
    58: CocoOpenImagesClass(58, "potted plant", "/m/03fp41", "Houseplant"),
    59: CocoOpenImagesClass(59, "bed", "/m/03ssj5", "Bed"),
    60: CocoOpenImagesClass(
        60,
        "dining table",
        "/m/0h8n5zk",
        "Kitchen & dining room table",
    ),
    61: CocoOpenImagesClass(61, "toilet", "/m/09g1w", "Toilet"),
    62: CocoOpenImagesClass(62, "tv", "/m/07c52", "Television"),
    63: CocoOpenImagesClass(63, "laptop", "/m/01c648", "Laptop"),
    64: CocoOpenImagesClass(64, "mouse", "/m/04rmv", "Mouse"),
    65: CocoOpenImagesClass(65, "remote", "/m/0qjjc", "Remote control"),
    66: CocoOpenImagesClass(66, "keyboard", "/m/01m2v", "Computer keyboard"),
    67: CocoOpenImagesClass(67, "cell phone", "/m/050k8", "Mobile phone"),
    68: CocoOpenImagesClass(68, "microwave", "/m/0fx9l", "Microwave oven"),
    69: CocoOpenImagesClass(69, "oven", "/m/029bxz", "Oven"),
    70: CocoOpenImagesClass(70, "toaster", "/m/01k6s3", "Toaster"),
    71: CocoOpenImagesClass(71, "sink", "/m/0130jx", "Sink"),
    72: CocoOpenImagesClass(72, "refrigerator", "/m/040b_t", "Refrigerator"),
    73: CocoOpenImagesClass(73, "book", "/m/0bt_c3", "Book"),
    74: CocoOpenImagesClass(74, "clock", "/m/01x3z", "Clock"),
    75: CocoOpenImagesClass(75, "vase", "/m/02s195", "Vase"),
    76: CocoOpenImagesClass(76, "scissors", "/m/01lsmm", "Scissors"),
    77: CocoOpenImagesClass(77, "teddy bear", "/m/0kmg4", "Teddy bear"),
    78: CocoOpenImagesClass(78, "hair drier", "/m/03wvsk", "Hair dryer"),
    79: CocoOpenImagesClass(79, "toothbrush", "/m/012xff", "Toothbrush"),
}


def validate_coco_names(model_names: Mapping[int, str] | list[str]) -> None:
    """Fail when a model does not expose the expected COCO 80 class order."""

    if isinstance(model_names, list):
        resolved_names = dict(enumerate(model_names))
    else:
        resolved_names = {int(index): str(name) for index, name in model_names.items()}

    expected_indices = set(COCO_TO_OPENIMAGES)
    actual_indices = set(resolved_names)
    if actual_indices != expected_indices:
        raise ValueError(
            "YOLO model must expose exactly the 80 COCO classes with indices 0-79."
        )

    mismatches = [
        (index, item.coco_name, resolved_names[index])
        for index, item in COCO_TO_OPENIMAGES.items()
        if resolved_names[index].strip().casefold() != item.coco_name.casefold()
    ]
    if mismatches:
        index, expected, actual = mismatches[0]
        raise ValueError(
            f"YOLO class mismatch at index {index}: expected {expected!r}, "
            f"got {actual!r}."
        )


def get_canonical_class(coco_index: int) -> CocoOpenImagesClass:
    """Return the canonical Open Images class for one COCO class index."""

    try:
        return COCO_TO_OPENIMAGES[int(coco_index)]
    except KeyError as error:
        raise ValueError(f"Unsupported COCO class index: {coco_index}.") from error


__all__ = [
    "COCO_TO_OPENIMAGES",
    "CocoOpenImagesClass",
    "TRACKING_CLASS_MAPPING_VERSION",
    "get_canonical_class",
    "validate_coco_names",
]
