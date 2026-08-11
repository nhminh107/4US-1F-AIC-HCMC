"""Tests for the fixed COCO 80 to Open Images tracking mapping."""

from __future__ import annotations

import pytest

from BackEnd.app.tracking.class_mapping import (
    COCO_TO_OPENIMAGES,
    get_canonical_class,
    validate_coco_names,
)


def test_mapping_covers_each_coco_index_once() -> None:
    assert list(COCO_TO_OPENIMAGES) == list(range(80))
    assert len({item.coco_name for item in COCO_TO_OPENIMAGES.values()}) == 80
    assert len({item.class_id for item in COCO_TO_OPENIMAGES.values()}) == 80


def test_approved_aliases_are_explicit() -> None:
    assert get_canonical_class(19).class_name == "Cattle"
    assert get_canonical_class(32).class_name == "Ball"
    assert get_canonical_class(41).class_name == "Coffee cup"
    assert get_canonical_class(58).class_name == "Houseplant"


def test_model_names_must_match_coco_order() -> None:
    names = {
        index: mapped_class.coco_name
        for index, mapped_class in COCO_TO_OPENIMAGES.items()
    }
    validate_coco_names(names)
    names[0] = "not-person"

    with pytest.raises(ValueError, match="class mismatch at index 0"):
        validate_coco_names(names)


def test_model_must_have_exactly_80_classes() -> None:
    with pytest.raises(ValueError, match="exactly the 80 COCO classes"):
        validate_coco_names({0: "person"})
