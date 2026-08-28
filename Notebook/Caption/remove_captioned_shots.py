"""Remove already-captioned shot IDs from a caption selection artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SHOT_ID_PATTERN = re.compile(
    r"VALUES\s*\(\s*'(?:[^']|'')*'\s*,\s*'(L\d{2}_V\d{3}_S\d{3})'\s*,",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions-sql",
        type=Path,
        action="append",
        required=True,
        help="Caption SQL file to exclude. Repeat the option for multiple files.",
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=REPO_ROOT / "Notebook/content/caption_selection",
    )
    parser.add_argument(
        "--no-restore-previous-exclusions",
        action="store_true",
        help="Do not restore rows previously removed by this helper before applying new SQL files.",
    )
    return parser.parse_args()


def captioned_shot_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Caption SQL does not exist: {path}")
    shot_ids = set(SHOT_ID_PATTERN.findall(path.read_text(encoding="utf-8")))
    if not shot_ids:
        raise ValueError(f"No caption shot_id values were found in: {path}")
    return shot_ids


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    arguments = parse_args()
    selection_dir = arguments.selection_dir.resolve()
    selected_path = selection_dir / "shots_for_caption.csv"
    audit_path = selection_dir / "caption_selection_audit.csv"
    summary_path = selection_dir / "selection_summary.json"
    if not selected_path.is_file() or not audit_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Selection directory must contain shots_for_caption.csv, audit, and summary.")

    captioned_ids = set().union(
        *(captioned_shot_ids(path.resolve()) for path in arguments.captions_sql)
    )
    with selected_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        selected_fields = list(reader.fieldnames or [])
    if "shot_id" not in selected_fields:
        raise ValueError("shots_for_caption.csv is missing shot_id.")

    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    output_rows: list[dict[str, str]] = []
    restored_ids: set[str] = set()
    removed_ids: set[str] = set()
    selected_before_exclusion = 0
    with audit_path.open("r", encoding="utf-8-sig", newline="") as source, temporary_audit.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        audit_fields = list(reader.fieldnames or [])
        required = {"shot_id", "selected", "selection_tier", "selection_reason"}
        if required - set(audit_fields):
            raise ValueError(f"caption_selection_audit.csv is missing: {sorted(required - set(audit_fields))}")
        writer = csv.DictWriter(target, fieldnames=audit_fields, extrasaction="raise")
        writer.writeheader()
        for row in reader:
            previously_removed = (
                row["selection_tier"] == "excluded_already_captioned"
                and row["selection_reason"] == "already_captioned"
            )
            restored = previously_removed and not arguments.no_restore_previous_exclusions
            selected_before = row["selected"] == "True" or restored
            if restored:
                restored_ids.add(row["shot_id"])
                row["selected"] = "True"
                row["selection_tier"] = "restored_previous_selection"
                row["selection_reason"] = "restored_previous_selection"
            if selected_before:
                selected_before_exclusion += 1
            if selected_before and row["shot_id"] in captioned_ids:
                removed_ids.add(row["shot_id"])
                row["selected"] = "False"
                row["selection_tier"] = "excluded_already_captioned"
                row["selection_reason"] = "already_captioned"
            if row["selected"] == "True":
                output_rows.append({field: row[field] for field in selected_fields})
            writer.writerow(row)
    temporary_audit.replace(audit_path)
    atomic_write_csv(selected_path, selected_fields, output_rows)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = summary.setdefault("selection", {})
    selection["selected_before_caption_exclusion"] = selected_before_exclusion
    selection["captioned_shot_ids_in_sql"] = len(captioned_ids)
    selection["captioned_shots_removed"] = len(removed_ids)
    selection["restored_previous_exclusions"] = len(restored_ids)
    selection["caption_sql_sources"] = [str(path.resolve()) for path in arguments.captions_sql]
    selection["selected_after_caption_exclusion"] = len(output_rows)
    summary["selected"] = len(output_rows)
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_summary.replace(summary_path)

    print(
        json.dumps(
            {
                "captioned_shot_ids_in_sql": len(captioned_ids),
                "restored_previous_exclusions": len(restored_ids),
                "captioned_shots_removed": len(removed_ids),
                "remaining_shots_for_caption": len(output_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
