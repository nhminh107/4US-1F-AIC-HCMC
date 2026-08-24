"""Build the Kaggle organizer ObjectDetection ingestion notebook."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).with_name("object_detection_ingestion_kaggle.ipynb")


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


cells = [
markdown("""# Object Detection - organizer JSONL ingestion

BTC Open Images detections are canonical. This notebook does not run YOLO, Faster R-CNN, or any GPU inference, and does not replace data from BTC. It validates JSONL then writes a database-ready CSV and SQL script for the existing objectdetection table.

Tracking YOLO detections remain in objecttrack and trackobservation; they must not be duplicated into objectdetection.
"""),
code(r'''from __future__ import annotations
import csv, json, math, shutil, time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
'''),
code(r'''# Configuration
INPUT_DIR = Path("/kaggle/input/btc-object-jsonl")
OBJECT_JSONL_ROOT = INPUT_DIR / "objects-aic25-b1-jsonl"
OUTPUT_DIR = Path("/kaggle/working/object_detection_ingestion")
VIDEO_IDS: list[str] | None = None
START_OFFSET = 0
MAX_VIDEOS: int | None = None
ROWS_PER_SQL_INSERT = 1_000
MODEL_NAME = "faster_rcnn/inception_resnet_v2"
MODEL_VERSION = "openimages_v4/1"
RUN_ID = time.strftime("object-detection-%Y%m%d-%H%M%S")

if not OBJECT_JSONL_ROOT.is_dir(): raise FileNotFoundError(f"Missing organizer JSONL root: {OBJECT_JSONL_ROOT}")
if ROWS_PER_SQL_INSERT < 1: raise ValueError("ROWS_PER_SQL_INSERT must be positive")
RUN_DIR = OUTPUT_DIR / RUN_ID
if RUN_DIR.exists(): raise FileExistsError(f"Refusing to overwrite {RUN_DIR}")
'''),
code(r'''# Validation and deterministic conversion. Frame identity follows BTC convention: VIDEO_ID_001.
OUTPUT_COLUMNS = ["frame_id","class_id","confidence","x_min","x_max","y_min","y_max","model_name","model_version"]

def keyframe_number(path: Path) -> int:
    try: result=int(path.stem)
    except ValueError as error: raise ValueError(f"Invalid numeric keyframe file: {path}") from error
    if result < 1: raise ValueError(f"Keyframe index must be positive: {path}")
    return result

def number(data: dict[str,Any], key: str, path: Path, line: int) -> float:
    try: value=float(data[key])
    except (KeyError,TypeError,ValueError) as error: raise ValueError(f"{path}:{line}: {key} must be numeric") from error
    if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError(f"{path}:{line}: {key} must be finite [0,1]")
    return value

def text(data: dict[str,Any], key: str, path: Path, line: int) -> str:
    value=data.get(key)
    if not isinstance(value,str) or not value: raise ValueError(f"{path}:{line}: {key} must be non-empty string")
    return value

def selected_paths() -> list[Path]:
    paths=sorted(OBJECT_JSONL_ROOT.glob("*/*.jsonl"))
    video_ids=sorted({path.parent.name for path in paths})
    if VIDEO_IDS is not None:
        unknown=set(VIDEO_IDS)-set(video_ids)
        if unknown: raise ValueError(f"VIDEO_IDS not found: {sorted(unknown)}")
        video_ids=[item for item in video_ids if item in set(VIDEO_IDS)]
    video_ids=video_ids[START_OFFSET:]
    if MAX_VIDEOS is not None: video_ids=video_ids[:MAX_VIDEOS]
    wanted=set(video_ids)
    return [path for path in paths if path.parent.name in wanted]

def iter_rows(paths: Iterable[Path], failures: list[dict[str,Any]]) -> Iterable[dict[str,Any]]:
    for path in paths:
        video_id=path.parent.name; index=keyframe_number(path); frame_id=f"{video_id}_{index:03d}"
        with path.open("r",encoding="utf-8") as handle:
            for line_number,line in enumerate(handle,1):
                if not line.strip(): continue
                try:
                    record=json.loads(line)
                    if not isinstance(record,dict): raise ValueError("JSON row must be object")
                    if record.get("video_id") != video_id or record.get("keyframe_index") != index: raise ValueError("BTC video_id/keyframe_index mismatch")
                    bbox=record.get("bbox")
                    if not isinstance(bbox,dict): raise ValueError("bbox must be object")
                    row={"frame_id":frame_id,"class_id":text(record,"class_mid",path,line_number),"confidence":number(record,"confidence",path,line_number),"x_min":number(bbox,"x_min",path,line_number),"x_max":number(bbox,"x_max",path,line_number),"y_min":number(bbox,"y_min",path,line_number),"y_max":number(bbox,"y_max",path,line_number),"model_name":MODEL_NAME,"model_version":MODEL_VERSION}
                    if row["x_min"] >= row["x_max"] or row["y_min"] >= row["y_max"]: raise ValueError("bbox has non-positive area")
                    yield row
                except Exception as error:
                    failures.append({"path":str(path.relative_to(OBJECT_JSONL_ROOT)),"line":line_number,"reason":str(error)})

PATHS=selected_paths()
print(f"Selected {len(PATHS):,} JSONL files across {len(set(path.parent.name for path in PATHS)):,} videos")
'''),
code(r'''# Output: database CSV, failures, manifest, and batched standard SQL. No database connection is made.
def literal(value: Any) -> str:
    if isinstance(value,float): return format(value,".9g")
    return "'" + str(value).replace("'","''") + "'"

temporary=RUN_DIR.with_name(RUN_DIR.name+".partial")
if temporary.exists(): shutil.rmtree(temporary)
temporary.mkdir(parents=True)
failures: list[dict[str,Any]]=[]
count=0
with (temporary/"object_detection.csv").open("w",newline="",encoding="utf-8") as csv_file, (temporary/"insert_object_detection.sql").open("w",encoding="utf-8") as sql_file:
    writer=csv.DictWriter(csv_file,fieldnames=OUTPUT_COLUMNS); writer.writeheader()
    sql_file.write("BEGIN;\n")
    batch=[]
    for row in iter_rows(PATHS,failures):
        writer.writerow(row); count+=1; batch.append(row)
        if len(batch)>=ROWS_PER_SQL_INSERT:
            values=",\n".join("("+", ".join(literal(item[column]) for column in OUTPUT_COLUMNS)+")" for item in batch)
            sql_file.write("INSERT INTO objectdetection (frame_id, class_id, confidence, x_min, x_max, y_min, y_max, model_name, model_version) VALUES\n"+values+";\n")
            batch.clear()
    if batch:
        values=",\n".join("("+", ".join(literal(item[column]) for column in OUTPUT_COLUMNS)+")" for item in batch)
        sql_file.write("INSERT INTO objectdetection (frame_id, class_id, confidence, x_min, x_max, y_min, y_max, model_name, model_version) VALUES\n"+values+";\n")
    sql_file.write("COMMIT;\n")
with (temporary/"failures.jsonl").open("w",encoding="utf-8") as handle:
    for row in failures: handle.write(json.dumps(row,ensure_ascii=False)+"\n")
summary={"run_id":RUN_ID,"model_name":MODEL_NAME,"model_version":MODEL_VERSION,"jsonl_files":len(PATHS),"rows":count,"failures":len(failures),"note":"Run insert_object_detection.sql exactly once after ClassID and official Frame have been loaded. ObjectDetection has no natural idempotency key."}
(temporary/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
temporary.rename(RUN_DIR)
print(summary)
print("Download Kaggle Output:",RUN_DIR)
'''),
code(r'''# Post-run validation streams the large CSV instead of loading all organizer rows into RAM.
row_count=0
for table in pd.read_csv(RUN_DIR/"object_detection.csv",chunksize=100_000):
    assert list(table.columns)==OUTPUT_COLUMNS
    assert table[["confidence","x_min","x_max","y_min","y_max"]].apply(lambda column: column.between(0,1).all()).all()
    assert (table.x_min<table.x_max).all() and (table.y_min<table.y_max).all()
    row_count+=len(table)
assert row_count==json.loads((RUN_DIR/"summary.json").read_text())["rows"]
assert (RUN_DIR/"insert_object_detection.sql").read_text(encoding="utf-8").startswith("BEGIN;\n")
print("Validation passed:",row_count,"ObjectDetection rows")
'''),
]
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUTPUT.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("Wrote",OUTPUT)
