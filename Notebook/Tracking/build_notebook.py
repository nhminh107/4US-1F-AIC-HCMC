"""Build the standalone Kaggle YOLO26 + ByteTrack notebook."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).with_name("tracking_yolo26_bytetrack_kaggle.ipynb")


def code(source: str) -> dict:
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type":"markdown","metadata":{},"source":source.splitlines(keepends=True)}


cells=[
markdown("""# Tracking - YOLO26 + ByteTrack (Kaggle)

This notebook is the temporal Object Detection source. YOLO detects selected video frames, ByteTrack links them inside each shot, and outputs objecttrack plus trackobservation. It never writes duplicate YOLO boxes into organizer objectdetection, never connects to PostgreSQL, and never uploads data.

It decodes a video once, but only materializes a BGR image after the frame passes the sampling gate. The detector threshold is lower than the new-track threshold so ByteTrack can reconnect occluded objects.
"""),
code(r'''# Install only missing lightweight notebook dependencies. Do not install or replace Kaggle Torch/CUDA.
import importlib.util, subprocess, sys
INSTALL_MISSING_PACKAGES=True
requirements={"av":["av"],"ultralytics":["ultralytics>=8.4,<8.5"],"pyarrow":["pyarrow"]}
missing=[name for name in requirements if importlib.util.find_spec(name) is None]
if missing and not INSTALL_MISSING_PACKAGES: raise RuntimeError(f"Missing {missing}; set INSTALL_MISSING_PACKAGES=True.")
for name in missing: subprocess.check_call([sys.executable,"-m","pip","install","-q",*requirements[name]])
if missing: print("Installed",missing,". Restart kernel before running the next cell.")
else: print("Tracking dependencies already available.")
'''),
code(r'''# Imports and GPU preflight. Run after any restart.
from __future__ import annotations
import csv, gc, itertools, json, math, os, re, shutil, time, traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable. Enable Kaggle Accelerator=GPU.")
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
print("Torch",torch.__version__,"CUDA",torch.version.cuda,"GPU",torch.cuda.get_device_name(0))
'''),
code(r'''# Configuration
INPUT_DIR=Path("/kaggle/input/btc-tracking-input")
OUTPUT_DIR=Path("/kaggle/working/tracking_output")
VIDEOS_FILE=INPUT_DIR/"videos.csv"
SHOTS_FILE=INPUT_DIR/"shot.csv"                # fallback shots.csv
VIDEO_ROOT=INPUT_DIR/"videos"
MODEL_PATH=INPUT_DIR/"models"/"yolo26n.pt"     # upload a local approved YOLO26 COCO-80 weight

VIDEO_START=0
VIDEO_END: int | None=None
SAMPLING_FPS=2.0
BATCH_SIZE=32
IMAGE_SIZE=640
DETECTOR_CONFIDENCE=0.10                       # low boxes may associate to an existing track
NEW_TRACK_CONFIDENCE=0.25                      # track_high_thresh and new_track_thresh
IOU_THRESHOLD=0.70
MAX_DETECTIONS=300
MAX_LOST_SECONDS=3.0
CLASS_INDICES=(0,1,2,3,4,5,6,7,8,14,15,16,17,18,19,24,25,26,28,32,36,37)
DEVICE="cuda:0"
RUN_ID=time.strftime("tracking-%Y%m%d-%H%M%S")
RUN_PIPELINE=True
CONTINUE_ON_ERROR=True
ROWS_PER_SQL_INSERT=1_000

if SAMPLING_FPS<=0 or BATCH_SIZE<1 or IMAGE_SIZE<1 or MAX_LOST_SECONDS<=0: raise ValueError("Invalid sampling/batch settings")
if not 0<=DETECTOR_CONFIDENCE<=NEW_TRACK_CONFIDENCE<=1: raise ValueError("Require 0 <= DETECTOR_CONFIDENCE <= NEW_TRACK_CONFIDENCE <= 1")
if not VIDEOS_FILE.is_file(): raise FileNotFoundError(f"Missing {VIDEOS_FILE}")
if not SHOTS_FILE.is_file(): SHOTS_FILE=SHOTS_FILE.with_name("shots.csv")
if not SHOTS_FILE.is_file() or not MODEL_PATH.is_file(): raise FileNotFoundError("Need videos.csv, shot.csv/shots.csv, and local models/yolo26n.pt; videos/ is optional when video_url is supplied")
RUN_DIR=OUTPUT_DIR/RUN_ID
if RUN_DIR.exists(): raise FileExistsError(f"Refusing to overwrite {RUN_DIR}")
'''),
code(r'''# Input schema, local video resolver, and COCO/Open Images map for the 22 tracked classes.
TRACKED_CLASSES={
0:("person","/m/01g317"),1:("bicycle","/m/0199g"),2:("car","/m/0k4j"),3:("motorcycle","/m/04_sv"),4:("airplane","/m/0cmf2"),5:("bus","/m/01bjv"),6:("train","/m/07jdr"),7:("truck","/m/07r04"),8:("boat","/m/019jd"),
14:("bird","/m/015p6"),15:("cat","/m/01yrx"),16:("dog","/m/0bt9lr"),17:("horse","/m/03k3r"),18:("sheep","/m/07bgp"),19:("cow","/m/0k1xq1"),24:("backpack","/m/01940j"),25:("umbrella","/m/0hnnb"),26:("handbag","/m/080hkjn"),28:("suitcase","/m/01s55n"),32:("sports ball","/m/018xm"),36:("skateboard","/m/06_fw"),37:("surfboard","/m/019jd")}
# Correct the two Open Images values that share no useful inference behavior with the selection logic.
TRACKED_CLASSES[19]=("cow","/m/01xq0k1")
TRACKED_CLASSES[37]=("surfboard","/m/019jd")

def resolve_video(row: dict[str,Any]) -> Path:
    raw=str(row.get("video_path") or row.get("video_url") or "").strip()
    if not raw: raise ValueError(f"{row.get('video_id')}: missing video_path")
    raw_path=Path(raw); choices=([raw_path] if raw_path.is_absolute() else [])+[VIDEO_ROOT/raw_path,VIDEO_ROOT/raw_path.name]
    path=next((item.resolve() for item in choices if item.is_file()),None)
    if path is None: raise FileNotFoundError(f"{row.get('video_id')}: video not found under VIDEO_ROOT: {raw}")
    return path

def load_input() -> tuple[list[dict[str,Any]],dict[str,list[dict[str,int]]]]:
    videos=pd.read_csv(VIDEOS_FILE,dtype={"video_id":"string"})
    shots=pd.read_csv(SHOTS_FILE,dtype={"shot_id":"string","video_id":"string"})
    if "video_id" not in videos or not ({"video_path","video_url"}&set(videos)): raise ValueError("videos.csv requires video_id and video_path/video_url")
    if not {"shot_id","video_id","shot_index","start_ms","end_ms"}.issubset(shots): raise ValueError("shot.csv schema invalid")
    if videos.video_id.isna().any() or videos.duplicated("video_id").any(): raise ValueError("video_id must be present and unique")
    for key in ("shot_index","start_ms","end_ms"): shots[key]=pd.to_numeric(shots[key],errors="raise").astype("int64")
    if shots.shot_id.isna().any() or shots.duplicated("shot_id").any() or (shots.end_ms<=shots.start_ms).any(): raise ValueError("shot input invalid")
    selected=sorted(videos.video_id.astype(str).tolist())[VIDEO_START:VIDEO_END]
    video_by_id={str(row.video_id):dict(row._asdict()) for row in videos.itertuples(index=False)}
    records=[{"video_id":video_id,"video_path":resolve_video(video_by_id[video_id])} for video_id in selected]
    by_video={str(video_id):group.sort_values(["start_ms","shot_index","shot_id"],kind="stable").to_dict("records") for video_id,group in shots.groupby("video_id",sort=False)}
    for item in records:
        if not by_video.get(item["video_id"]): raise ValueError(f"{item['video_id']}: no shots")
    return records,by_video

VIDEOS,SHOTS_BY_VIDEO=load_input()
print("Selected videos:",len(VIDEOS),"shots:",sum(len(SHOTS_BY_VIDEO[x["video_id"]]) for x in VIDEOS))
'''),
code(r'''# Runtime ByteTrack config. Buffer is in sampled-frame units, not original video FPS.
BUFFER_SIZE=max(1,math.ceil(MAX_LOST_SECONDS*SAMPLING_FPS))
RUNTIME_TRACKER_YAML="""tracker_type: bytetrack
track_high_thresh: %s
track_low_thresh: %s
new_track_thresh: %s
track_buffer: %d
match_thresh: 0.80
fuse_score: true
"""%(NEW_TRACK_CONFIDENCE,DETECTOR_CONFIDENCE,NEW_TRACK_CONFIDENCE,BUFFER_SIZE)

def runtime_tracker_path(directory: Path) -> Path:
    path=directory/"bytetrack.runtime.yaml"
    path.write_text(RUNTIME_TRACKER_YAML,encoding="utf-8")
    return path

def numpy_value(value: Any) -> np.ndarray:
    for attribute in ("detach","cpu"):
        method=getattr(value,attribute,None)
        if callable(method): value=method()
    method=getattr(value,"numpy",None)
    if callable(method): value=method()
    return np.asarray(value)

def reset_tracker(model: YOLO) -> None:
    predictor=getattr(model,"predictor",None)
    for tracker in getattr(predictor,"trackers",()) if predictor is not None else ():
        reset=getattr(tracker,"reset",None)
        if callable(reset): reset()

def normalize_box(xyxy: np.ndarray,width: int,height: int) -> tuple[float,float,float,float] | None:
    x1,y1,x2,y2=xyxy
    x1,x2=np.clip([float(x1)/width,float(x2)/width],0,1); y1,y2=np.clip([float(y1)/height,float(y2)/height],0,1)
    return (float(x1),float(x2),float(y1),float(y2)) if x1<x2 and y1<y2 else None

def tracked_boxes(result: Any) -> list[tuple[int,int,float,np.ndarray]]:
    boxes=getattr(result,"boxes",None)
    if boxes is None or not bool(getattr(boxes,"is_track",False)) or getattr(boxes,"id",None) is None: return []
    ids=numpy_value(boxes.id).reshape(-1); classes=numpy_value(boxes.cls).reshape(-1); scores=numpy_value(boxes.conf).reshape(-1); coords=numpy_value(boxes.xyxy).reshape(-1,4)
    if len({len(ids),len(classes),len(scores),len(coords)})!=1: raise RuntimeError("YOLO output arrays have inconsistent lengths")
    return [(int(track_id),int(class_id),float(score),np.asarray(box,np.float32)) for track_id,class_id,score,box in zip(ids,classes,scores,coords) if int(track_id)>=0 and int(class_id) in TRACKED_CLASSES]
'''),
code(r'''# Tracking core. PyAV frame.to_ndarray happens only after the 2 FPS gate.
@dataclass
class Accumulator:
    local_track_id:int; shot_id:str; class_id:str; start_ms:int; end_ms:int; start_frame_idx:int; end_frame_idx:int; confidence_sum:float=0; observation_count:int=0
    def add(self,timestamp_ms:int,frame_idx:int,confidence:float) -> None:
        self.end_ms=timestamp_ms; self.end_frame_idx=frame_idx; self.confidence_sum+=confidence; self.observation_count+=1

def iter_decoded(path: Path) -> Iterable[tuple[int,int,Any]]:
    with av.open(str(path)) as container:
        stream=container.streams.video[0]; fallback=float(stream.average_rate) if stream.average_rate else 30.0
        for index,frame in enumerate(container.decode(stream)):
            timestamp=round(float(frame.pts*frame.time_base)*1000) if frame.pts is not None and frame.time_base is not None else round(index/fallback*1000)
            yield timestamp,index,frame

def track_video(model: YOLO,video: dict[str,Any],shots: list[dict[str,int]],tracker_path: Path) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    tracks,observations,accumulators,local_ids=[],[],{},{}
    next_local=1; shot_index=0; active_shot=None; next_sample=shots[0]["start_ms"]; interval=1000/SAMPLING_FPS; pending=[]
    def flush() -> None:
        nonlocal next_local
        if not pending: return
        results=list(model.track(source=[item[3] for item in pending],persist=True,tracker=str(tracker_path),conf=DETECTOR_CONFIDENCE,iou=IOU_THRESHOLD,max_det=MAX_DETECTIONS,classes=list(CLASS_INDICES),imgsz=IMAGE_SIZE,device=DEVICE,verbose=False))
        if len(results)!=len(pending): raise RuntimeError("YOLO result count mismatch")
        for (shot,timestamp,frame_idx,image),result in zip(pending,results):
            height,width=image.shape[:2]
            for internal,class_index,confidence,xyxy in tracked_boxes(result):
                box=normalize_box(xyxy,width,height)
                if box is None: continue
                class_id=TRACKED_CLASSES[class_index][1]; key=(shot["shot_id"],class_id,internal)
                if key not in local_ids:
                    local_ids[key]=next_local
                    accumulators[next_local]=Accumulator(next_local,shot["shot_id"],class_id,timestamp,timestamp,frame_idx,frame_idx)
                    next_local+=1
                local_id=local_ids[key]; accumulators[local_id].add(timestamp,frame_idx,confidence)
                x_min,x_max,y_min,y_max=box
                observations.append({"local_track_id":local_id,"frame_idx":frame_idx,"timestamp_ms":timestamp,"confidence":confidence,"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max})
        pending.clear()
    for timestamp,frame_idx,frame in iter_decoded(video["video_path"]):
        while shot_index<len(shots) and timestamp>=shots[shot_index]["end_ms"]:
            shot_index+=1
            if shot_index<len(shots): next_sample=shots[shot_index]["start_ms"]
        if shot_index>=len(shots): break
        shot=shots[shot_index]
        if timestamp<shot["start_ms"] or timestamp<next_sample: continue
        while next_sample<=timestamp: next_sample+=interval
        if active_shot!=shot["shot_id"]:
            flush(); reset_tracker(model); active_shot=shot["shot_id"]
        image=frame.to_ndarray(format="bgr24")
        pending.append((shot,timestamp,frame_idx,image))
        if len(pending)>=BATCH_SIZE: flush()
    flush()
    for item in accumulators.values():
        if item.observation_count:
            tracks.append({"local_track_id":item.local_track_id,"shot_id":item.shot_id,"class_id":item.class_id,"start_ms":item.start_ms,"end_ms":max(item.end_ms,item.start_ms+1),"start_frame_idx":item.start_frame_idx,"end_frame_idx":item.end_frame_idx,"observation_count":item.observation_count,"avg_confidence":item.confidence_sum/item.observation_count,"model_name":"YOLO26","model_version":MODEL_PATH.name,"tracker_name":"ByteTrack","tracker_version":"ultralytics","sampling_fps":SAMPLING_FPS,"mapping_version":"coco80-openimages-v1"})
    valid={item["local_track_id"] for item in tracks}
    return tracks,[item for item in observations if item["local_track_id"] in valid]
'''),
code(r'''# Output writers. SQL preserves generated objecttrack IDs through a temporary local-ID map.
TRACK_COLUMNS=["local_track_id","shot_id","class_id","start_ms","end_ms","start_frame_idx","end_frame_idx","observation_count","avg_confidence","model_name","model_version","tracker_name","tracker_version","sampling_fps","mapping_version"]
OBS_COLUMNS=["local_track_id","frame_idx","timestamp_ms","confidence","x_min","x_max","y_min","y_max"]
DB_TRACK_COLUMNS=TRACK_COLUMNS[1:]
def literal(value: Any) -> str:
    if isinstance(value,(int,np.integer)): return str(int(value))
    if isinstance(value,(float,np.floating)): return format(float(value),".9g")
    return "'" + str(value).replace("'","''") + "'"
INTEGER_COLUMNS={"local_track_id","start_ms","end_ms","start_frame_idx","end_frame_idx","observation_count","frame_idx","timestamp_ms"}
FLOAT_COLUMNS={"avg_confidence","sampling_fps","confidence","x_min","x_max","y_min","y_max"}
def csv_literal(value: str,column: str) -> str:
    if column in INTEGER_COLUMNS: return str(int(value))
    if column in FLOAT_COLUMNS: return format(float(value),".9g")
    return literal(value)
def write_sql(path: Path,track_csv: Path,observation_csv: Path) -> None:
    with path.open("w",encoding="utf-8") as handle:
        handle.write("BEGIN;\nCREATE TEMP TABLE _aic_tracking_map (local_track_id bigint PRIMARY KEY, track_id bigint NOT NULL) ON COMMIT DROP;\n")
        with track_csv.open("r",newline="",encoding="utf-8") as source:
            for row in csv.DictReader(source):
                values=", ".join(csv_literal(row[column],column) for column in DB_TRACK_COLUMNS)
                columns=", ".join(DB_TRACK_COLUMNS)
                handle.write("WITH inserted AS (INSERT INTO objecttrack ("+columns+") VALUES ("+values+") RETURNING track_id) INSERT INTO _aic_tracking_map (local_track_id, track_id) SELECT "+csv_literal(row["local_track_id"],"local_track_id")+", track_id FROM inserted;\n")
        with observation_csv.open("r",newline="",encoding="utf-8") as source:
            reader=csv.DictReader(source)
            while batch:=list(itertools.islice(reader,ROWS_PER_SQL_INSERT)):
                values=",\n".join("("+", ".join(csv_literal(row[column],column) for column in OBS_COLUMNS)+")" for row in batch)
                handle.write("INSERT INTO trackobservation (track_id, frame_idx, timestamp_ms, confidence, x_min, x_max, y_min, y_max) SELECT m.track_id, v.frame_idx, v.timestamp_ms, v.confidence, v.x_min, v.x_max, v.y_min, v.y_max FROM (VALUES "+values+") AS v(local_track_id, frame_idx, timestamp_ms, confidence, x_min, x_max, y_min, y_max) JOIN _aic_tracking_map AS m ON m.local_track_id=v.local_track_id;\n")
        handle.write("COMMIT;\n")
'''),
code(r'''# Real run. ClassID rows for all mapped Open Images MIDs must be loaded before executing generated SQL.
if RUN_PIPELINE:
    started=time.monotonic(); temporary=RUN_DIR.with_name(RUN_DIR.name+".partial")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    tracker_path=runtime_tracker_path(temporary)
    model=YOLO(str(MODEL_PATH))
    names=getattr(model,"names",{})
    for index,(expected,_) in TRACKED_CLASSES.items():
        actual=str(names[index] if isinstance(names,dict) else names[index]).casefold()
        if actual!=expected.casefold(): raise ValueError(f"YOLO class mismatch index {index}: {actual!r} != {expected!r}")
    failures=[]; track_count=0; observation_count=0; next_local_track_id=1
    track_csv=temporary/"object_track.csv"; observation_csv=temporary/"track_observation.csv"
    with track_csv.open("w",newline="",encoding="utf-8") as track_file, observation_csv.open("w",newline="",encoding="utf-8") as observation_file:
        track_writer=csv.DictWriter(track_file,fieldnames=TRACK_COLUMNS); observation_writer=csv.DictWriter(observation_file,fieldnames=OBS_COLUMNS)
        track_writer.writeheader(); observation_writer.writeheader()
        for position,video in enumerate(VIDEOS,1):
            try:
                tracks,observations=track_video(model,video,SHOTS_BY_VIDEO[video["video_id"]],tracker_path)
                remap={item["local_track_id"]:next_local_track_id+offset for offset,item in enumerate(tracks)}
                for item in tracks: item["local_track_id"]=remap[item["local_track_id"]]
                for item in observations: item["local_track_id"]=remap[item["local_track_id"]]
                track_writer.writerows(tracks); observation_writer.writerows(observations)
                track_count+=len(tracks); observation_count+=len(observations); next_local_track_id+=len(tracks)
                print(f"{position}/{len(VIDEOS)} {video['video_id']}: tracks={len(tracks):,}, observations={len(observations):,}")
            except Exception as error:
                if not CONTINUE_ON_ERROR: raise
                failures.append({"video_id":video["video_id"],"reason":str(error),"traceback":traceback.format_exc(limit=3)})
    with (temporary/"failures.jsonl").open("w",encoding="utf-8") as handle:
        for item in failures: handle.write(json.dumps(item,ensure_ascii=False)+"\n")
    write_sql(temporary/"insert_tracking.sql",track_csv,observation_csv)
    summary={"run_id":RUN_ID,"videos_requested":len(VIDEOS),"videos_failed":len(failures),"tracks":track_count,"observations":observation_count,"sampling_fps":SAMPLING_FPS,"detector_confidence":DETECTOR_CONFIDENCE,"new_track_confidence":NEW_TRACK_CONFIDENCE,"max_lost_seconds":MAX_LOST_SECONDS,"track_buffer_sampled_frames":BUFFER_SIZE,"batch_size":BATCH_SIZE,"image_size":IMAGE_SIZE,"gpu":torch.cuda.get_device_name(0),"elapsed_seconds":round(time.monotonic()-started,3),"peak_vram_bytes":int(torch.cuda.max_memory_allocated())}
    (temporary/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.rename(RUN_DIR)
    print("Completed. Download Kaggle Output:",RUN_DIR)
else:
    print("RUN_PIPELINE=False: input validation only.")
'''),
code(r'''# Post-run validation.
if RUN_PIPELINE:
    tracks=pd.read_csv(RUN_DIR/"object_track.csv")
    assert tracks.local_track_id.is_unique
    assert (tracks.end_ms>tracks.start_ms).all() and (tracks.observation_count>0).all()
    known_track_ids=set(tracks.local_track_id)
    observation_count=0
    for observations in pd.read_csv(RUN_DIR/"track_observation.csv",chunksize=100_000):
        assert set(observations.local_track_id).issubset(known_track_ids)
        assert observations[["confidence","x_min","x_max","y_min","y_max"]].apply(lambda item:item.between(0,1).all()).all()
        assert (observations.x_min<observations.x_max).all() and (observations.y_min<observations.y_max).all()
        observation_count+=len(observations)
    sql=(RUN_DIR/"insert_tracking.sql").read_text(encoding="utf-8")
    assert "CREATE TEMP TABLE _aic_tracking_map" in sql and "INSERT INTO trackobservation" in sql
    print("Validation passed:",len(tracks),"tracks;",observation_count,"observations")
''')
]
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUTPUT.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("Wrote",OUTPUT)
