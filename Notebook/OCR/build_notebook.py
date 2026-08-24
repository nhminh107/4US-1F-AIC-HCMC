"""Build a standalone Kaggle OCR notebook."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path(__file__).with_name("ocr_kaggle.ipynb")


def cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


cells = [
cell("markdown", """# OCR - shot-aware GPU preprocessing

Notebook reads only attached Kaggle CSV/image files and writes under /kaggle/working. It never connects to PostgreSQL, Cloudflare/R2, or uploads anything.

It selects representative keyframes per shot before OCR: temporal coverage anchors plus optional diversity candidates sourced from an existing SigLIP frame-embedding artifact. Default recognition remains PP-OCRv5 mobile detector with VietOCR vgg_transformer. Outputs are CSV and SQL INSERT statements for manual import into the existing ocr table.
"""),
cell("code", r'''# Install once in a Kaggle session if packages are missing. Restart the kernel after installation.
import importlib.util, subprocess, sys
INSTALL_MISSING_PACKAGES = True
PADDLE_GPU_WHEEL = "paddlepaddle-gpu==3.0.0"
PADDLE_WHEEL_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
requirements = {
    "paddle": [PADDLE_GPU_WHEEL, "-i", PADDLE_WHEEL_INDEX],
    "paddleocr": ["paddleocr>=3.0,<4"],
    "vietocr": ["vietocr"],
    "cv2": ["opencv-python-headless"],
    "pyarrow": ["pyarrow"],
}
missing = [name for name in requirements if importlib.util.find_spec(name) is None]
if missing and not INSTALL_MISSING_PACKAGES:
    raise RuntimeError(f"Missing {missing}; set INSTALL_MISSING_PACKAGES=True then restart kernel.")
for name in missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *requirements[name]])
print("Installed", missing, "- restart kernel if this list is non-empty." if missing else "OCR dependencies are already available.")
'''),
cell("code", r'''# Imports and GPU preflight. Run after package-install restart.
from __future__ import annotations
import gc, json, math, os, shutil, time, traceback, unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
import cv2
import numpy as np
import pandas as pd
import paddle
import torch
from PIL import Image

if not (paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0):
    raise RuntimeError("Paddle cannot access CUDA. Enable Kaggle GPU, install a CUDA-compatible paddlepaddle-gpu wheel, and restart.")
if not torch.cuda.is_available():
    raise RuntimeError("VietOCR cannot access CUDA. Enable Kaggle GPU and restart.")
paddle.set_device("gpu:0")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("Paddle", paddle.__version__, "| Torch", torch.__version__, "| CUDA", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
'''),
cell("code", r'''# Configuration. Change only this cell for a Kaggle run.
INPUT_DIR = Path("/kaggle/input/btc-ocr-input")
OUTPUT_DIR = Path("/kaggle/working/ocr_output")
SHOT_FILE = INPUT_DIR / "shot.csv"                 # fallback: shots.csv
KEYFRAME_FILE = INPUT_DIR / "keyframe.csv"         # fallback: keyframes.csv
FRAME_ROOT = INPUT_DIR / "keyframes"

VIDEO_START = 0
VIDEO_END: int | None = None                       # half-open, sorted video_id
OCR_TEMPORAL_GAP_MS = 8_000
OCR_MAX_CANDIDATES_PER_SHOT = 4
OCR_SIGLIP_DIVERSE_CANDIDATES = 1
FRAME_EMBEDDING_ARTIFACT_ROOT: Path | None = None  # .../<run>/ containing metadata/, vectors/

DETECTION_BATCH_SIZE = 32
RECOGNITION_BATCH_SIZE = 128
MIN_DETECTION_BATCH_SIZE = 2
MIN_RECOGNITION_BATCH_SIZE = 8
FRAME_PROCESS_CHUNK_SIZE = 64
DETECTION_BOX_THRESHOLD = 0.50
RECOGNITION_SCORE_THRESHOLD = 0.45
CROP_PADDING_RATIO = 0.04
MINIMUM_CROP_SIDE = 3
OCR_DEDUP_WINDOW_MS = 3_000
OCR_DEDUP_IOU_THRESHOLD = 0.50

DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"
RECOGNITION_BACKEND = "vietocr"                    # vietocr | paddleocr
VIETOCR_MODEL_NAME = "vgg_transformer"
PADDLE_RECOGNITION_MODEL_NAME = "latin_PP-OCRv5_mobile_rec"
LANGUAGE = "vi"
RUN_ID = time.strftime("ocr-%Y%m%d-%H%M%S")
RUN_PIPELINE = True                                 # set False only for a dry candidate-plan review
CONTINUE_ON_ERROR = True

def csv_fallback(path: Path, alternative: str) -> Path:
    return path if path.is_file() else path.with_name(alternative)

SHOT_FILE = csv_fallback(SHOT_FILE, "shots.csv")
KEYFRAME_FILE = csv_fallback(KEYFRAME_FILE, "keyframes.csv")
if not SHOT_FILE.is_file() or not KEYFRAME_FILE.is_file():
    raise FileNotFoundError(f"Need shot/keyframe CSV: {SHOT_FILE}; {KEYFRAME_FILE}")
if not FRAME_ROOT.is_dir(): raise FileNotFoundError(f"FRAME_ROOT does not exist: {FRAME_ROOT}")
if OCR_TEMPORAL_GAP_MS <= 0 or OCR_MAX_CANDIDATES_PER_SHOT < 1 or OCR_SIGLIP_DIVERSE_CANDIDATES < 0:
    raise ValueError("Invalid candidate settings")
if RECOGNITION_BACKEND not in {"vietocr", "paddleocr"}: raise ValueError("Invalid recognition backend")
RUN_DIR = OUTPUT_DIR / RUN_ID
if RUN_DIR.exists(): raise FileExistsError(f"Refusing to overwrite: {RUN_DIR}")
'''),
cell("markdown", """## Candidate selection

Each shot receives at least one temporal anchor. Long shots are covered with evenly-spaced anchors, capped by OCR_MAX_CANDIDATES_PER_SHOT. If a Frame Embedding artifact is supplied, the notebook adds up to OCR_SIGLIP_DIVERSE_CANDIDATES frames by farthest-first cosine distance without replacing temporal anchors.

An official frame whose shot_id is empty is assigned by timestamp to its containing shot. Images are resolved only by absolute path, FRAME_ROOT/frame_path, or FRAME_ROOT/basename; there is no R2 download or recursive guessing.
"""),
cell("code", r'''# Input validation, shot assignment, image resolver, and deterministic candidate plan.
SHOT_COLUMNS = {"shot_id", "video_id", "shot_index", "start_ms", "end_ms"}
FRAME_COLUMNS = {"frame_id", "video_id", "shot_id", "timestamp_ms", "fps", "frame_idx", "source", "frame_path", "width", "height"}
CANDIDATE_COLUMNS = ["frame_id", "video_id", "assigned_shot_id", "timestamp_ms", "frame_idx", "source", "frame_path", "resolved_path", "selection_reason"]

def optional(value: Any) -> str | None:
    if value is None or pd.isna(value): return None
    result = str(value).strip()
    return result if result and result.lower() not in {"none", "null", "nan"} else None

def resolve_image(value: Any) -> Path | None:
    raw = optional(value)
    if raw is None: return None
    path = Path(raw).expanduser()
    choices = ([path] if path.is_absolute() else []) + [FRAME_ROOT / path, FRAME_ROOT / path.name]
    return next((item.resolve() for item in choices if item.is_file()), None)

def load_input() -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = pd.read_csv(SHOT_FILE, dtype={"shot_id": "string", "video_id": "string"})
    frames = pd.read_csv(KEYFRAME_FILE, dtype={"frame_id": "string", "video_id": "string", "shot_id": "string", "source": "string", "frame_path": "string"})
    absent = SHOT_COLUMNS - set(shots), FRAME_COLUMNS - set(frames)
    if absent[0] or absent[1]: raise ValueError(f"CSV schema missing shots={sorted(absent[0])}, frames={sorted(absent[1])}")
    for key in ("shot_index", "start_ms", "end_ms"): shots[key] = pd.to_numeric(shots[key], errors="raise").astype("int64")
    for key in ("timestamp_ms", "frame_idx"): frames[key] = pd.to_numeric(frames[key], errors="raise").astype("int64")
    frames["fps"] = pd.to_numeric(frames["fps"], errors="raise")
    if shots.duplicated("shot_id").any() or frames.duplicated("frame_id").any(): raise ValueError("shot_id/frame_id must be unique")
    if shots[["shot_id","video_id"]].isna().any().any() or frames[["frame_id","video_id"]].isna().any().any(): raise ValueError("IDs cannot be null")
    if (shots.end_ms <= shots.start_ms).any() or (frames.timestamp_ms < 0).any() or (frames.frame_idx < 0).any() or (frames.fps <= 0).any(): raise ValueError("Invalid temporal/frame values")
    if not set(frames.source.astype(str)).issubset({"official","extracted"}): raise ValueError("source must be official/extracted")
    for key in ("shot_id","video_id"): shots[key] = shots[key].astype(str).str.strip()
    for key in ("frame_id","video_id","source","frame_path"): frames[key] = frames[key].astype(str).str.strip()
    frames["shot_id"] = frames["shot_id"].map(optional)
    return shots.sort_values(["video_id","start_ms","shot_index","shot_id"],kind="stable"), frames

def assign_shots(shots: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str,Any]]]:
    by_id = shots.set_index("shot_id")[["video_id","start_ms","end_ms"]].to_dict("index")
    by_video = {video: data.sort_values(["start_ms","end_ms"]) for video,data in shots.groupby("video_id",sort=False)}
    assigned, failures = [], []
    for row in frames.to_dict("records"):
        existing = optional(row["shot_id"])
        if existing in by_id and by_id[existing]["video_id"] == row["video_id"]:
            assigned.append(existing); continue
        choices = by_video.get(row["video_id"])
        matching = choices[(choices.start_ms <= row["timestamp_ms"]) & (row["timestamp_ms"] < choices.end_ms)] if choices is not None else pd.DataFrame()
        if len(matching) == 1: assigned.append(str(matching.iloc[0].shot_id))
        else:
            assigned.append(None); failures.append({"frame_id":str(row["frame_id"]),"reason":"no_unambiguous_containing_shot"})
    result = frames.copy(); result["assigned_shot_id"] = assigned
    return result, failures

def embedding_vectors(root: Path | None) -> dict[str,np.ndarray]:
    if root is None: return {}
    if not (root/"metadata").is_dir() or not (root/"vectors").is_dir():
        raise FileNotFoundError("FRAME_EMBEDDING_ARTIFACT_ROOT needs metadata/ and vectors/")
    output = {}
    for meta_path in sorted((root/"metadata").glob("part-*.parquet")):
        metadata = pd.read_parquet(meta_path,columns=["frame_id","vector_shard","vector_row"])
        for shard, group in metadata.groupby("vector_shard",sort=False):
            matrix = np.load(root/"vectors"/str(shard),mmap_mode="r")
            for row in group.itertuples(index=False):
                vector=np.asarray(matrix[int(row.vector_row)],dtype=np.float32); norm=float(np.linalg.norm(vector))
                if norm > 0 and np.isfinite(vector).all(): output[str(row.frame_id)] = vector/norm
    return output

def temporal_anchors(group: pd.DataFrame, shot: pd.Series) -> list[int]:
    count=min(len(group),OCR_MAX_CANDIDATES_PER_SHOT,max(1,math.ceil((shot.end_ms-shot.start_ms)/OCR_TEMPORAL_GAP_MS)))
    selected=[]
    for order in range(count):
        target=shot.start_ms+(shot.end_ms-shot.start_ms)*(order+.5)/count
        ranked=group.assign(_d=(group.timestamp_ms-target).abs(),_source=(group.source!="extracted").astype(int)).sort_values(["_d","_source","frame_idx","frame_id"],kind="stable")
        selected.append(int(next(index for index in ranked.index if index not in selected)))
    return selected

def add_diversity(group: pd.DataFrame, selected: list[int], vectors: dict[str,np.ndarray]) -> list[int]:
    seeds=[vectors.get(str(group.loc[index].frame_id)) for index in selected]
    seeds=[item for item in seeds if item is not None]
    additions=0
    while vectors and additions < OCR_SIGLIP_DIVERSE_CANDIDATES and len(selected) < OCR_MAX_CANDIDATES_PER_SHOT:
        pool=[(int(index),vectors.get(str(row.frame_id))) for index,row in group.iterrows() if index not in selected]
        pool=[item for item in pool if item[1] is not None]
        if not pool: break
        index,vector=max(pool,key=lambda item: float(np.min(1-np.asarray(seeds)@item[1])) if seeds else 0.0)
        selected.append(index); seeds.append(vector); additions+=1
    return selected

def candidate_plan(shots: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame,list[dict[str,Any]]]:
    frames, failures=assign_shots(shots,frames)
    selected_videos=sorted(frames.video_id.unique().tolist())[VIDEO_START:VIDEO_END]
    frames=frames[frames.video_id.isin(selected_videos)&frames.assigned_shot_id.notna()].copy()
    vectors,rows,shot_by_id=embedding_vectors(FRAME_EMBEDDING_ARTIFACT_ROOT),[],shots.set_index("shot_id")
    for shot_id,group in frames.groupby("assigned_shot_id",sort=True):
        group=group.sort_values(["timestamp_ms","frame_idx","frame_id"],kind="stable")
        anchors=temporal_anchors(group,shot_by_id.loc[shot_id]); selected=add_diversity(group,anchors.copy(),vectors)
        for index in selected:
            row=group.loc[index]; image=resolve_image(row.frame_path)
            if image is None:
                failures.append({"frame_id":str(row.frame_id),"reason":"image_missing","path":str(row.frame_path)}); continue
            rows.append({"frame_id":str(row.frame_id),"video_id":str(row.video_id),"assigned_shot_id":str(shot_id),"timestamp_ms":int(row.timestamp_ms),"frame_idx":int(row.frame_idx),"source":str(row.source),"frame_path":str(row.frame_path),"resolved_path":str(image),"selection_reason":"temporal_anchor" if index in anchors else "siglip_diverse"})
    output=pd.DataFrame(rows,columns=CANDIDATE_COLUMNS).drop_duplicates("frame_id").sort_values(["video_id","timestamp_ms","frame_idx","frame_id"],kind="stable")
    return output.reset_index(drop=True), failures

SHOTS,FRAMES=load_input()
CANDIDATES,PLAN_FAILURES=candidate_plan(SHOTS,FRAMES)
print(f"Frames={len(FRAMES):,}; candidates={len(CANDIDATES):,}; reduction={1-len(CANDIDATES)/max(1,len(FRAMES)):.1%}; plan failures={len(PLAN_FAILURES):,}")
display(CANDIDATES.head())
'''),
cell("code", r'''# Pure selection regression test; no model, database, or image input is touched.
def selection_smoke_test() -> None:
    old=OCR_TEMPORAL_GAP_MS,OCR_MAX_CANDIDATES_PER_SHOT,OCR_SIGLIP_DIVERSE_CANDIDATES
    try:
        globals()["OCR_TEMPORAL_GAP_MS"],globals()["OCR_MAX_CANDIDATES_PER_SHOT"],globals()["OCR_SIGLIP_DIVERSE_CANDIDATES"]=3000,3,1
        frames=pd.DataFrame({"frame_id":["a","b","c","d"],"timestamp_ms":[500,2500,5500,8500],"frame_idx":[1,2,3,4],"source":["official","extracted","official","extracted"]},index=[10,20,30,40])
        anchors=temporal_anchors(frames,pd.Series({"start_ms":0,"end_ms":9000}))
        vectors={"a":np.array([1,0],np.float32),"b":np.array([1,0],np.float32),"c":np.array([0,1],np.float32),"d":np.array([0,1],np.float32)}
        assert len(anchors)==3 and len(set(anchors))==3
        assert len(add_diversity(frames,anchors[:1],vectors)) <= 3
    finally:
        globals()["OCR_TEMPORAL_GAP_MS"],globals()["OCR_MAX_CANDIDATES_PER_SHOT"],globals()["OCR_SIGLIP_DIVERSE_CANDIDATES"]=old
selection_smoke_test()
print("Candidate-selection smoke test passed.")
'''),
cell("markdown", """## GPU OCR and result deduplication

Frames are decoded only in bounded chunks. Detection batches frames; recognition batches rectified crops. A true CUDA out-of-memory condition retries the failed operation with a smaller batch, while all other failures remain visible.

After inference, equal normalized text in the same shot is removed only if its boxes overlap and timestamps are close. Numbers n are re-assigned per frame after dedup so the existing primary key (frame_id, n) remains valid.
"""),
cell("code", r'''# GPU engine, geometry helpers, and OOM retry.
def text_norm(value: str) -> str: return " ".join(unicodedata.normalize("NFC",str(value)).split())

def polygon4(value: np.ndarray, width: int, height: int) -> np.ndarray:
    points=np.asarray(value,dtype=np.float32).reshape(-1,2)
    if len(points)<4 or not np.isfinite(points).all(): raise ValueError("invalid polygon")
    if len(points)!=4: points=cv2.boxPoints(cv2.minAreaRect(points))
    points[:,0]=np.clip(points[:,0],0,width-1); points[:,1]=np.clip(points[:,1],0,height-1)
    sums,diffs=points.sum(1),np.diff(points,axis=1).ravel()
    output=np.asarray([points[np.argmin(sums)],points[np.argmin(diffs)],points[np.argmax(sums)],points[np.argmax(diffs)]],np.float32)
    if len(np.unique(output,axis=0))!=4: raise ValueError("collapsed polygon")
    return output

def crop_text(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    height,width=image.shape[:2]; polygon=polygon4(polygon,width,height); center=polygon.mean(0,keepdims=True)
    polygon=polygon4(center+(polygon-center)*(1+2*CROP_PADDING_RATIO),width,height)
    target_w=int(round(max(np.linalg.norm(polygon[1]-polygon[0]),np.linalg.norm(polygon[2]-polygon[3]))))
    target_h=int(round(max(np.linalg.norm(polygon[3]-polygon[0]),np.linalg.norm(polygon[2]-polygon[1]))))
    if target_w<MINIMUM_CROP_SIDE or target_h<MINIMUM_CROP_SIDE: raise ValueError("small crop")
    destination=np.asarray([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]],np.float32)
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(polygon,destination),(target_w,target_h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def normalized_box(polygon: np.ndarray, width: int, height: int) -> tuple[float,float,float,float]:
    polygon=polygon4(polygon,width,height)
    return float(polygon[:,0].min()/width),float(polygon[:,0].max()/width),float(polygon[:,1].min()/height),float(polygon[:,1].max()/height)

def is_oom(error: BaseException) -> bool:
    message=str(error).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)

class OCREngine:
    def __init__(self) -> None:
        from paddleocr import TextDetection, TextRecognition
        self.detector=TextDetection(model_name=DETECTION_MODEL_NAME,device="gpu:0",limit_side_len=1280,limit_type="max",thresh=.3,box_thresh=DETECTION_BOX_THRESHOLD,unclip_ratio=1.6)
        self.backend=RECOGNITION_BACKEND
        if self.backend=="vietocr":
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
            config=Cfg.load_config_from_name(VIETOCR_MODEL_NAME); config["device"]="cuda:0"
            self.recognizer=Predictor(config); self.recognizer.model.eval()
        else: self.recognizer=TextRecognition(model_name=PADDLE_RECOGNITION_MODEL_NAME,device="gpu:0")

    def detect(self,images: list[np.ndarray],batch_size: int) -> list[list[tuple[np.ndarray,float]]]:
        output=[]
        for item in self.detector.predict(images,batch_size=batch_size):
            polygons,scores=list(item.get("dt_polys",[])),list(item.get("dt_scores",[]))
            if len(polygons)!=len(scores): raise RuntimeError("Paddle detection polygon/score mismatch")
            output.append([(np.asarray(poly,np.float32),float(score)) for poly,score in zip(polygons,scores)])
        if len(output)!=len(images): raise RuntimeError("Paddle detection count/order mismatch")
        return output

    def recognize(self,crops: list[np.ndarray],batch_size: int) -> list[tuple[str,float]]:
        if not crops: return []
        if self.backend=="vietocr":
            images=[Image.fromarray(item[:,:,::-1]) for item in crops]; output=[]
            with torch.inference_mode():
                for start in range(0,len(images),batch_size):
                    texts,scores=self.recognizer.predict_batch(images[start:start+batch_size],return_prob=True)
                    output.extend((str(text),float(score)) for text,score in zip(texts,scores))
            return output
        return [(str(item.get("rec_text","")),float(item.get("rec_score",0))) for item in self.recognizer.predict(crops,batch_size=batch_size)]

    def close(self) -> None:
        for model in (self.detector,self.recognizer):
            close=getattr(model,"close",None)
            if callable(close): close()

def adaptive(call,values: list[Any],initial: int,minimum: int) -> tuple[list[Any],int]:
    batch=initial
    while True:
        try:
            output=[]
            for start in range(0,len(values),batch): output.extend(call(values[start:start+batch],batch))
            return output,batch
        except RuntimeError as error:
            if not is_oom(error) or batch<=minimum: raise
            batch=max(minimum,batch//2); gc.collect(); torch.cuda.empty_cache()
            print("CUDA OOM: retrying operation with batch",batch)
'''),
cell("code", r'''# Bounded inference, spatial-temporal dedup, CSV/SQL artifacts, and tests.
DB_COLUMNS=["frame_id","n","text","language","x_min","x_max","y_min","y_max"]

def infer_chunk(engine: OCREngine,records: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]],int,int]:
    images,valid,failures=[],[],[]
    for item in records:
        image=cv2.imread(item["resolved_path"])
        if image is None: failures.append({"frame_id":item["frame_id"],"reason":"image_unreadable","path":item["resolved_path"]})
        else: images.append(image); valid.append(item)
    if not images: return [],failures,DETECTION_BATCH_SIZE,RECOGNITION_BATCH_SIZE
    detected,det_batch=adaptive(engine.detect,images,DETECTION_BATCH_SIZE,MIN_DETECTION_BATCH_SIZE)
    crops,metadata=[],[]
    for index,(image,regions) in enumerate(zip(images,detected)):
        height,width=image.shape[:2]; regions=[(poly,score) for poly,score in regions if score>=DETECTION_BOX_THRESHOLD]
        for polygon,_ in sorted(regions,key=lambda item:float(item[0][:,1].mean())):
            try: crops.append(crop_text(image,polygon)); metadata.append((index,polygon,width,height))
            except ValueError: pass
    recognized,rec_batch=adaptive(engine.recognize,crops,RECOGNITION_BATCH_SIZE,MIN_RECOGNITION_BATCH_SIZE)
    rows=[]; count=defaultdict(int)
    for (index,polygon,width,height),(text,score) in zip(metadata,recognized):
        text=text_norm(text)
        if not text or score<RECOGNITION_SCORE_THRESHOLD: continue
        item=valid[index]; x_min,x_max,y_min,y_max=normalized_box(polygon,width,height); frame_id=item["frame_id"]
        rows.append({"frame_id":frame_id,"n":count[frame_id],"text":text,"language":LANGUAGE,"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max,"video_id":item["video_id"],"shot_id":item["assigned_shot_id"],"timestamp_ms":int(item["timestamp_ms"]),"recognition_confidence":score})
        count[frame_id]+=1
    return rows,failures,det_batch,rec_batch

def iou(a: dict[str,Any],b: dict[str,Any]) -> float:
    overlap=max(0,min(a["x_max"],b["x_max"])-max(a["x_min"],b["x_min"]))*max(0,min(a["y_max"],b["y_max"])-max(a["y_min"],b["y_min"]))
    union=(a["x_max"]-a["x_min"])*(a["y_max"]-a["y_min"])+(b["x_max"]-b["x_min"])*(b["y_max"]-b["y_min"])-overlap
    return overlap/union if union else 0.0

def deduplicate(rows: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    kept,dropped,recent=[],[],defaultdict(list)
    for row in sorted(rows,key=lambda item:(item["video_id"],item["shot_id"],item["timestamp_ms"],item["frame_id"],item["n"])):
        key=(row["shot_id"],text_norm(row["text"]).casefold())
        recent[key]=[old for old in recent[key] if row["timestamp_ms"]-old["timestamp_ms"]<=OCR_DEDUP_WINDOW_MS]
        if any(iou(row,old)>=OCR_DEDUP_IOU_THRESHOLD for old in recent[key]):
            dropped.append({**row,"reason":"same_text_similar_box_within_temporal_window"}); continue
        recent[key].append(row); kept.append(row)
    indices=defaultdict(int)
    for row in kept: row["n"]=indices[row["frame_id"]]; indices[row["frame_id"]]+=1
    return kept,dropped

def sql_literal(value: Any) -> str:
    if value is None or (isinstance(value,float) and math.isnan(value)): return "NULL"
    if isinstance(value,(int,np.integer)): return str(int(value))
    if isinstance(value,(float,np.floating)): return format(float(value),".9g")
    return "'" + str(value).replace("'","''") + "'"

def write_jsonl(path: Path,items: Iterable[dict[str,Any]]) -> None:
    with path.open("w",encoding="utf-8") as handle:
        for item in items: handle.write(json.dumps(item,ensure_ascii=False)+"\n")

def write_output(raw: list[dict[str,Any]],failures: list[dict[str,Any]],started: float,det_batch: int,rec_batch: int) -> Path:
    temporary=RUN_DIR.with_name(RUN_DIR.name+".partial")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    kept,dropped=deduplicate(raw)
    output=pd.DataFrame(kept,columns=[*DB_COLUMNS,"video_id","shot_id","timestamp_ms","recognition_confidence"])
    CANDIDATES.to_csv(temporary/"ocr_candidates.csv",index=False)
    output.to_csv(temporary/"ocr.csv",index=False)
    pd.DataFrame(dropped).to_csv(temporary/"ocr_deduplicated.csv",index=False)
    write_jsonl(temporary/"failures.jsonl",failures)
    with (temporary/"insert_ocr.sql").open("w",encoding="utf-8") as handle:
        handle.write("BEGIN;\n")
        for row in output[DB_COLUMNS].to_dict("records"):
            values=", ".join(sql_literal(row[key]) for key in DB_COLUMNS)
            handle.write(f"INSERT INTO ocr (frame_id, n, text, language, x_min, x_max, y_min, y_max) VALUES ({values}) ON CONFLICT (frame_id, n) DO NOTHING;\n")
        handle.write("COMMIT;\n")
    elapsed=max(time.monotonic()-started,.001)
    summary={"run_id":RUN_ID,"detection_model":DETECTION_MODEL_NAME,"recognition_backend":RECOGNITION_BACKEND,"recognition_model":VIETOCR_MODEL_NAME if RECOGNITION_BACKEND=="vietocr" else PADDLE_RECOGNITION_MODEL_NAME,"gpu":torch.cuda.get_device_name(0),"paddle_version":paddle.__version__,"torch_version":torch.__version__,"input_frames":len(FRAMES),"candidate_frames":len(CANDIDATES),"raw_ocr_rows":len(raw),"kept_ocr_rows":len(output),"deduplicated_ocr_rows":len(dropped),"failures":len(failures),"detection_batch_final":det_batch,"recognition_batch_final":rec_batch,"elapsed_seconds":round(elapsed,3),"candidate_frames_per_second":round(len(CANDIDATES)/elapsed,3),"peak_vram_bytes":int(torch.cuda.max_memory_allocated())}
    (temporary/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.rename(RUN_DIR)
    return RUN_DIR

# Dedup regression test: same text/box near in time is dropped, later text is preserved.
row={"frame_id":"f1","n":0,"text":"Khuyến mãi","language":"vi","x_min":.1,"x_max":.5,"y_min":.1,"y_max":.2,"video_id":"v","shot_id":"s","recognition_confidence":.9}
kept,dropped=deduplicate([{**row,"timestamp_ms":100},{**row,"frame_id":"f2","timestamp_ms":500},{**row,"frame_id":"f3","timestamp_ms":5000}])
assert len(kept)==2 and len(dropped)==1 and [item["n"] for item in kept]==[0,0]
print("Post-OCR dedup smoke test passed.")
'''),
cell("code", r'''# Real run. Set RUN_PIPELINE=False in the configuration cell for a dry candidate-plan review.
if RUN_PIPELINE:
    started=time.monotonic()
    raw,failures=[],list(PLAN_FAILURES)
    final_det,final_rec=DETECTION_BATCH_SIZE,RECOGNITION_BATCH_SIZE
    engine=OCREngine()
    try:
        records=CANDIDATES.to_dict("records")
        for start in range(0,len(records),FRAME_PROCESS_CHUNK_SIZE):
            chunk=records[start:start+FRAME_PROCESS_CHUNK_SIZE]
            try:
                rows,chunk_failures,final_det,final_rec=infer_chunk(engine,chunk)
                raw.extend(rows); failures.extend(chunk_failures)
                print(f"Processed {start+len(chunk):,}/{len(records):,}; raw OCR rows={len(raw):,}")
            except Exception as error:
                if not CONTINUE_ON_ERROR: raise
                failures.append({"reason":"chunk_failed","frame_ids":[item["frame_id"] for item in chunk],"error":str(error),"traceback":traceback.format_exc(limit=3)})
        output=write_output(raw,failures,started,final_det,final_rec)
        print("Completed. Download Kaggle Output directory:",output)
        display(pd.read_csv(output/"ocr.csv").head())
    finally:
        engine.close()
else:
    print("Dry plan only. Set RUN_PIPELINE=True to execute GPU OCR.")
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}")
