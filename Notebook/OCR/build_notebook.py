"""Build the Kaggle OCR notebook."""
from __future__ import annotations
import json
from pathlib import Path

OUTPUT = Path(__file__).with_name("ocr_kaggle.ipynb")
def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code": value.update({"execution_count": None, "outputs": []})
    return value

cells = [
cell("markdown", """# OCR - shot-aware, resumable GPU preprocessing

The notebook reads Kaggle input and writes under `/kaggle/working`. It does not connect directly to PostgreSQL or upload data. Missing selected images may be downloaded through `image_url`, including Cloudflare R2 URLs. It selects candidates once, checkpoints successful frames continuously, then writes the existing CSV/SQL/failure artifacts.
"""),
cell("code", r'''# Install only genuinely missing OCR packages; restart the kernel if this installs anything.
import importlib.util, re, subprocess, sys
INSTALL_MISSING_PACKAGES = True
# Kaggle normally includes torch, numpy, OpenCV and pyarrow. Never reinstall them here.
# Paddle GPU 3.x wheels are not published on PyPI; they must use Paddle's CUDA wheel index.
PADDLE_VERSION = "3.3.0"
def paddle_cuda_index() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
        if not match: raise RuntimeError("CUDA version was not reported by nvidia-smi")
        major, minor = map(int, match.groups())
        if (major, minor) >= (12, 9): return "cu129"
        if (major, minor) >= (12, 6): return "cu126"
        if (major, minor) >= (11, 8): return "cu118"
        raise RuntimeError(f"Unsupported CUDA {major}.{minor}; Paddle GPU requires CUDA 11.8 or newer")
    except FileNotFoundError as error:
        raise RuntimeError("nvidia-smi is unavailable. Enable Kaggle GPU before installing Paddle GPU.") from error

missing = [module for module in ("paddle", "paddleocr", "vietocr") if importlib.util.find_spec(module) is None]
if missing and not INSTALL_MISSING_PACKAGES: raise RuntimeError(f"Missing {missing}; enable installation then restart.")
installed = []
if "paddle" in missing:
    index = f"https://www.paddlepaddle.org.cn/packages/stable/{paddle_cuda_index()}/"
    command = [sys.executable, "-m", "pip", "install", "--quiet", f"paddlepaddle-gpu=={PADDLE_VERSION}", "-i", index]
    print("Installing Paddle GPU from", index)
    subprocess.check_call(command); installed.append("paddle")
other = ["paddleocr>=3.0,<4" for item in missing if item == "paddleocr"] + ["vietocr" for item in missing if item == "vietocr"]
if other:
    print("Installing", ", ".join(other))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *other]); installed.extend(missing)
print("Installed", installed, "; restart the kernel, then Run All." if installed else "OCR dependencies already available; nothing was reinstalled.")
'''),
cell("code", r'''# Imports and GPU compatibility verification.
from __future__ import annotations
import gc, json, math, os, shutil, threading, time, traceback, unicodedata, urllib.error, urllib.parse, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
import cv2
import numpy as np
import pandas as pd
import paddle
import torch
from PIL import Image
from tqdm.auto import tqdm
print("torch:", torch.__version__); print("torch CUDA available:", torch.cuda.is_available()); print("paddle:", paddle.__version__); print("paddle compiled with CUDA:", paddle.device.is_compiled_with_cuda())
if not (paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0): raise RuntimeError("Paddle cannot access CUDA. Enable Kaggle GPU, install a compatible Paddle GPU package, restart.")
if not torch.cuda.is_available(): raise RuntimeError("VietOCR cannot access CUDA. Enable Kaggle GPU and restart.")
paddle.set_device("gpu:0"); torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
print("GPU:", torch.cuda.get_device_name(0), "| Torch CUDA:", torch.version.cuda)
'''),
cell("code", r'''# Configuration. Keep RUN_ID unchanged to resume an interrupted run.
INPUT_DIR = Path("/kaggle/input/btc-ocr-input"); OUTPUT_DIR = Path("/kaggle/working/ocr_output")
SHOT_FILE = INPUT_DIR / "shot.csv"; KEYFRAME_FILE = INPUT_DIR / "keyframe.csv"; FRAME_ROOT = INPUT_DIR / "keyframes"
VIDEO_START = 0; VIDEO_END: int | None = None
OCR_TEMPORAL_GAP_MS = 8_000; OCR_MAX_CANDIDATES_PER_SHOT = 4; OCR_SIGLIP_DIVERSE_CANDIDATES = 1; FRAME_EMBEDDING_ARTIFACT_ROOT: Path | None = None
DETECTION_BATCH_SIZE = 32; RECOGNITION_BATCH_SIZE = 128; MIN_DETECTION_BATCH_SIZE = 2; MIN_RECOGNITION_BATCH_SIZE = 8; FRAME_PROCESS_CHUNK_SIZE = 64
DETECTION_BOX_THRESHOLD = .50; RECOGNITION_SCORE_THRESHOLD = .45; CROP_PADDING_RATIO = .04; MINIMUM_CROP_SIDE = 3; OCR_DEDUP_WINDOW_MS = 3_000; OCR_DEDUP_IOU_THRESHOLD = .50
DETECTION_MODEL_NAME = "PP-OCRv5_mobile_det"; RECOGNITION_BACKEND = "vietocr"; VIETOCR_MODEL_NAME = "vgg_transformer"; PADDLE_RECOGNITION_MODEL_NAME = "latin_PP-OCRv5_mobile_rec"; LANGUAGE = "vi"
RUN_ID = "ocr-run"; RUN_PIPELINE = True; CONTINUE_ON_ERROR = True; MAX_WORKING_BYTES = 15 * 1024 ** 3; MAX_DOWNLOAD_WORKERS = 6; DOWNLOAD_RETRIES = 3; DOWNLOAD_TIMEOUT_SECONDS = 120
def fallback(path: Path, name: str) -> Path: return path if path.is_file() else path.with_name(name)
SHOT_FILE, KEYFRAME_FILE = fallback(SHOT_FILE, "shots.csv"), fallback(KEYFRAME_FILE, "keyframes.csv")
if not SHOT_FILE.is_file() or not KEYFRAME_FILE.is_file(): raise FileNotFoundError(f"Need shot/keyframe CSV: {SHOT_FILE}; {KEYFRAME_FILE}")
if RECOGNITION_BACKEND not in {"vietocr", "paddleocr"}: raise ValueError("Invalid recognition backend")
RUN_DIR = OUTPUT_DIR / RUN_ID; CHECKPOINT_DIR = OUTPUT_DIR / f"{RUN_ID}.checkpoint"; IMAGE_CACHE_DIR = CHECKPOINT_DIR / "image_cache"
if RUN_DIR.exists(): raise FileExistsError(f"Final output exists: {RUN_DIR}; choose a new RUN_ID for a new run.")
'''),
cell("markdown", """## Candidate selection

Existing valid `shot_id` values are retained. Empty/invalid IDs are assigned by a vectorized binary search per video, preserving `start_ms <= timestamp_ms < end_ms`. Candidate selection executes exactly once; unresolved selected images remain candidates for the URL fallback stage.
"""),
cell("code", r'''# [1/6] Load metadata; [2/6] assign shots efficiently; [3/6] select candidates once.
SHOT_COLUMNS = {"shot_id","video_id","shot_index","start_ms","end_ms"}; FRAME_COLUMNS = {"frame_id","video_id","shot_id","timestamp_ms","fps","frame_idx","source","frame_path","width","height"}
CANDIDATE_COLUMNS = ["frame_id","video_id","assigned_shot_id","timestamp_ms","frame_idx","source","frame_path","resolved_path","selection_reason"]
def optional(value: Any) -> str | None:
    if value is None or pd.isna(value): return None
    value = str(value).strip(); return value if value and value.lower() not in {"none","null","nan"} else None
def resolve_image(value: Any) -> Path | None:
    raw = optional(value)
    if raw is None: return None
    path = Path(raw).expanduser(); choices = ([path] if path.is_absolute() else []) + [FRAME_ROOT / path, FRAME_ROOT / path.name]
    return next((item.resolve() for item in choices if item.is_file()), None)
def load_input() -> tuple[pd.DataFrame,pd.DataFrame]:
    shots = pd.read_csv(SHOT_FILE, dtype={"shot_id":"string","video_id":"string"}); frames = pd.read_csv(KEYFRAME_FILE, dtype={"frame_id":"string","video_id":"string","shot_id":"string","source":"string","frame_path":"string"})
    absent = SHOT_COLUMNS-set(shots), FRAME_COLUMNS-set(frames)
    if absent[0] or absent[1]: raise ValueError(f"CSV schema missing shots={sorted(absent[0])}, frames={sorted(absent[1])}")
    for key in ("shot_index","start_ms","end_ms"): shots[key] = pd.to_numeric(shots[key], errors="raise").astype("int64")
    for key in ("timestamp_ms","frame_idx"): frames[key] = pd.to_numeric(frames[key], errors="raise").astype("int64")
    frames["fps"] = pd.to_numeric(frames["fps"], errors="raise")
    if shots.duplicated("shot_id").any() or frames.duplicated("frame_id").any(): raise ValueError("shot_id/frame_id must be unique")
    if (shots.end_ms<=shots.start_ms).any() or (frames.timestamp_ms<0).any() or (frames.frame_idx<0).any() or (frames.fps<=0).any(): raise ValueError("Invalid temporal/frame values")
    if not set(frames.source.astype(str)).issubset({"official","extracted"}): raise ValueError("source must be official/extracted")
    for key in ("shot_id","video_id"): shots[key] = shots[key].astype(str).str.strip()
    for key in ("frame_id","video_id","source","frame_path"): frames[key] = frames[key].astype(str).str.strip()
    frames["shot_id"] = frames.shot_id.map(optional)
    return shots.sort_values(["video_id","start_ms","shot_index","shot_id"], kind="stable"), frames
def assign_shots(shots: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame,list[dict[str,Any]]]:
    result = frames.copy(); lookup = shots.set_index("shot_id").video_id; valid = result.shot_id.notna() & result.shot_id.map(lookup).eq(result.video_id); result["assigned_shot_id"] = pd.NA; result.loc[valid,"assigned_shot_id"] = result.loc[valid,"shot_id"]
    pending = result.loc[~valid,["frame_id","video_id","timestamp_ms"]]
    for video_id, group in pending.groupby("video_id", sort=False):
        source = shots.loc[shots.video_id.eq(video_id)].sort_values("start_ms",kind="stable")
        if source.empty: continue
        timestamps = group.timestamp_ms.to_numpy(); starts, ends = source.start_ms.to_numpy(), source.end_ms.to_numpy()
        pos = np.searchsorted(starts,timestamps,side="right")-1; active = np.searchsorted(starts,timestamps,side="right")-np.searchsorted(np.sort(ends),timestamps,side="right")
        okay = (pos>=0) & (active==1); candidate = np.full(len(group),None,dtype=object); candidate[okay] = source.shot_id.to_numpy()[pos[okay]]
        candidate[(pos>=0) & (ends[np.maximum(pos,0)]<=timestamps)] = None
        result.loc[group.index,"assigned_shot_id"] = candidate
    failures = [{"frame_id":str(frame_id),"reason":"no_unambiguous_containing_shot"} for frame_id in result.loc[result.assigned_shot_id.isna(),"frame_id"]]
    return result, failures
def embedding_vectors(root: Path | None) -> dict[str,np.ndarray]:
    if root is None: return {}
    if not (root/"metadata").is_dir() or not (root/"vectors").is_dir(): raise FileNotFoundError("FRAME_EMBEDDING_ARTIFACT_ROOT needs metadata/ and vectors/")
    output = {}
    for meta_path in sorted((root/"metadata").glob("part-*.parquet")):
        metadata = pd.read_parquet(meta_path,columns=["frame_id","vector_shard","vector_row"])
        for shard, group in metadata.groupby("vector_shard",sort=False):
            matrix=np.load(root/"vectors"/str(shard),mmap_mode="r")
            for row in group.itertuples(index=False):
                vector=np.asarray(matrix[int(row.vector_row)],dtype=np.float32); norm=float(np.linalg.norm(vector))
                if norm>0 and np.isfinite(vector).all(): output[str(row.frame_id)] = vector/norm
    return output
def temporal_anchors(group: pd.DataFrame, shot: pd.Series) -> list[int]:
    count=min(len(group),OCR_MAX_CANDIDATES_PER_SHOT,max(1,math.ceil((shot.end_ms-shot.start_ms)/OCR_TEMPORAL_GAP_MS))); selected=[]
    for order in range(count):
        target=shot.start_ms+(shot.end_ms-shot.start_ms)*(order+.5)/count; ranked=group.assign(_d=(group.timestamp_ms-target).abs(),_source=(group.source!="extracted").astype(int)).sort_values(["_d","_source","frame_idx","frame_id"],kind="stable"); selected.append(next(index for index in ranked.index if index not in selected))
    return selected
def add_diversity(group: pd.DataFrame, selected: list[int], vectors: dict[str,np.ndarray]) -> list[int]:
    seeds=[vectors[str(group.loc[index].frame_id)] for index in selected if str(group.loc[index].frame_id) in vectors]
    for _ in range(min(OCR_SIGLIP_DIVERSE_CANDIDATES,OCR_MAX_CANDIDATES_PER_SHOT-len(selected))):
        pool=[(index,vectors.get(str(row.frame_id))) for index,row in group.loc[~group.index.isin(selected)].iterrows()]; pool=[item for item in pool if item[1] is not None]
        if not pool: break
        index,vector=max(pool,key=lambda item:float(np.min(1-np.asarray(seeds)@item[1])) if seeds else 0.0); selected.append(index); seeds.append(vector)
    return selected
def candidate_plan(shots: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame,list[dict[str,Any]]]:
    assigned, failures=assign_shots(shots,frames); videos=sorted(assigned.video_id.unique().tolist())[VIDEO_START:VIDEO_END]; assigned=assigned[assigned.video_id.isin(videos)&assigned.assigned_shot_id.notna()].copy(); vectors,rows,shot_by_id=embedding_vectors(FRAME_EMBEDDING_ARTIFACT_ROOT),[],shots.set_index("shot_id")
    for shot_id,group in tqdm(assigned.groupby("assigned_shot_id",sort=True),desc="Selecting shots",unit="shot"):
        group=group.sort_values(["timestamp_ms","frame_idx","frame_id"],kind="stable"); anchors=temporal_anchors(group,shot_by_id.loc[shot_id]); selected=add_diversity(group,anchors.copy(),vectors)
        for index in selected:
            row=group.loc[index]; image=resolve_image(row.frame_path); rows.append({"frame_id":str(row.frame_id),"video_id":str(row.video_id),"assigned_shot_id":str(shot_id),"timestamp_ms":int(row.timestamp_ms),"frame_idx":int(row.frame_idx),"source":str(row.source),"frame_path":str(row.frame_path),"resolved_path":str(image) if image else None,"selection_reason":"temporal_anchor" if index in anchors else "siglip_diverse"})
    return pd.DataFrame(rows,columns=CANDIDATE_COLUMNS).drop_duplicates("frame_id").sort_values(["video_id","timestamp_ms","frame_idx","frame_id"],kind="stable").reset_index(drop=True), failures
stage=time.perf_counter(); print("[1/6] Loading metadata"); SHOTS,FRAMES=load_input(); print(f"Loaded {len(SHOTS):,} shots, {len(FRAMES):,} frames in {time.perf_counter()-stage:.1f}s")
stage=time.perf_counter(); print("[2/6] Assigning shots; [3/6] Selecting OCR candidates"); CANDIDATES,PLAN_FAILURES=candidate_plan(SHOTS,FRAMES); LOCAL_IMAGES=int(CANDIDATES.resolved_path.notna().sum()); print(f"Candidates={len(CANDIDATES):,}; local={LOCAL_IMAGES:,}; need download={len(CANDIDATES)-LOCAL_IMAGES:,}; assignment failures={len(PLAN_FAILURES):,}; {time.perf_counter()-stage:.1f}s"); display(CANDIDATES.head())
'''),
cell("code", r'''# [4/6] Download only unresolved selected candidates; directory size is scanned once, not per MB.
_size_lock=threading.Lock(); WORKING_BYTES=sum(path.stat().st_size for path in OUTPUT_DIR.rglob("*") if path.is_file()) if OUTPUT_DIR.exists() else 0
def reserve_size(amount: int) -> None:
    global WORKING_BYTES
    with _size_lock:
        if WORKING_BYTES+amount>MAX_WORKING_BYTES: raise RuntimeError("Kaggle working output would exceed MAX_WORKING_BYTES")
        WORKING_BYTES+=amount
def remove_cached(path: Path) -> None:
    global WORKING_BYTES
    if path.is_file():
        size=path.stat().st_size; path.unlink()
        with _size_lock: WORKING_BYTES=max(0,WORKING_BYTES-size)
def download_one(row: dict[str,Any]) -> Path:
    url=optional(row.get("image_url"))
    if url is None or not url.startswith(("http://","https://")): raise ValueError("image_url is required for missing local image")
    suffix=Path(urllib.parse.urlsplit(url).path).suffix.lower() or ".jpg"; target=IMAGE_CACHE_DIR/f"{row['frame_id']}{suffix}"
    if target.is_file() and target.stat().st_size>0: return target
    for attempt in range(1,DOWNLOAD_RETRIES+1):
        part=target.with_suffix(target.suffix+".part")
        try:
            with urllib.request.urlopen(url,timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,part.open("wb") as handle:
                while chunk:=response.read(1024*1024): reserve_size(len(chunk)); handle.write(chunk)
            if part.stat().st_size==0: raise RuntimeError("empty image download")
            part.replace(target); return target
        except Exception:
            remove_cached(part)
            if attempt==DOWNLOAD_RETRIES: raise
            time.sleep(attempt)
def download_candidates(candidates: pd.DataFrame,frames: pd.DataFrame) -> tuple[pd.DataFrame,list[dict[str,Any]]]:
    output=candidates.copy(); missing=output[output.resolved_path.isna()].copy(); failures=[]
    if missing.empty: return output,failures
    urls=frames.set_index("frame_id")["image_url"] if "image_url" in frames else pd.Series(dtype="string"); missing["image_url"]=missing.frame_id.map(urls); IMAGE_CACHE_DIR.mkdir(parents=True,exist_ok=True); records=missing.to_dict("records")
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
        futures={pool.submit(download_one,row):row for row in records}
        for future in tqdm(as_completed(futures),total=len(futures),desc="Downloading candidates",unit="image"):
            row=futures[future]
            try: output.loc[output.frame_id.eq(row["frame_id"]),"resolved_path"]=str(future.result())
            except Exception as error: failures.append({"frame_id":row["frame_id"],"reason":"image_download_failed","error":str(error)})
    return output,failures
stage=time.perf_counter(); print("[4/6] Resolving/downloading images"); CANDIDATES,DOWNLOAD_FAILURES=download_candidates(CANDIDATES,FRAMES); OCR_CANDIDATES=CANDIDATES.loc[CANDIDATES.resolved_path.notna()].copy(); print(f"Download success={len(OCR_CANDIDATES)-LOCAL_IMAGES:,}; download failures={len(DOWNLOAD_FAILURES):,}; OCR-ready={len(OCR_CANDIDATES):,}; {time.perf_counter()-stage:.1f}s")
'''),
cell("code", r'''# [5/6] GPU OCR with per-chunk checkpoint; [6/6] compatible output artifacts.
DB_COLUMNS=["frame_id","n","text","language","x_min","x_max","y_min","y_max"]
def text_norm(value: str) -> str: return " ".join(unicodedata.normalize("NFC",str(value)).split())
def polygon4(value: np.ndarray,w: int,h: int) -> np.ndarray:
    points=np.asarray(value,np.float32).reshape(-1,2)
    if len(points)<4 or not np.isfinite(points).all(): raise ValueError("invalid polygon")
    if len(points)!=4: points=cv2.boxPoints(cv2.minAreaRect(points))
    points[:,0]=np.clip(points[:,0],0,w-1); points[:,1]=np.clip(points[:,1],0,h-1); sums,diffs=points.sum(1),np.diff(points,axis=1).ravel(); output=np.asarray([points[np.argmin(sums)],points[np.argmin(diffs)],points[np.argmax(sums)],points[np.argmax(diffs)]],np.float32)
    if len(np.unique(output,axis=0))!=4: raise ValueError("collapsed polygon")
    return output
def crop_text(image: np.ndarray,polygon: np.ndarray) -> np.ndarray:
    h,w=image.shape[:2]; polygon=polygon4(polygon,w,h); center=polygon.mean(0,keepdims=True); polygon=polygon4(center+(polygon-center)*(1+2*CROP_PADDING_RATIO),w,h); target_w=int(round(max(np.linalg.norm(polygon[1]-polygon[0]),np.linalg.norm(polygon[2]-polygon[3])))); target_h=int(round(max(np.linalg.norm(polygon[3]-polygon[0]),np.linalg.norm(polygon[2]-polygon[1]))))
    if target_w<MINIMUM_CROP_SIDE or target_h<MINIMUM_CROP_SIDE: raise ValueError("small crop")
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(polygon,np.asarray([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]],np.float32)),(target_w,target_h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
def is_oom(error: BaseException) -> bool: return "out of memory" in str(error).lower() and ("cuda" in str(error).lower() or "gpu" in str(error).lower())
class OCREngine:
    def __init__(self) -> None:
        from paddleocr import TextDetection,TextRecognition
        self.detector=TextDetection(model_name=DETECTION_MODEL_NAME,device="gpu:0",limit_side_len=1280,limit_type="max",thresh=.3,box_thresh=DETECTION_BOX_THRESHOLD,unclip_ratio=1.6); self.backend=RECOGNITION_BACKEND
        if self.backend=="vietocr":
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
            config=Cfg.load_config_from_name(VIETOCR_MODEL_NAME); config["device"]="cuda:0"; self.recognizer=Predictor(config); self.recognizer.model.eval()
        else: self.recognizer=TextRecognition(model_name=PADDLE_RECOGNITION_MODEL_NAME,device="gpu:0")
    def detect(self,images: list[np.ndarray],batch: int) -> list[list[tuple[np.ndarray,float]]]:
        output=[]
        for item in self.detector.predict(images,batch_size=batch): output.append([(np.asarray(poly,np.float32),float(score)) for poly,score in zip(list(item.get("dt_polys",[])),list(item.get("dt_scores",[])))])
        if len(output)!=len(images): raise RuntimeError("Paddle detection count/order mismatch")
        return output
    def recognize(self,crops: list[np.ndarray],batch: int) -> list[tuple[str,float]]:
        if not crops: return []
        if self.backend=="vietocr":
            images=[Image.fromarray(item[:,:,::-1]) for item in crops]; output=[]
            with torch.inference_mode():
                for start in range(0,len(images),batch):
                    texts,scores=self.recognizer.predict_batch(images[start:start+batch],return_prob=True); output.extend((str(text),float(score)) for text,score in zip(texts,scores))
            return output
        return [(str(item.get("rec_text","")),float(item.get("rec_score",0))) for item in self.recognizer.predict(crops,batch_size=batch)]
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
            batch=max(minimum,batch//2); gc.collect(); torch.cuda.empty_cache(); print("CUDA OOM: retrying with batch",batch)
def infer_chunk(engine: OCREngine,records: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[str],int,int]:
    images,valid,failures=[],[],[]
    for item in records:
        image=cv2.imread(item["resolved_path"])
        if image is None: failures.append({"frame_id":item["frame_id"],"reason":"image_unreadable","path":item["resolved_path"]})
        else: images.append(image); valid.append(item)
    if not images: return [],failures,[],DETECTION_BATCH_SIZE,RECOGNITION_BATCH_SIZE
    detected,det_batch=adaptive(engine.detect,images,DETECTION_BATCH_SIZE,MIN_DETECTION_BATCH_SIZE); crops,metadata=[],[]
    for index,(image,regions) in enumerate(zip(images,detected)):
        h,w=image.shape[:2]
        for polygon,_ in sorted([(p,s) for p,s in regions if s>=DETECTION_BOX_THRESHOLD],key=lambda item:float(item[0][:,1].mean())):
            try: crops.append(crop_text(image,polygon)); metadata.append((index,polygon,w,h))
            except ValueError: pass
    recognized,rec_batch=adaptive(engine.recognize,crops,RECOGNITION_BATCH_SIZE,MIN_RECOGNITION_BATCH_SIZE); rows=[]; count=defaultdict(int)
    for (index,polygon,w,h),(text,score) in zip(metadata,recognized):
        text=text_norm(text)
        if not text or score<RECOGNITION_SCORE_THRESHOLD: continue
        item=valid[index]; poly=polygon4(polygon,w,h); frame_id=item["frame_id"]; rows.append({"frame_id":frame_id,"n":count[frame_id],"text":text,"language":LANGUAGE,"x_min":float(poly[:,0].min()/w),"x_max":float(poly[:,0].max()/w),"y_min":float(poly[:,1].min()/h),"y_max":float(poly[:,1].max()/h),"video_id":item["video_id"],"shot_id":item["assigned_shot_id"],"timestamp_ms":int(item["timestamp_ms"]),"recognition_confidence":score}); count[frame_id]+=1
    return rows,failures,[item["frame_id"] for item in valid],det_batch,rec_batch
def append_jsonl(path: Path,items: Iterable[dict[str,Any]]) -> None:
    with path.open("a",encoding="utf-8") as handle:
        for item in items: handle.write(json.dumps(item,ensure_ascii=False)+"\\n")
def read_jsonl(path: Path) -> list[dict[str,Any]]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.is_file() else []
def iou(a: dict[str,Any],b: dict[str,Any]) -> float:
    overlap=max(0,min(a["x_max"],b["x_max"])-max(a["x_min"],b["x_min"]))*max(0,min(a["y_max"],b["y_max"])-max(a["y_min"],b["y_min"])); union=(a["x_max"]-a["x_min"])*(a["y_max"]-a["y_min"])+(b["x_max"]-b["x_min"])*(b["y_max"]-b["y_min"])-overlap; return overlap/union if union else 0.0
def deduplicate(rows: list[dict[str,Any]]) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    kept,dropped,recent=[],[],defaultdict(list)
    for row in sorted(rows,key=lambda item:(item["video_id"],item["shot_id"],item["timestamp_ms"],item["frame_id"],item["n"])):
        key=(row["shot_id"],text_norm(row["text"]).casefold()); recent[key]=[old for old in recent[key] if row["timestamp_ms"]-old["timestamp_ms"]<=OCR_DEDUP_WINDOW_MS]
        if any(iou(row,old)>=OCR_DEDUP_IOU_THRESHOLD for old in recent[key]): dropped.append({**row,"reason":"same_text_similar_box_within_temporal_window"}); continue
        recent[key].append(row); kept.append(row)
    numbers=defaultdict(int)
    for row in kept: row["n"]=numbers[row["frame_id"]]; numbers[row["frame_id"]]+=1
    return kept,dropped
def literal(value: Any) -> str:
    if value is None or (isinstance(value,float) and math.isnan(value)): return "NULL"
    if isinstance(value,(int,np.integer)): return str(int(value))
    if isinstance(value,(float,np.floating)): return format(float(value),".9g")
    return "'"+str(value).replace("'","''")+"'"
def write_output(raw: list[dict[str,Any]],failures: list[dict[str,Any]],started: float,det_batch: int,rec_batch: int) -> Path:
    temporary=RUN_DIR.with_name(RUN_DIR.name+".partial")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True); kept,dropped=deduplicate(raw); output=pd.DataFrame(kept,columns=[*DB_COLUMNS,"video_id","shot_id","timestamp_ms","recognition_confidence"])
    CANDIDATES.to_csv(temporary/"ocr_candidates.csv",index=False); output.to_csv(temporary/"ocr.csv",index=False); pd.DataFrame(dropped).to_csv(temporary/"ocr_deduplicated.csv",index=False)
    with (temporary/"failures.jsonl").open("w",encoding="utf-8") as handle:
        for item in failures: handle.write(json.dumps(item,ensure_ascii=False)+"\\n")
    with (temporary/"insert_ocr.sql").open("w",encoding="utf-8") as handle:
        handle.write("BEGIN;\\n")
        for row in output[DB_COLUMNS].to_dict("records"): handle.write("INSERT INTO ocr (frame_id, n, text, language, x_min, x_max, y_min, y_max) VALUES ("+", ".join(literal(row[key]) for key in DB_COLUMNS)+") ON CONFLICT (frame_id, n) DO NOTHING;\\n")
        handle.write("COMMIT;\\n")
    elapsed=max(time.monotonic()-started,.001); summary={"run_id":RUN_ID,"detection_model":DETECTION_MODEL_NAME,"recognition_backend":RECOGNITION_BACKEND,"recognition_model":VIETOCR_MODEL_NAME if RECOGNITION_BACKEND=="vietocr" else PADDLE_RECOGNITION_MODEL_NAME,"gpu":torch.cuda.get_device_name(0),"paddle_version":paddle.__version__,"torch_version":torch.__version__,"input_frames":len(FRAMES),"candidate_frames":len(CANDIDATES),"raw_ocr_rows":len(raw),"kept_ocr_rows":len(output),"deduplicated_ocr_rows":len(dropped),"failures":len(failures),"detection_batch_final":det_batch,"recognition_batch_final":rec_batch,"elapsed_seconds":round(elapsed,3),"candidate_frames_per_second":round(len(OCR_CANDIDATES)/elapsed,3),"peak_vram_bytes":int(torch.cuda.max_memory_allocated())}; (temporary/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); temporary.rename(RUN_DIR); return RUN_DIR
if RUN_PIPELINE:
    started=time.monotonic(); CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True); completed={item["frame_id"] for item in read_jsonl(CHECKPOINT_DIR/"completed.jsonl")}; pending=OCR_CANDIDATES.loc[~OCR_CANDIDATES.frame_id.isin(completed)]; print(f"[5/6] Running OCR: already OCR'd={len(completed):,}; remaining={len(pending):,}"); final_det,final_rec=DETECTION_BATCH_SIZE,RECOGNITION_BATCH_SIZE; engine=OCREngine()
    try:
        for start in tqdm(range(0,len(pending),FRAME_PROCESS_CHUNK_SIZE),desc="OCR chunks",unit="chunk"):
            chunk=pending.iloc[start:start+FRAME_PROCESS_CHUNK_SIZE].to_dict("records")
            try:
                rows,chunk_failures,successful,final_det,final_rec=infer_chunk(engine,chunk); append_jsonl(CHECKPOINT_DIR/"raw.jsonl",rows); append_jsonl(CHECKPOINT_DIR/"failures.jsonl",chunk_failures); append_jsonl(CHECKPOINT_DIR/"completed.jsonl",({"frame_id":frame_id} for frame_id in successful))
                for item in chunk:
                    path=Path(item["resolved_path"])
                    if IMAGE_CACHE_DIR in path.parents: remove_cached(path)
            except Exception as error:
                if not CONTINUE_ON_ERROR: raise
                append_jsonl(CHECKPOINT_DIR/"failures.jsonl",[{"reason":"chunk_failed","frame_ids":[item["frame_id"] for item in chunk],"error":str(error),"traceback":traceback.format_exc(limit=3)}])
        print("[6/6] Writing outputs"); output=write_output(read_jsonl(CHECKPOINT_DIR/"raw.jsonl"),PLAN_FAILURES+DOWNLOAD_FAILURES+read_jsonl(CHECKPOINT_DIR/"failures.jsonl"),started,final_det,final_rec); print("Completed. Kaggle output:",output); display(pd.read_csv(output/"ocr.csv").head())
    finally: engine.close()
else: print("Dry plan only. Set RUN_PIPELINE=True to execute GPU OCR.")
'''),
]

notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUTPUT.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(f"Wrote {OUTPUT}")
