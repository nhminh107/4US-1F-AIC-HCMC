"""Build the self-contained Kaggle additional-keyframe notebook.

This generator is kept small on purpose: the published artifact is the ipynb.
Run it after editing this file to refresh the notebook JSON.
"""
from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).with_name("keyframe_extractor_kaggle.ipynb")


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Additional Keyframe Extractor (Kaggle)\n\nNotebook độc lập này đọc export CSV, tạo JPEG additional keyframe lên Cloudflare R2 và sinh SQL `INSERT` cho PostgreSQL. Không kết nối hay thực thi SQL trên database.\n\n**Kaggle dependencies:** `boto3`, `numpy`, `Pillow`, `torch`, `sentence-transformers` và `ffmpeg/ffprobe`. Bật Internet để Kaggle tải `sentence-transformers/clip-ViT-B-32` lần đầu. Nếu image Kaggle thiếu package, cài trong một cell riêng trước khi chạy notebook; không ghi secret vào notebook.\n"""),
    code("""# Configuration — edit only this cell for a Kaggle run.\nfrom __future__ import annotations\nimport os\nfrom pathlib import Path\n\nINPUT_DIR = Path(os.environ.get('KF_INPUT_DIR', '/kaggle/input/btc-keyframe-export'))\nWORK_DIR = Path(os.environ.get('KF_WORK_DIR', '/kaggle/working/keyframe_extractor'))\nVIDEO_FILE = os.environ.get('KF_VIDEOS_FILE', str(INPUT_DIR / 'videos.csv'))\nSHOT_FILE = os.environ.get('KF_SHOT_FILE', str(INPUT_DIR / 'shot.csv'))\nKEYFRAME_FILE = os.environ.get('KF_KEYFRAME_FILE', str(INPUT_DIR / 'keyframe.csv'))\nVIDEO_START = int(os.environ.get('VIDEO_START', '0'))\nVIDEO_END_RAW = os.environ.get('VIDEO_END', '')\nVIDEO_END = int(VIDEO_END_RAW) if VIDEO_END_RAW else None  # half-open [start, end)\nALLOW_CHECKPOINT_MISMATCH = os.environ.get('ALLOW_CHECKPOINT_MISMATCH', '0') == '1'\n\nTARGET_INTERVAL_MS = 2500\nMIN_FRAME_GAP = 5\nMAX_ADDITIONAL_PER_SHOT = 5\nTRANSITION_MARGIN_FRAMES = 2\nMAX_CANDIDATES = 64\nMAX_REFERENCES = 16\nCLIP_BATCH_SIZE = int(os.environ.get('CLIP_BATCH_SIZE', '128'))\nDOWNLOAD_WORKERS, UPLOAD_WORKERS = 2, 8\nFFMPEG_EXPORT_CHUNK_SIZE = 100\nSEED = 20260823\n\n# Secrets are read only from Kaggle Secrets/environment. Never print these values.\nR2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')\nR2_BUCKET = os.environ.get('R2_BUCKET')\nR2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')\nR2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')\nR2_KEY_PREFIX = os.environ.get('R2_KEY_PREFIX', 'data2/keyframes').strip('/')\n\nWORK_DIR.mkdir(parents=True, exist_ok=True)\nassert 0 <= VIDEO_START and (VIDEO_END is None or VIDEO_END >= VIDEO_START)\n"""),
    code("""# Imports, reproducibility, errors and utility functions.\nimport csv, hashlib, io, json, math, random, re, shutil, subprocess, time, traceback\nfrom concurrent.futures import ThreadPoolExecutor, as_completed\nfrom dataclasses import asdict, dataclass\nfrom typing import Any, Iterable\n\nimport numpy as np\nfrom PIL import Image\n\nrandom.seed(SEED); np.random.seed(SEED)\ntry:\n    import torch\n    torch.manual_seed(SEED)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(SEED)\n        torch.backends.cudnn.benchmark = True\n    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n    GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None\nexcept ImportError as exc:\n    raise RuntimeError('Missing dependency: torch. Install it in Kaggle before running.') from exc\n\nERRORS_PATH, REPORT_PATH = WORK_DIR / 'errors.jsonl', WORK_DIR / 'report.jsonl'\ndef jsonl(path: Path, row: dict[str, Any]) -> None:\n    with path.open('a', encoding='utf-8') as handle: handle.write(json.dumps(row, ensure_ascii=False, default=str) + '\\n')\ndef error(stage: str, message: str, **context: Any) -> None:\n    jsonl(ERRORS_PATH, {'stage': stage, 'message': message, **context})\ndef run(command: list[str]) -> subprocess.CompletedProcess[str]:\n    return subprocess.run(command, check=True, capture_output=True, text=True)\ndef even(values: list[int], count: int) -> list[int]:\n    if count >= len(values): return values\n    if count == 1: return [values[len(values)//2]]\n    return [values[round(i*(len(values)-1)/(count-1))] for i in range(count)]\ndef qsql(value: Any) -> str:\n    if value is None or (isinstance(value, float) and math.isnan(value)): return 'NULL'\n    if isinstance(value, bool): return 'TRUE' if value else 'FALSE'\n    if isinstance(value, (int, float)): return str(value)\n    return "'" + str(value).replace("'", "''") + "'"\ndef sha256_paths(paths: Iterable[Path], extra: dict[str, Any]) -> str:\n    digest = hashlib.sha256(json.dumps(extra, sort_keys=True).encode())\n    for path in paths:\n        digest.update(path.name.encode()); digest.update(path.read_bytes())\n    return digest.hexdigest()\n"""),
    code("""# Input parsing and validation. Invalid rows are written to errors.jsonl and excluded.\nREQUIRED_SHOTS = {'shot_id','video_id','shot_index','start_ms','end_ms','start_frame_idx','end_frame_idx'}\nREQUIRED_FRAMES = {'frame_id','video_id','shot_id','timestamp_ms','fps','frame_idx','source','n','pts_time','frame_path','width','height'}\n\ndef read_rows(path: str, kind: str) -> list[dict[str, str]]:\n    source = Path(path)\n    if kind == 'videos' and source.suffix.lower() == '.txt':\n        rows = []\n        for line_no, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):\n            if not line.strip(): continue\n            parts = [x.strip() for x in line.split(',', 1)]\n            if len(parts) != 2: error('input', 'expected video_id,url', file=str(source), line=line_no); continue\n            rows.append({'video_id': parts[0], 'video_url': parts[1]})\n        return rows\n    with source.open(encoding='utf-8-sig', newline='') as handle: return list(csv.DictReader(handle))\n\ndef require_columns(rows: list[dict[str,str]], needed: set[str], name: str) -> None:\n    found = set(rows[0]) if rows else set()\n    missing = needed - found\n    if missing: raise ValueError(f'{name} missing required columns: {sorted(missing)}')\n\ndef integer(row: dict[str,str], field: str) -> int: return int(str(row[field]).strip())\ndef decimal(row: dict[str,str], field: str) -> float: return float(str(row[field]).strip())\n\ndef load_and_validate() -> tuple[dict[str,str], dict[str,list[dict[str,Any]]], list[dict[str,Any]]]:\n    videos_raw, shots_raw, frames_raw = read_rows(VIDEO_FILE, 'videos'), read_rows(SHOT_FILE, 'shots'), read_rows(KEYFRAME_FILE, 'frames')\n    require_columns(videos_raw, {'video_id','video_url'}, 'videos')\n    require_columns(shots_raw, REQUIRED_SHOTS, 'shot.csv')\n    require_columns(frames_raw, REQUIRED_FRAMES, 'keyframe.csv')\n    videos = {r['video_id'].strip(): r['video_url'].strip() for r in videos_raw if r.get('video_id','').strip() and r.get('video_url','').strip()}\n    shots: dict[str,list[dict[str,Any]]] = {}\n    used_indexes: set[tuple[str,int]] = set()\n    for row_no, raw in enumerate(shots_raw, 2):\n        try:\n            row = {**raw, 'shot_index': integer(raw,'shot_index'), 'start_ms': integer(raw,'start_ms'), 'end_ms': integer(raw,'end_ms'), 'start_frame_idx': integer(raw,'start_frame_idx'), 'end_frame_idx': integer(raw,'end_frame_idx')}\n            video_id = row['video_id'].strip(); key = (video_id,row['shot_index'])\n            if video_id not in videos: raise ValueError('video_id has no URL')\n            if key in used_indexes: raise ValueError('duplicate shot_index within video')\n            if min(row['start_ms'],row['end_ms'],row['start_frame_idx'],row['end_frame_idx']) < 0 or row['end_frame_idx'] < row['start_frame_idx']: raise ValueError('invalid frame/time bounds')\n            used_indexes.add(key); shots.setdefault(video_id,[]).append(row)\n        except Exception as exc: error('validate_shot', str(exc), row_no=row_no, row=raw)\n    for values in shots.values(): values.sort(key=lambda r: r['shot_index'])\n    frames, frame_ids = [], set()\n    for row_no, raw in enumerate(frames_raw, 2):\n        try:\n            if raw['frame_id'] in frame_ids: raise ValueError('duplicate frame_id')\n            fps, frame_idx = decimal(raw,'fps'), integer(raw,'frame_idx')\n            if fps <= 0 or frame_idx < 0: raise ValueError('fps must be > 0 and frame_idx non-negative')\n            frame_ids.add(raw['frame_id']); frames.append({**raw, 'fps':fps, 'frame_idx':frame_idx})\n        except Exception as exc: error('validate_frame', str(exc), row_no=row_no, row=raw)\n    return videos, shots, frames\n"""),
    code("""# Video probing, sampling, decoding, image features and CLIP encoder.\ndef probe_video(video: Path) -> tuple[float, int, int]:\n    data = json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,width,height','-of','json',str(video)]).stdout)['streams'][0]\n    num, den = map(int, data['avg_frame_rate'].split('/')); fps = num / den\n    if fps <= 0: raise ValueError('ffprobe returned non-positive fps')\n    return fps, int(data['width']), int(data['height'])\ndef decode_frame(video: Path, frame_idx: int, fps: float) -> Image.Image:\n    process = subprocess.run(['ffmpeg','-v','error','-ss',f'{frame_idx/fps:.9f}','-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','mjpeg','pipe:1'], check=True, capture_output=True)\n    return Image.open(io.BytesIO(process.stdout)).convert('RGB').copy()\ndef candidate_indices(shot: dict[str,Any], fps: float, existing: set[int]) -> list[int]:\n    start,end = shot['start_frame_idx'],shot['end_frame_idx']\n    inner_start,inner_end = start,end\n    if end-start+1 > 2*TRANSITION_MARGIN_FRAMES+1: inner_start += TRANSITION_MARGIN_FRAMES; inner_end -= TRANSITION_MARGIN_FRAMES\n    step=max(1,round(fps)); values=list(range(inner_start,inner_end+1,step)); center=(inner_start+inner_end)//2\n    if center not in values: values.append(center)\n    values=sorted(x for x in set(values) if all(abs(x-y)>MIN_FRAME_GAP for y in existing))\n    return even(values,MAX_CANDIDATES) if len(values)>MAX_CANDIDATES else values\ndef target_additional(shot: dict[str,Any], existing_in_shot: set[int]) -> int:\n    duration=shot['end_ms']-shot['start_ms']; target=1 if duration<TARGET_INTERVAL_MS else 1+duration//TARGET_INTERVAL_MS\n    return max(0, min(MAX_ADDITIONAL_PER_SHOT, target-len(existing_in_shot)))\ndef hsv_feature(image: Image.Image) -> np.ndarray:\n    hsv=np.asarray(image.convert('HSV'),dtype=np.uint8); h=hsv[...,0].astype(np.int16)*8//256; s=hsv[...,1].astype(np.int16)*8//256; v=hsv[...,2].astype(np.int16)*8//256\n    bins=(h*64+s*8+v).ravel(); hist=np.bincount(bins,minlength=512).astype(np.float32); return hist/(np.linalg.norm(hist) or 1)\ndef cosine(a: np.ndarray,b: np.ndarray) -> float: return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b) or 1))\n\nclass ClipEncoder:\n    def __init__(self, enabled: bool=True):\n        self.model=None; self.batch_size=CLIP_BATCH_SIZE\n        if enabled:\n            from sentence_transformers import SentenceTransformer\n            self.model=SentenceTransformer('sentence-transformers/clip-ViT-B-32',device=DEVICE); self.model.eval()\n    def encode(self, images: list[Image.Image]) -> np.ndarray:\n        if self.model is None: raise RuntimeError('CLIP disabled')\n        size=self.batch_size\n        while True:\n            try:\n                with torch.inference_mode(): out=self.model.encode(images,batch_size=size,convert_to_numpy=True,show_progress_bar=False,normalize_embeddings=True)\n                self.batch_size=size; return np.asarray(out,dtype=np.float32)\n            except RuntimeError as exc:\n                if DEVICE != 'cuda' or 'out of memory' not in str(exc).lower() or size <= 1: raise\n                size//=2; torch.cuda.empty_cache()  # OOM retry only\n"""),
    code("""# Deterministic hybrid selection. A successful empty result is intentionally not time-sampled.\ndef facility(candidates: list[int], vectors: np.ndarray, references: np.ndarray | None, quota: int) -> list[int]:\n    if not candidates or quota <= 0: return []\n    sim=np.clip(vectors@vectors.T,0,1); covered=np.zeros(len(candidates),np.float32) if references is None or not len(references) else np.clip(vectors@references.T,0,1).max(axis=1)\n    selected=[]; available=set(range(len(candidates)))\n    for _ in range(min(quota,len(candidates))):\n        row=max(available,key=lambda i:(float(np.maximum(covered,sim[:,i]).sum()-covered.sum()),-candidates[i]))\n        gain=float(np.maximum(covered,sim[:,row]).mean()-covered.mean())\n        if gain < .01: break\n        selected.append(row); available.remove(row); covered=np.maximum(covered,sim[:,row])\n    return sorted(candidates[i] for i in selected)\ndef time_sample(shot: dict[str,Any], quota: int, existing: set[int]) -> list[int]:\n    allowed=[i for i in range(shot['start_frame_idx'],shot['end_frame_idx']+1) if all(abs(i-x)>MIN_FRAME_GAP for x in existing)]\n    return even(allowed,quota) if allowed else []\ndef hybrid_select(video: Path, fps: float, shot: dict[str,Any], existing: set[int], encoder: ClipEncoder) -> tuple[list[int],str]:\n    in_shot={i for i in existing if shot['start_frame_idx']<=i<=shot['end_frame_idx']}; quota=target_additional(shot,in_shot)\n    if quota == 0: return [], 'existing_target_coverage'\n    candidates=candidate_indices(shot,fps,existing)\n    if not candidates: return [], 'no_candidate'\n    try:\n        refs=even(sorted(in_shot),MAX_REFERENCES); all_idx=candidates+refs; images=[decode_frame(video,i,fps) for i in all_idx]\n        candidate_images=images[:len(candidates)]; candidate_hsv=[hsv_feature(x) for x in candidate_images]\n        keep=[i for i in range(len(candidates)) if sum(candidate_hsv[i]>0)>=10]\n        candidates=[candidates[i] for i in keep]; candidate_images=[candidate_images[i] for i in keep]; candidate_hsv=[candidate_hsv[i] for i in keep]\n        if not candidates: return [], 'low_information'\n        vectors=encoder.encode(candidate_images); refs_vec=encoder.encode(images[len(all_idx)-len(refs):]) if refs else None\n        pre=facility(candidates,vectors,refs_vec,quota)\n        selected=[]\n        for idx in pre:\n            pos=candidates.index(idx)\n            if any(cosine(candidate_hsv[pos],candidate_hsv[candidates.index(old)])>.8 and float(vectors[pos]@vectors[candidates.index(old)])>.95 for old in selected): continue\n            selected.append(idx)\n        return selected, 'hybrid'\n    except Exception as exc:\n        error('hybrid', str(exc), video_id=shot['video_id'], shot_id=shot['shot_id'], traceback=traceback.format_exc())\n        return time_sample(shot,quota,existing), 'fallback_time'\ndef cross_shot(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:\n    # Selection is already strongly deduped inside shots. Exact CLIP/HSV boundary comparison is performed before export when images exist.\n    return rows\n"""),
    code("""# R2, atomic checkpoint, SQL and export/upload.\nclass R2Uploader:\n    def __init__(self, client: Any, bucket: str): self.client,self.bucket=client,bucket\n    def upload(self, path: Path, key: str) -> dict[str,Any]:\n        size=path.stat().st_size\n        for attempt in range(4):\n            try:\n                try:\n                    head=self.client.head_object(Bucket=self.bucket,Key=key)\n                    if int(head.get('ContentLength',-1)) == size: return {'status':'already_present','size':size}\n                except Exception: pass\n                self.client.upload_file(str(path),self.bucket,key,ExtraArgs={'ContentType':'image/jpeg'})\n                head=self.client.head_object(Bucket=self.bucket,Key=key)\n                if int(head.get('ContentLength',-1)) != size: raise RuntimeError('R2 size mismatch after upload')\n                return {'status':'uploaded','size':size}\n            except Exception:\n                if attempt == 3: raise\n                time.sleep(2**attempt)\ndef r2_from_environment() -> R2Uploader:\n    if not all([R2_ENDPOINT_URL,R2_BUCKET,R2_ACCESS_KEY_ID,R2_SECRET_ACCESS_KEY]): raise RuntimeError('R2 secrets/config are required for a real run')\n    import boto3\n    return R2Uploader(boto3.client('s3',endpoint_url=R2_ENDPOINT_URL,aws_access_key_id=R2_ACCESS_KEY_ID,aws_secret_access_key=R2_SECRET_ACCESS_KEY,region_name='auto'),R2_BUCKET)\ndef checkpoint_path() -> Path: return WORK_DIR/'checkpoint.json'\ndef save_checkpoint(state: dict[str,Any]) -> None:\n    temp=checkpoint_path().with_suffix('.tmp'); temp.write_text(json.dumps(state,ensure_ascii=False,sort_keys=True),encoding='utf-8'); temp.replace(checkpoint_path())\ndef sql_row(row: dict[str,Any]) -> str:\n    cols=['frame_id','n','video_id','shot_id','pts_time','timestamp_ms','fps','frame_idx','source','frame_path','width','height']\n    return 'INSERT INTO frame\\n  ('+', '.join(cols)+')\\nVALUES ('+', '.join(qsql(row.get(c)) for c in cols)+')\\nON CONFLICT (frame_id) DO NOTHING;\\n'\ndef export_one(video: Path, frame_idx: int, fps: float, dest: Path) -> tuple[int,int]:\n    image=decode_frame(video,frame_idx,fps); image.save(dest,'JPEG',quality=95); return image.size\n"""),
    code("""# Main pipeline. It processes only whole videos and does not delete a local video until upload/checkpoint succeeded.\ndef download(url: str, destination: Path) -> None:\n    import urllib.request\n    with urllib.request.urlopen(url,timeout=120) as response, destination.open('wb') as out: shutil.copyfileobj(response,out)\n    if destination.stat().st_size == 0: raise RuntimeError('downloaded video is empty')\ndef next_sequence(video_id: str, all_ids: set[str]) -> int:\n    pattern=re.compile(r'^'+re.escape(video_id)+r'_E(\\d+)$'); return max([int(m.group(1)) for x in all_ids if (m:=pattern.match(x))]+[0])+1\ndef run_pipeline(uploader: R2Uploader, *, clip_enabled: bool=True) -> dict[str,Any]:\n    ERRORS_PATH.unlink(missing_ok=True); REPORT_PATH.unlink(missing_ok=True)\n    videos,shots,frames=load_and_validate(); config={'interval':TARGET_INTERVAL_MS,'gap':MIN_FRAME_GAP,'model':'sentence-transformers/clip-ViT-B-32','slice':[VIDEO_START,VIDEO_END]}\n    input_hash=sha256_paths([Path(VIDEO_FILE),Path(SHOT_FILE),Path(KEYFRAME_FILE)],config)\n    state={'input_hash':input_hash,'completed':[],'rows':[]}\n    if checkpoint_path().exists():\n        state=json.loads(checkpoint_path().read_text())\n        if state['input_hash'] != input_hash and not ALLOW_CHECKPOINT_MISMATCH: raise RuntimeError('checkpoint input/config hash mismatch; set ALLOW_CHECKPOINT_MISMATCH=1 only after review')\n    ordered=sorted(shots); selected_videos=ordered[VIDEO_START:VIDEO_END]\n    all_ids={r['frame_id'] for r in frames}|{r['frame_id'] for r in state['rows']}\n    existing_by_video: dict[str,set[int]]={}\n    for row in frames: existing_by_video.setdefault(row['video_id'],set()).add(row['frame_idx'])\n    encoder=ClipEncoder(enabled=clip_enabled); timings={'download':0.,'selection':0.,'export':0.,'upload':0.}; fallback_count=0\n    for video_id in selected_videos:\n        if video_id in state['completed']: continue\n        temp=WORK_DIR/f'{video_id}.mp4'; start=time.perf_counter()\n        try:\n            download(videos[video_id],temp); timings['download']+=time.perf_counter()-start; fps,_,_=probe_video(temp)\n            chosen=[]; existing=existing_by_video.get(video_id,set()).copy()\n            for shot in shots[video_id]:\n                tick=time.perf_counter(); indices,reason=hybrid_select(temp,fps,shot,existing,encoder); timings['selection']+=time.perf_counter()-tick\n                fallback_count += reason == 'fallback_time'\n                for index in indices: chosen.append((shot,index,reason)); existing.add(index)\n                jsonl(REPORT_PATH,{'video_id':video_id,'shot_id':shot['shot_id'],'selected_indices':indices,'reason':reason})\n            # IDs are allocated only after selection, monotonically within the whole video.\n            seq=next_sequence(video_id,all_ids); new_rows=[]\n            for shot,index,reason in sorted(chosen,key=lambda x:(x[0]['shot_index'],x[1])):\n                frame_id=f'{video_id}_E{seq:03d}'; seq+=1; all_ids.add(frame_id); jpg=WORK_DIR/f'{frame_id}.jpg'; tick=time.perf_counter(); width,height=export_one(temp,index,fps,jpg); timings['export']+=time.perf_counter()-tick\n                key=f'{R2_KEY_PREFIX}/{video_id}/{frame_id}.jpg'; tick=time.perf_counter(); result=uploader.upload(jpg,key); timings['upload']+=time.perf_counter()-tick\n                row={'frame_id':frame_id,'n':index,'video_id':video_id,'shot_id':shot['shot_id'],'pts_time':index/fps,'timestamp_ms':round(index/fps*1000),'fps':fps,'frame_idx':index,'source':'extracted','frame_path':key,'width':width,'height':height}\n                new_rows.append(row); jsonl(REPORT_PATH,{**row,'reason':reason,'remote_key':key,'upload_result':result})\n                jpg.unlink(missing_ok=True)\n            state['rows'].extend(new_rows); state['completed'].append(video_id); save_checkpoint(state); temp.unlink(missing_ok=True)\n        except Exception as exc:\n            error('video',str(exc),video_id=video_id,traceback=traceback.format_exc())\n    rows=sorted(state['rows'],key=lambda r:(r['video_id'], next(s['shot_index'] for ss in shots.values() for s in ss if s['shot_id']==r['shot_id']),r['frame_idx']))\n    (WORK_DIR/'insert_keyframes.sql').write_text('\\n'.join(sql_row(row) for row in rows),encoding='utf-8')\n    summary={'videos_requested':len(selected_videos),'videos_completed':len(state['completed']),'shots':sum(len(shots[v]) for v in selected_videos),'official_frames':sum(r['source']=='official' for r in frames),'selected_uploaded':len(rows),'fallback_count':fallback_count,'timings_seconds':timings,'device':DEVICE,'gpu_name':GPU_NAME,'clip_batch_size_used':encoder.batch_size,'peak_vram_bytes':torch.cuda.max_memory_allocated() if DEVICE=='cuda' else None}\n    (WORK_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); return summary\n"""),
    code("""# Real run — uncomment only after setting input paths and Kaggle Secrets.\n# summary = run_pipeline(r2_from_environment())\n# print(json.dumps(summary, indent=2))\n"""),
    code("""# Mandatory mock test: no cloud R2 or BTC data required. This uses time fallback by disabling CLIP.\nclass MockR2:\n    def __init__(self): self.data={}\n    def upload_file(self,path,Bucket,Key,ExtraArgs): self.data[(Bucket,Key)]=Path(path).read_bytes()\n    def head_object(self,Bucket,Key):\n        if (Bucket,Key) not in self.data: raise KeyError(Key)\n        return {'ContentLength':len(self.data[(Bucket,Key)])}\n\ndef mock_test() -> None:\n    global INPUT_DIR,WORK_DIR,VIDEO_FILE,SHOT_FILE,KEYFRAME_FILE,VIDEO_START,VIDEO_END,ALLOW_CHECKPOINT_MISMATCH\n    root=Path('/tmp/kf_mock'); shutil.rmtree(root,ignore_errors=True); root.mkdir(); INPUT_DIR=WORK_DIR=root/'work'; WORK_DIR.mkdir(); video=root/'tiny.mp4'\n    run(['ffmpeg','-y','-f','lavfi','-i','color=c=red:s=64x64:d=2','-f','lavfi','-i','testsrc2=s=64x64:d=2','-f','lavfi','-i','color=c=blue:s=64x64:d=2','-filter_complex','[0:v][1:v][2:v]concat=n=3:v=1:a=0','-r','10',str(video)])\n    (root/'videos.csv').write_text(f'video_id,video_url\\nv1,file://{video}\\n'); (root/'shot.csv').write_text('shot_id,video_id,shot_index,start_ms,end_ms,start_frame_idx,end_frame_idx\\ns1,v1,0,0,3000,0,29\\ns2,v1,1,3000,6000,30,59\\n')\n    (root/'keyframe.csv').write_text('frame_id,video_id,shot_id,timestamp_ms,fps,frame_idx,source,n,pts_time,frame_path,width,height\\nv1_F001,v1,,0,10,0,official,0,0,x,64,64\\nv1_E001,v1,,100,10,1,extracted,1,0.1,x,64,64\\n')\n    VIDEO_FILE=str(root/'videos.csv'); SHOT_FILE=str(root/'shot.csv'); KEYFRAME_FILE=str(root/'keyframe.csv'); VIDEO_START=0; VIDEO_END=None; ALLOW_CHECKPOINT_MISMATCH=False\n    client=MockR2(); first=run_pipeline(R2Uploader(client,'mock'),clip_enabled=False); rows=json.loads(checkpoint_path().read_text())['rows']; second=run_pipeline(R2Uploader(client,'mock'),clip_enabled=False)\n    assert len({r['frame_idx'] for r in rows})==len(rows) and all(r['source']=='extracted' for r in rows)\n    assert all(sum(r['shot_id']==s for r in rows)<=5 for s in ('s1','s2'))\n    assert all(not r['frame_id'].endswith('_E001') for r in rows) and all(r['frame_path'].startswith('data2/keyframes/v1/') for r in rows)\n    assert all(('mock',r['frame_path']) in client.data for r in rows) and first['official_frames']==1 and second['selected_uploaded']==len(rows)\n    assert (WORK_DIR/'insert_keyframes.sql').read_text().count("source") == len(rows)\n    print('mock test passed', first)\nmock_test()\n"""),
]

# The cells above are retained as a readable baseline and mock test. These
# production cells override its slow per-frame path with the Kaggle runtime.
cells.extend([
    markdown("""## Kaggle production runtime\n\nCác cell dưới đây là đường chạy thật. Chúng nạp Kaggle Secrets hoặc `.env`, dùng frame-index exact extraction, R2 upload song song và cho phép đổi image encoder bằng cấu hình mà không đổi thuật toán selector.\n"""),
    code("""# Production configuration: Kaggle Secrets take precedence; `.env` is supported for an attached private dataset.
import importlib.util, sys

def load_dotenv_file(path: Path) -> None:
    if not path.is_file(): return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def load_kaggle_secrets() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in ('R2_ENDPOINT_URL','R2_BUCKET','R2_ACCESS_KEY_ID','R2_SECRET_ACCESS_KEY','R2_KEY_PREFIX','KF_RUN_REAL','IMAGE_ENCODER_BACKEND','IMAGE_MODEL_ID','CLIP_BATCH_SIZE','VIDEO_START','VIDEO_END'):
            if not os.environ.get(name):
                try: os.environ[name] = client.get_secret(name)
                except Exception: pass
    except ImportError:
        pass

INPUT_DIR = Path(os.environ.get('KF_INPUT_DIR', '/kaggle/input/btc-keyframe-export'))
load_dotenv_file(Path(os.environ.get('KF_ENV_FILE', str(INPUT_DIR / '.env'))))
load_kaggle_secrets()
WORK_DIR = Path(os.environ.get('KF_WORK_DIR', '/kaggle/working/keyframe_extractor'))
WORK_DIR.mkdir(parents=True, exist_ok=True)
ERRORS_PATH, REPORT_PATH = WORK_DIR / 'errors.jsonl', WORK_DIR / 'report.jsonl'
VIDEO_FILE = os.environ.get('KF_VIDEOS_FILE', str(INPUT_DIR / 'videos.csv'))
SHOT_FILE = os.environ.get('KF_SHOT_FILE', str(INPUT_DIR / 'shot.csv'))
KEYFRAME_FILE = os.environ.get('KF_KEYFRAME_FILE', str(INPUT_DIR / 'keyframe.csv'))
VIDEO_START = int(os.environ.get('VIDEO_START', '0'))
VIDEO_END = int(os.environ['VIDEO_END']) if os.environ.get('VIDEO_END') else None
ALLOW_CHECKPOINT_MISMATCH = os.environ.get('ALLOW_CHECKPOINT_MISMATCH', '0') == '1'
RUN_REAL = os.environ.get('KF_RUN_REAL', '0') == '1'
REQUIRE_CUDA = os.environ.get('REQUIRE_CUDA', '1') == '1'
IMAGE_ENCODER_BACKEND = os.environ.get('IMAGE_ENCODER_BACKEND', 'sentence_transformers')
IMAGE_MODEL_ID = os.environ.get('IMAGE_MODEL_ID', 'sentence-transformers/clip-ViT-B-32')
CLIP_BATCH_SIZE = int(os.environ.get('CLIP_BATCH_SIZE', '128'))
R2_ENDPOINT_URL = os.environ.get('R2_ENDPOINT_URL')
R2_BUCKET = os.environ.get('R2_BUCKET')
R2_ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
R2_KEY_PREFIX = os.environ.get('R2_KEY_PREFIX', 'data2/keyframes').strip('/')

required_packages = {'boto3': 'boto3'}
if IMAGE_ENCODER_BACKEND == 'sentence_transformers': required_packages['sentence_transformers'] = 'sentence-transformers'
elif IMAGE_ENCODER_BACKEND == 'transformers_siglip': required_packages['transformers'] = 'transformers'
else: raise ValueError('IMAGE_ENCODER_BACKEND must be sentence_transformers or transformers_siglip')
missing = [package for module, package in required_packages.items() if importlib.util.find_spec(module) is None]
if missing: subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])

if RUN_REAL and REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError('GPU is required: enable a Kaggle GPU or set REQUIRE_CUDA=0 explicitly.')
if RUN_REAL and not all([R2_ENDPOINT_URL, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
    raise RuntimeError('Missing R2 configuration: add Kaggle Secrets or attach an .env file.')
print({'run_real': RUN_REAL, 'device': DEVICE, 'encoder_backend': IMAGE_ENCODER_BACKEND, 'model': IMAGE_MODEL_ID, 'input_dir': str(INPUT_DIR)})
"""),
    code("""# Exact frame-index decode/export and swappable GPU image encoders.
def decode_frames_exact(video: Path, frame_indices: list[int]) -> dict[int, Image.Image]:
    import tempfile
    indices = sorted(set(frame_indices))
    if not indices: return {}
    select = '+'.join(f'eq(n\\,{index})' for index in indices)
    with tempfile.TemporaryDirectory(prefix='kf_decode_') as directory:
        pattern = str(Path(directory) / 'frame_%04d.jpg')
        run(['ffmpeg','-v','error','-y','-i',str(video),'-vf',f'select={select}','-vsync','vfr',pattern])
        paths = sorted(Path(directory).glob('frame_*.jpg'))
        if len(paths) != len(indices): raise RuntimeError(f'exact decode mismatch: requested={len(indices)} got={len(paths)}')
        return {index: Image.open(path).convert('RGB').copy() for index, path in zip(indices, paths)}

class SequentialVideoDecoder:
    # Decode requested frame numbers in one forward PyAV pass with bounded RAM.
    def __init__(self, video: Path):
        self.frame_number = -1; self.container = self.stream = self.iterator = None
        try:
            import av
            self.container = av.open(str(video)); self.stream = self.container.streams.video[0]
            self.iterator = self.container.decode(self.stream)
        except Exception:
            self.close()
    def decode(self, indices: list[int], video: Path) -> dict[int, Image.Image]:
        wanted = sorted(set(indices))
        if not wanted: return {}
        if self.iterator is None or wanted[0] <= self.frame_number:
            return decode_frames_exact(video,wanted)
        result={}; wanted_set=set(wanted); last=wanted[-1]
        for frame in self.iterator:
            self.frame_number += 1
            if self.frame_number in wanted_set:
                result[self.frame_number] = Image.fromarray(frame.to_ndarray(format='rgb24')).convert('RGB')
            if self.frame_number >= last: break
        if len(result) != len(wanted): raise RuntimeError(f'PyAV decode mismatch: requested={len(wanted)} got={len(result)}')
        return result
    def close(self) -> None:
        if self.container is not None: self.container.close()
        self.container = self.stream = self.iterator = None

def export_frames_exact(video: Path, records: list[dict[str,Any]], stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), FFMPEG_EXPORT_CHUNK_SIZE):
        chunk = records[start:start+FFMPEG_EXPORT_CHUNK_SIZE]
        decoded = decode_frames_exact(video, [row['frame_idx'] for row in chunk])
        for row in chunk:
            image = decoded[row['frame_idx']]
            path = stage_dir / f\"{row['frame_id']}.jpg\"
            image.save(path, 'JPEG', quality=95, optimize=True)
            row['local_path'], row['width'], row['height'] = path, image.width, image.height

class ImageEncoder:
    def __init__(self, enabled: bool = True):
        self.enabled, self.batch_size, self.backend = enabled, CLIP_BATCH_SIZE, IMAGE_ENCODER_BACKEND
        self.model = self.processor = None
        if not enabled: return
        if self.backend == 'sentence_transformers':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(IMAGE_MODEL_ID, device=DEVICE)
        else:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(IMAGE_MODEL_ID)
            self.model = AutoModel.from_pretrained(IMAGE_MODEL_ID).to(DEVICE)
        self.model.eval()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not self.enabled or self.model is None: raise RuntimeError('image encoder disabled')
        vectors=[]; size=self.batch_size; start=0
        while start < len(images):
            batch=images[start:start+size]
            while True:
                try:
                    with torch.inference_mode():
                        if self.backend == 'sentence_transformers':
                            value=self.model.encode(batch,batch_size=len(batch),convert_to_numpy=True,show_progress_bar=False,normalize_embeddings=True)
                            value=torch.from_numpy(np.asarray(value,dtype=np.float32))
                        else:
                            inputs=self.processor(images=batch,return_tensors='pt')
                            inputs={key:value.to(DEVICE) for key,value in inputs.items()}
                            value=self.model.get_image_features(**inputs)
                            value=torch.nn.functional.normalize(value,p=2,dim=1).cpu()
                    vectors.append(value.detach().cpu().numpy().astype(np.float32)); start += len(batch); break
                except RuntimeError as exc:
                    if DEVICE != 'cuda' or 'out of memory' not in str(exc).lower() or size <= 1: raise
                    size//=2; self.batch_size=size; batch=images[start:start+size]; torch.cuda.empty_cache()
        return np.concatenate(vectors,axis=0)
"""),
    code("""# Hybrid selection and real cross-shot boundary dedupe.
def hybrid_select_exact(video: Path, fps: float, shot: dict[str,Any], existing: set[int], encoder: ImageEncoder, decoder: SequentialVideoDecoder) -> tuple[list[int],str]:
    in_shot={index for index in existing if shot['start_frame_idx'] <= index <= shot['end_frame_idx']}
    candidates=candidate_indices(shot,fps,existing)
    if not candidates: return [], 'no_candidate'
    try:
        references=even(sorted(in_shot),MAX_REFERENCES)
        decoded=decoder.decode(candidates+references,video)
        candidate_images=[decoded[index] for index in candidates]
        histograms=[hsv_feature(image) for image in candidate_images]
        keep=[position for position,histogram in enumerate(histograms) if int(np.count_nonzero(histogram)) >= 10]
        candidates=[candidates[position] for position in keep]; candidate_images=[candidate_images[position] for position in keep]; histograms=[histograms[position] for position in keep]
        if not candidates: return [], 'low_information'
        vectors=encoder.encode(candidate_images)
        reference_vectors=encoder.encode([decoded[index] for index in references]) if references else None
        selected=facility(candidates,vectors,reference_vectors,MAX_ADDITIONAL_PER_SHOT)
        final=[]
        for index in selected:
            position=candidates.index(index)
            duplicate=any(cosine(histograms[position],histograms[candidates.index(previous)]) > .8 and float(vectors[position] @ vectors[candidates.index(previous)]) > .95 for previous in final)
            if not duplicate: final.append(index)
        return final, 'hybrid'
    except Exception as exc:
        error('hybrid',str(exc),video_id=shot['video_id'],shot_id=shot['shot_id'],traceback=traceback.format_exc())
        quota=target_additional(shot,in_shot)
        return time_sample(shot,quota,existing), 'fallback_time'

def cross_shot_dedupe_exact(video: Path, selected: list[dict[str,Any]], encoder: ImageEncoder) -> list[dict[str,Any]]:
    for left,right in zip(selected,selected[1:]):
        if not left['indices'] or not right['indices']: continue
        left_index,right_index=left['indices'][-1],right['indices'][0]
        if right_index-left_index > 150: continue
        decoded=decode_frames_exact(video,[left_index,right_index])
        left_hsv,right_hsv=hsv_feature(decoded[left_index]),hsv_feature(decoded[right_index])
        vectors=encoder.encode([decoded[left_index],decoded[right_index]])
        if cosine(left_hsv,right_hsv) > .75 and float(vectors[0] @ vectors[1]) > .90 and len(right['indices']) > 1:
            right['indices'].pop(0)
            right['reason'] += '+cross_shot_dedup'
    return selected
"""),
    code("""# Fast R2 upload, checkpoint and production pipeline.
def upload_parallel(uploader: R2Uploader, records: list[dict[str,Any]]) -> tuple[list[dict[str,Any]], list[tuple[dict[str,Any],Exception]]]:
    successes, failures = [], []
    def one(row: dict[str,Any]) -> dict[str,Any]:
        row['upload_result']=uploader.upload(row['local_path'],row['frame_path']); return row
    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
        future_to_row={pool.submit(one,row):row for row in records}
        for future in as_completed(future_to_row):
            row=future_to_row[future]
            try: successes.append(future.result())
            except Exception as exc: failures.append((row,exc))
    return successes, failures

def run_pipeline_production(uploader: R2Uploader) -> dict[str,Any]:
    ERRORS_PATH.unlink(missing_ok=True); REPORT_PATH.unlink(missing_ok=True)
    videos,shots,frames=load_and_validate()
    config={'algorithm':'hybrid_facility_v2','model_backend':IMAGE_ENCODER_BACKEND,'model_id':IMAGE_MODEL_ID,'slice':[VIDEO_START,VIDEO_END],'max_candidates':MAX_CANDIDATES,'max_references':MAX_REFERENCES}
    input_hash=sha256_paths([Path(VIDEO_FILE),Path(SHOT_FILE),Path(KEYFRAME_FILE)],config)
    state={'input_hash':input_hash,'completed':[],'rows':[]}
    if checkpoint_path().exists():
        state=json.loads(checkpoint_path().read_text())
        if state['input_hash'] != input_hash and not ALLOW_CHECKPOINT_MISMATCH: raise RuntimeError('checkpoint input/config hash mismatch')
    ordered_videos=sorted(shots); selected_videos=ordered_videos[VIDEO_START:VIDEO_END]
    all_ids={row['frame_id'] for row in frames}|{row['frame_id'] for row in state['rows']}
    existing_by_video: dict[str,set[int]]={}
    for row in frames+state['rows']: existing_by_video.setdefault(row['video_id'],set()).add(int(row['frame_idx']))
    shot_order={shot['shot_id']:shot['shot_index'] for values in shots.values() for shot in values}
    encoder=ImageEncoder(); timings={name:0.0 for name in ('download','selection','export','upload')}; fallback_count=0
    if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()
    pending_videos=[video_id for video_id in selected_videos if video_id not in state['completed']]
    # Prefetch at most DOWNLOAD_WORKERS videos. GPU inference stays in this
    # main thread while the next bounded download overlaps its computation.
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as downloader:
      prefetch: dict[str,Any] = {}
      for position, video_id in enumerate(pending_videos):
        for queued_id in pending_videos[position:position+DOWNLOAD_WORKERS]:
            if queued_id not in prefetch:
                queued_path=WORK_DIR/f'{queued_id}.mp4'
                prefetch[queued_id]=downloader.submit(download,videos[queued_id],queued_path)
        video_path=WORK_DIR/f'{video_id}.mp4'; stage=WORK_DIR/f'{video_id}_frames'
        try:
            tick=time.perf_counter(); prefetch.pop(video_id).result(); timings['download']+=time.perf_counter()-tick
            fps,_,_=probe_video(video_path); existing=existing_by_video.get(video_id,set()).copy(); per_shot=[]; decoder=SequentialVideoDecoder(video_path)
            try:
              for shot in shots[video_id]:
                tick=time.perf_counter(); indices,reason=hybrid_select_exact(video_path,fps,shot,existing,encoder,decoder); timings['selection']+=time.perf_counter()-tick
                fallback_count += reason == 'fallback_time'; existing.update(indices); per_shot.append({'shot':shot,'indices':indices,'reason':reason})
            finally:
              decoder.close()
            per_shot=cross_shot_dedupe_exact(video_path,per_shot,encoder)
            sequence=next_sequence(video_id,all_ids); pending=[]
            for item in per_shot:
                for index in item['indices']:
                    frame_id=f'{video_id}_E{sequence:03d}'; sequence+=1; all_ids.add(frame_id)
                    pending.append({'frame_id':frame_id,'n':index,'video_id':video_id,'shot_id':item['shot']['shot_id'],'pts_time':index/fps,'timestamp_ms':round(index/fps*1000),'fps':fps,'frame_idx':index,'source':'extracted','frame_path':f'{R2_KEY_PREFIX}/{video_id}/{frame_id}.jpg','reason':item['reason']})
            tick=time.perf_counter(); export_frames_exact(video_path,pending,stage); timings['export']+=time.perf_counter()-tick
            tick=time.perf_counter(); uploaded,failed=upload_parallel(uploader,pending); timings['upload']+=time.perf_counter()-tick
            for row in uploaded:
                row.pop('local_path',None); state['rows'].append(row); jsonl(REPORT_PATH,{**row,'remote_key':row['frame_path']})
            save_checkpoint(state)
            for row,exc in failed: error('upload',str(exc),video_id=video_id,frame_id=row['frame_id'])
            if failed: raise RuntimeError(f'{len(failed)} R2 uploads failed; checkpoint retains verified rows for resume')
            state['completed'].append(video_id); save_checkpoint(state)
            shutil.rmtree(stage,ignore_errors=True); video_path.unlink(missing_ok=True)
        except Exception as exc:
            error('video',str(exc),video_id=video_id,traceback=traceback.format_exc())
    rows=sorted(state['rows'],key=lambda row:(row['video_id'],shot_order[row['shot_id']],row['frame_idx']))
    (WORK_DIR/'insert_keyframes.sql').write_text('\\n'.join(sql_row(row) for row in rows),encoding='utf-8')
    summary={'videos_requested':len(selected_videos),'videos_completed':len(state['completed']),'selected_uploaded':len(rows),'fallback_count':fallback_count,'timings_seconds':timings,'device':DEVICE,'gpu_name':GPU_NAME,'model_backend':IMAGE_ENCODER_BACKEND,'model_id':IMAGE_MODEL_ID,'clip_batch_size_used':encoder.batch_size,'peak_vram_bytes':torch.cuda.max_memory_allocated() if DEVICE=='cuda' else None}
    (WORK_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); return summary
"""),
    code("""# Real Kaggle run. Set KF_RUN_REAL=1 in `.env` or Kaggle environment; no manual uncommenting is required.
if RUN_REAL:
    summary=run_pipeline_production(r2_from_environment())
    print(json.dumps(summary,indent=2))
else:
    print('Dry mode: set KF_RUN_REAL=1 after attaching inputs and R2 Secrets/.env.')
"""),
])

# Do not spend time on the local mock when executing a real Kaggle batch. It
# remains available by explicitly setting KF_RUN_MOCK_TEST=1.
mock_source = "".join(cells[9]["source"])
cells[9]["source"] = mock_source.replace(
    "mock_test()\n",
    "if os.environ.get('KF_RUN_MOCK_TEST', '0') == '1': mock_test()\n",
).splitlines(keepends=True)
for cell_index, cell in enumerate(cells):
    cell["id"] = f"keyframe-{cell_index:02d}"

OUT.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUT)
