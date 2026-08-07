# Object Detection Module - Technical Decisions

File nay tra loi 10 cau hoi ky thuat ve module `BackEnd/app/object_detection/` da duoc code.

## Q1. `class_id` format trong `ObjectDetectionResult`

`class_id` dang duoc dung theo format `cNNN`, trong do `NNN` la class index cua model, zero-padded 3 chu so.

Vi du voi COCO/YOLO:

- `person` co class index `0`
- `class_id = "c000"`
- `car` co class index `2`
- `class_id = "c002"`

Ly do chon format nay:

- Khop voi schema DB `ClassID.class_id varchar(15)`.
- On dinh hon viec dung class name truc tiep vi class name co the co space, lowercase/uppercase, hoac thay doi ngon ngu hien thi.
- De join voi bang `ClassID`/`ClassMetadata`.

Code lien quan:

- `BackEnd/app/object_detection/class_mapper.py`
- `ClassMapper.class_id_for_index()`

## Q2. Bounding box - pixel hay normalized?

Co 2 lop output khac nhau:

### 1. Detection object / JSON debug output

`Detection.bbox` luu pixel coordinates theo thu tu:

```text
[x_min, y_min, x_max, y_max]
```

Vi du:

```json
{
  "bbox": {
    "x_min": 110,
    "y_min": 60,
    "x_max": 240,
    "y_max": 410
  }
}
```

### 2. `ObjectDetectionResult` contract / DB output

`ObjectDetectionResult` dung normalized coordinates trong khoang `[0.0, 1.0]`, vi schema DB da co constraint:

```sql
x_min between 0 and 1
x_max between 0 and 1
y_min between 0 and 1
y_max between 0 and 1
```

Trong dataclass, field order la:

```python
x_min, x_max, y_min, y_max
```

Con khi doc bbox dang list/dict theo format object-detection thong dung thi thu tu la:

```text
[x_min, y_min, x_max, y_max]
```

Code lien quan:

- `BackEnd/app/object_detection/schemas.py`
- `BoundingBox.normalized()`
- `BackEnd/app/object_detection/exporter.py`
- `detections_to_contracts()`

## Q3. `class_name` bi thieu trong `ObjectDetectionResult`

`ObjectDetectionResult` khong duoc bo sung `class_name`. Module phia sau neu can ten class se join/query qua `ClassMetadata` hoac bang `ClassID`.

Hien tai:

- JSON output co `class_name` de debug/doc lap.
- Contract pipeline/DB chi luu `class_id`.
- Mapping `class_id -> class_name` nam o `ClassMapper` va co the export thanh `ClassMetadata`.

Ly do:

- Tranh duplicate `class_name` trong tung detection.
- Giu `ObjectDetectionResult` khop voi contract san co trong `BackEnd/app/contracts/pipeline.py`.
- DB da co bang `ClassID(class_id, class_name)`.

## Q4. Class mapping - mot file hay nhieu file theo model?

Trong code hien tai, class mapping la per-model theo runtime:

- `YOLODetector` lay `self.model.names` tu model `ultralytics`.
- `ClassMapper` nhan `names` tu model.
- Neu model khong tra ve names, fallback sang `COCO_CLASSES`.

Chua tao `configs/class_mapping.json` trong lan code nay. Neu can config file trong buoc tiep theo, nen thiet ke theo per-model, khong nen mot flat file chung.

Format de xuat:

```json
{
  "yolov8-coco": {
    "0": "person",
    "1": "bicycle",
    "2": "car"
  },
  "rtdetr-coco": {
    "0": "person",
    "1": "bicycle",
    "2": "car"
  }
}
```

Voi open-vocabulary model nhu GroundingDINO, mapping khong nen gan cung theo index COCO; class id nen duoc sinh tu prompt/config cua lan chay.

## Q5. GroundingDINO va abstraction `detect_classes(frame, classes)`

GroundingDINO chua nam trong implementation hien tai.

Module hien tai co `Detector.detect(image, frame_id=None, img_path=None)` va `YOLODetector`. Voi YOLO, class filter co the truyen vao detector khi khoi tao:

```python
YOLODetector(class_names=["person", "car"])
```

Trong YOLO, filter class duoc dua vao model inference qua tham so `classes`, khong phai chi filter sau ket qua.

Neu them GroundingDINO, can xu ly khac:

- `classes`/text prompts phai la input truoc inference.
- Nen tao `GroundingDINODetector(prompts=[...])`.
- Khong nen dung abstraction "detect tat ca roi filter sau" cho GroundingDINO.

Ket luan: abstraction dung hon la detector duoc cau hinh truoc khi detect, con post-filter chi la buoc phu tuy chon.

## Q6. "Duplicate removing" la NMS hay frame-level dedup?

Trong implementation hien tai:

- YOLO da chay built-in NMS thong qua `ultralytics`.
- `run_detection_on_chunk(..., apply_nms=False)` mac dinh khong chay NMS lan hai.
- `postprocess.nms()` chi la helper tuy chon neu caller muon ap dung lai voi output model khac hoac raw boxes.

Vay "duplicate removing" hien tai nen hieu la IoU-based NMS tuy chon, khong phai frame-level dedup.

Khong co frame-level dedup trong object detection. Frame-level dedup neu can nen nam o tang keyframe selection/embedding/retrieval, khong nam trong postprocess cua detection.

## Q7. `model_name` va `model_version` - ai set, format gi?

Trong code hien tai, `YOLODetector` set:

```python
model_name = "YOLO"
model_version = str(model_path)
```

Vi du:

```python
YOLODetector(model_path="yolov8n.pt")
```

se tao:

```text
model_name = "YOLO"
model_version = "yolov8n.pt"
```

Neu dung custom weights:

```text
model_version = "runs/detect/train/weights/best.pt"
```

Day la muc du toi thieu de truy vet weights da dung. Neu production can reproducibility chat hon, nen doi format thanh:

```text
model_name = "ultralytics-yolo"
model_version = "yolov8n.pt@sha256:<weights_hash>"
```

hoac tach them `framework_version`, nhung contract hien tai chua co field do.

## Q8. GPU batching - single frame hay batch?

Implementation hien tai chay single-frame theo vong lap tung anh:

```python
for img_path in img_paths:
    detections = detector.detect(image)
```

`chunk_size` trong `run_detection.py` chi la chunk de ghi output dinh ky va gioi han memory, khong phai GPU batch size.

Vay hien tai chua co true batch inference.

Day la quyet dinh theo huong don gian va de debug, khop voi OCR detection style co san trong repo. Neu uu tien throughput tren GPU, buoc tiep theo nen them:

```python
Detector.detect_batch(images, frame_ids, img_paths)
```

va config:

```python
batch_size = 16
```

Trong YOLO/ultralytics, co the truyen list image vao `model.predict(...)` de batch.

## Q9. Exporter - ai quyet dinh format output?

Trong code hien tai, caller quyet dinh export.

Module cung cap:

- `export_results_json(...)` de ghi JSON.
- `detections_to_contracts(...)` / `frame_results_to_contracts(...)` de convert sang `ObjectDetectionResult`.

`run_detection.py` mac dinh export JSON tai:

```text
BackEnd/app/object_detection/output/object_detection_results.json
```

Trong production pipeline, module phia sau nen doc object tu memory/contract, khong nen doc lai file JSON. File JSON phu hop cho smoke test, debug, hoac cache tam.

De tracking chay dung hon, luong production nen la:

```text
Keyframes -> ObjectDetectionResult[] in memory -> Object Tracking
```

Sau do tang persistence co the ghi DB bang bang `ObjectDetection`, khong can SQLite/file trung gian.

## Q10. `timestamp_ms` bi thieu trong `ObjectDetectionResult`

`ObjectDetectionResult` hien tai khong co `timestamp_ms`, va minh khong them field nay de giu contract khop voi `BackEnd/app/contracts/pipeline.py`.

Module phia sau, dac biet Object Tracking, nen join detection voi `FrameMetadata` qua:

```text
ObjectDetectionResult.frame_id -> FrameMetadata.frame_id
```

Tu do lay:

- `timestamp_ms`
- `frame_idx`
- `shot_id`
- `video_id`
- `fps`

Ly do khong embed timestamp vao tung detection:

- Mot frame co nhieu detections, neu moi detection lap lai timestamp/frame metadata se duplicate.
- DB schema da tach `ObjectDetection` va `Frame`.
- Tracking can them `shot_id`, `frame_idx`, `fps`, khong chi timestamp.

Neu sau nay can API response tien loi cho frontend, co the tao response DTO rieng co embed `timestamp_ms`, nhung khong nen sua core `ObjectDetectionResult` neu pipeline contract van tach frame metadata.

## Tom tat nhung gi da code

Module da tao:

- `detector.py`: abstract `Detector`
- `yolo_detector.py`: YOLO detector dung `ultralytics`
- `parser.py`: parse YOLO result thanh `Detection`
- `schemas.py`: `BoundingBox`, `Detection`, `FrameDetectionResult`
- `preprocess.py`: load/resize image
- `postprocess.py`: confidence filter, class filter, optional NMS
- `class_mapper.py`: COCO mapping va `ClassMapper`
- `exporter.py`: JSON export va convert sang `ObjectDetectionResult`
- `utils.py`: scan keyframes, chunking, draw/save annotated image
- `run_detection.py`: entrypoint chay detection hang loat
- `test_detection.py`: smoke test theo `video`, `shot`, `limit`

Dependency da them:

```text
ultralytics>=8.3.0
```

Da verify:

- `python3 -m compileall BackEnd/app/object_detection`
- `python3 -m BackEnd.app.object_detection.run_detection --help`
- import/basic checks cho `ClassMapper`, NMS, exporter contract conversion

Chua verify inference that vi environment hien tai thieu `cv2`/OpenCV runtime, du `opencv-python` da co trong `requirements.txt`.
