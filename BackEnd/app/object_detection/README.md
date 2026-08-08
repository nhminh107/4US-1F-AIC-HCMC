# Object Detection Module

Module nay phu trach phat hien object tren keyframe, chuan hoa ket qua detection, va export sang JSON hoac contract `ObjectDetectionResult` de cac module phia sau nhu tracking/database dung tiep.

## Vi tri detection chinh

Detection chinh nam o:

```text
BackEnd/app/object_detection/yolo_detector.py
```

Class chinh:

```python
YOLODetector
```

Function chinh:

```python
YOLODetector.detect(image, frame_id=None, img_path=None)
```

Day la noi goi model YOLO thong qua `ultralytics`, nhan anh OpenCV/numpy array, sau do tra ve danh sach `Detection`.

Entry point chay detection hang loat nam o:

```text
BackEnd/app/object_detection/run_detection.py
```

Function quan trong:

```python
run_detection()
run_detection_on_chunk()
```

`run_detection()` scan keyframes, khoi tao `YOLODetector`, xu ly theo chunk, va ghi JSON output. `run_detection_on_chunk()` xu ly mot danh sach anh da co san va phu hop de goi tu pipeline khac.

## Cai dat dependency

Chay tu root project:

```bash
pip install -r requirements.txt
```

Dependency lien quan truc tiep den object detection:

```text
opencv-python
numpy>=1.24.0
torch>=2.0.0
ultralytics>=8.3.0
pytest>=8.0.0
```

Luu y: neu dung model mac dinh `yolov8n.pt`, `ultralytics` co the tu tai weights trong lan chay dau tien.

## Cach chay detection

Chay nhanh tren mot so keyframe dau tien:

```bash
cd 4US-1F-AIC-HCMC
python3 -m BackEnd.app.object_detection.run_detection --limit 20
```

Chay voi model cu the:

```bash
python3 -m BackEnd.app.object_detection.run_detection --model yolov8n.pt --limit 20
```

Chi detect mot so class:

```bash
python3 -m BackEnd.app.object_detection.run_detection --classes person,car --limit 20
```

Chi detect theo `class_id`:

```bash
python3 -m BackEnd.app.object_detection.run_detection --class-ids c000,c002 --limit 20
```

Output mac dinh:

```text
BackEnd/app/object_detection/output/object_detection_results.json
```

## Cach chay smoke test inference

Smoke test co goi detector that:

```bash
python3 -m BackEnd.app.object_detection.test_detection --limit 10
```

Loc theo video:

```bash
python3 -m BackEnd.app.object_detection.test_detection --video K01_V001 --limit 10
```

Loc theo shot:

```bash
python3 -m BackEnd.app.object_detection.test_detection --video K01_V001 --shot-min 1 --shot-max 5
```

## Cach chay unit test

Unit test khong load YOLO/OpenCV, chi kiem tra logic deterministic:

```bash
python3 -m BackEnd.app.object_detection.test
```

Neu dung pytest:

```bash
pytest BackEnd/app/object_detection/test.py
```

File test cover:

- bbox order va normalized coordinates
- format `class_id` dang `cNNN`
- confidence filter
- class filter
- IoU va NMS
- contract `ObjectDetectionResult`

## Cac file chinh va chuc nang

```text
object_detection/
├── __init__.py
├── detector.py
├── yolo_detector.py
├── parser.py
├── preprocess.py
├── postprocess.py
├── class_mapper.py
├── exporter.py
├── schemas.py
├── utils.py
├── run_detection.py
├── test_detection.py
├── test.py
└── README.md
```

### `detector.py`

Dinh nghia abstract interface `Detector`.

Chuc nang:

- Tao contract chung cho moi detector implementation.
- Bat buoc detector phai co `detect(...)`.
- Giup sau nay them model khac nhu RT-DETR/GroundingDINO ma pipeline khong phai sua nhieu.

### `yolo_detector.py`

Implementation detection chinh bang YOLO.

Class/function quan trong:

```python
YOLODetector
YOLODetector.detect(...)
```

Chuc nang:

- Load model bang `ultralytics.YOLO`.
- Truyen confidence threshold, IoU threshold, device, class filter vao model.
- Goi `parser.parse_yolo_result()` de doi output YOLO thanh `Detection`.

### `parser.py`

Parse output model ve schema noi bo.

Function quan trong:

```python
parse_yolo_result(...)
```

Chuc nang:

- Doc `boxes.xyxy`, `boxes.conf`, `boxes.cls` tu output YOLO.
- Tao object `Detection`.
- Gan `class_id`, `class_name`, `model_name`, `model_version`.

### `schemas.py`

Dinh nghia data model noi bo cua module.

Class quan trong:

```python
BoundingBox
Detection
FrameDetectionResult
```

Chuc nang:

- `BoundingBox`: luu bbox pixel theo format noi bo `[x_min, y_min, x_max, y_max]`.
- `Detection`: mot object duoc detect trong mot frame.
- `FrameDetectionResult`: tat ca detection cua mot anh.

Luu y quan trong:

- JSON/debug output dung bbox pixel.
- `ObjectDetectionResult`/DB dung bbox normalized `[0.0, 1.0]`.
- Khi convert sang contract, field order la `x_min, x_max, y_min, y_max` theo `BackEnd/app/contracts/pipeline.py`.

### `preprocess.py`

Tien xu ly anh.

Function quan trong:

```python
load_image(...)
resize_image(...)
preprocess(...)
```

Chuc nang:

- Doc anh bang OpenCV.
- Resize theo `max_side` neu can.
- Tra ve numpy array BGR cho detector.

### `postprocess.py`

Xu ly sau detection.

Function quan trong:

```python
filter_by_confidence(...)
filter_by_classes(...)
filter_by_class(...)
clip_detections(...)
iou(...)
nms(...)
resolve_class_indices(...)
```

Chuc nang:

- Loc theo confidence.
- Loc theo class name/class id.
- Clip bbox ve bien anh.
- Tinh IoU.
- NMS tuy chon.
- Convert class names/class ids thanh class indices de truyen vao YOLO.

Luu y: YOLO/ultralytics da co built-in NMS. `nms(...)` trong file nay la helper tuy chon, khong mac dinh chay lan hai trong production path.

### `class_mapper.py`

Quan ly mapping class index, class id, va class name.

Class/function quan trong:

```python
ClassMapper
ClassMapper.class_id_for_index(...)
ClassMapper.class_name_for_index(...)
ClassMapper.to_class_metadata(...)
```

Chuc nang:

- Dung format `class_id = cNNN`.
- Vi du `person -> c000`, `car -> c002`.
- Fallback class set la COCO 80 classes.
- Export mapping sang `ClassMetadata`.
- Warn neu class index vuot gioi han `c999`.

### `exporter.py`

Export ket qua detection.

Function quan trong:

```python
export_results_json(...)
detections_to_contracts(...)
frame_results_to_contracts(...)
class_metadata(...)
```

Chuc nang:

- Ghi JSON de debug/cache tam.
- Convert `Detection`/`FrameDetectionResult` sang `ObjectDetectionResult`.
- Chuan hoa bbox pixel thanh normalized bbox khi export sang contract.

Trong production pipeline, module phia sau nen dung object trong memory/contract thay vi doc lai JSON.

### `utils.py`

Ham phu tro.

Function quan trong:

```python
scan_keyframes(...)
iter_chunks(...)
default_frame_id(...)
draw_detections(...)
save_annotated_image(...)
```

Chuc nang:

- Scan anh trong `data/keyframes`.
- Chia chunk.
- Tao frame id tam tu ten file.
- Ve bbox len anh de debug.

### `run_detection.py`

Script/entrypoint chay detection hang loat.

Function quan trong:

```python
run_detection(...)
run_detection_on_chunk(...)
main()
```

Chuc nang:

- Scan keyframes.
- Khoi tao `YOLODetector`.
- Chay detection theo chunk.
- Export JSON sau moi chunk de tranh mat ket qua neu bi dung giua chung.

### `test_detection.py`

Smoke test co goi model that.

Chuc nang:

- Chay detection tren tap con keyframes.
- Ho tro filter theo video, shot range, limit.
- In so object detect duoc theo tung frame.

### `test.py`

Unit test logic.

Chuc nang:

- Kiem tra bbox normalized/order.
- Kiem tra `class_id` format.
- Kiem tra confidence/class filter.
- Kiem tra IoU/NMS.
- Kiem tra contract `ObjectDetectionResult`.

## Ly do chia file

Module duoc chia file theo boundary ro rang:

- `detector.py` va `yolo_detector.py`: tach interface voi implementation model.
- `schemas.py`: gom data structure de cac file khac dung chung.
- `parser.py`: tach logic parse output model, vi moi model co output format khac nhau.
- `preprocess.py`: tach viec doc/resize anh khoi detector.
- `postprocess.py`: gom cac buoc xu ly detection khong phu thuoc model.
- `class_mapper.py`: tach mapping class de DB/tracking/retrieval co chung mot dinh danh.
- `exporter.py`: tach output format va contract conversion khoi inference.
- `run_detection.py`: giu orchestration/CLI rieng, khong tron vao detector.
- `test.py` va `test_detection.py`: tach unit test nhanh voi smoke test inference that.

Cach chia nay giup thay YOLO bang model khac ma van giu duoc pipeline chung:

```text
preprocess -> detector implementation -> parser -> postprocess -> exporter/contract
```
