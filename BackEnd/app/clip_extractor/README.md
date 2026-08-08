# Module Clip Extractor

Module này nhận **1 record Shot** và chia Shot dài thành nhiều record Clip nhỏ để
đưa vào pipeline Embedding/Caption. Cấu trúc tuân theo quy ước chung:

- code: `BackEnd/app/clip_extractor/`;
- test: `BackEnd/test/clip_extractor/`;
- tổ chức dưới dạng class, hàm ghép pipeline là `ClipExtractor.run()`;
- input/output là record theo data contract;
- có demo và không phụ thuộc package Python bên ngoài.

## Hành vi chuẩn

- Shot dài **không quá 10 giây**: không cần chia, trả về `[]`. Module Embedding có
  thể xử lý trực tiếp Shot này như tài liệu pipeline đã lưu ý.
- Shot dài **hơn 10 giây**: chia thành ít nhất 2 Clip liên tục, mỗi Clip không quá
  10 giây.
- Mốc `start_ms`/`end_ms` của Clip là mốc tuyệt đối trên video gốc.
- Mốc `start_frame_idx`/`end_frame_idx` được nội suy từ đúng record Shot và các
  Clip liên tiếp dùng chung một biên frame, nên không bị hở hoặc chồng.
- Mặc định chỉ tạo metadata, **không lưu thêm file video**, để tránh dư dữ liệu.
- Khi thật sự cần file MP4 cho demo/model ngoài, bật `materialize_files=True`;
  FFmpeg sẽ cắt chính xác từng Clip.

Ví dụ Shot `80000-105000 ms` (25 giây) được chia cân bằng thành 3 Clip:

```text
80000-88334, 88334-96667, 96667-105000
```

Cách chia cân bằng tránh sinh một Clip cuối chỉ dài vài mili-giây.

## Input

```python
shot = {
    "shot_id": "L21_V001_S0003",
    "video_id": "L21_V001",
    "start_ms": 80000,
    "end_ms": 105000,
    "start_frame_idx": 2400,
    "end_frame_idx": 3150,
    "video_path": "data/video/L21_V001.mp4",  # chỉ cần khi xuất MP4
}
```

`video_path` là trường runtime tùy chọn, không bắt buộc thuộc bảng Shot.

## Output theo contract

`ClipExtractor.run()` trả trực tiếp một `list`. Mỗi phần tử trong list là một
`dict` chứa đúng metadata của một Clip để chuyển sang Database:

```python
[
    {
        "clip_id": "L21V001S0003C01",
        "shot_id": "L21_V001_S0003",
        "start_ms": 80000,
        "end_ms": 88334,
        "start_frame_idx": 2400,
        "end_frame_idx": 2650,
        "sampling_fps": 30.0,
        "clip_path": None,
    },
    # ...các Clip tiếp theo
]
```

Đây là đúng 8 cột của bảng `clipwindow`. Khi chỉ sinh metadata, `clip_path` là
`None`; khi bật xuất MP4, trường này chứa đường dẫn file. Các trường runtime như
`video_id`, `duration_ms`, `clip_index` không xuất hiện trong output của `run()`.

`sampling_fps` mặc định được suy ra từ số frame và thời gian của Shot. Nếu nhóm
quy định một FPS lấy mẫu cố định, truyền trực tiếp vào config, ví dụ
`ClipExtractorConfig(sampling_fps=2.0)`. Khi xuất MP4, FFmpeg cũng áp dụng FPS đó.

## Quy ước `clip_id`

Cột `clipwindow.clip_id` chỉ có `varchar(15)`, vì vậy kiểu cũ
`L21_V001_S0003_C0001` không hợp lệ. Module mặc định nén ID thành:

```text
L21_V001_S0003 + Clip 1 -> L21V001S0003C01
```

ID này dài đúng 15 ký tự, ổn định và duy nhất trong cùng Shot. Với Shot ID không
theo mẫu dữ liệu AIC, module tạo một ID băm ổn định vẫn không quá 15 ký tự.

## Ghép vào pipeline

Chạy từ thư mục gốc của project:

```python
from BackEnd.app.clip_extractor import ClipExtractor

extractor = ClipExtractor()
clip_rows = extractor.run(shot_record)
print(clip_rows)
```

`clip_rows` đã là `list[dict]`, vì vậy Database module có thể nhận và insert
trực tiếp; không cần gọi thêm `to_contract()`.

Nếu terminal đang đứng ngay trong thư mục `BackEnd/`, import bằng
`from app.clip_extractor import ClipExtractor` để tránh lỗi
`ModuleNotFoundError: No module named 'BackEnd'`.

Module này **không tự insert SQL**. Database module nhận `clip_rows` và lưu theo
transaction của pipeline chung.

Nếu nhóm thống nhất quy tắc ID khác, truyền factory riêng; giá trị trả về vẫn
phải duy nhất và không quá 15 ký tự:

```python
extractor = ClipExtractor(
    clip_id_factory=lambda shot, index: "CLIP%011d" % index
)
```

## Xuất file MP4 khi cần

Máy cần có `ffmpeg` và `ffprobe` trong `PATH`.

```python
from BackEnd.app.clip_extractor import ClipExtractor, ClipExtractorConfig

extractor = ClipExtractor(
    ClipExtractorConfig(
        materialize_files=True,
        output_root="data/clips",
    )
)

clips = extractor.run(shot_record)
```

File được lưu theo cấu trúc:

```text
data/clips/<video_id>/<clip_id>.mp4
```

## Demo

Chỉ sinh metadata:

```bash
python -m BackEnd.app.clip_extractor.demo \
  BackEnd/examples/clip_extractor/shot.json
```

Sinh cả MP4:

```bash
python -m BackEnd.app.clip_extractor.demo \
  BackEnd/examples/clip_extractor/shot.json \
  --materialize \
  --video-path data/video/L21_V001.mp4 \
  --output-root data/clips
```

## Chạy test

Không cần cài `pytest`:

```bash
python -m unittest discover -s BackEnd/test/clip_extractor -v
```

Test gồm validation contract, trường hợp Shot ngắn/dài, kiểm tra không có khoảng
trống/chồng lấn theo cả mili-giây và frame, giới hạn ID/đường dẫn, cùng kiểm thử
tích hợp FFmpeg bằng video thật.

PowerShell, chạy từ thư mục gốc project:

```powershell
$videoPath = "C:\Users\GIA HIEN\Downloads\Videos_L21_a\video\L21_V031.mp4"
$env:CLIP_EXTRACTOR_TEST_VIDEO = (Resolve-Path $videoPath).Path
python -m unittest discover -s BackEnd/test/clip_extractor -t . -v
```

Kết quả đầy đủ phải là `Ran 7 tests` và `OK`, không có `skipped=1`.
