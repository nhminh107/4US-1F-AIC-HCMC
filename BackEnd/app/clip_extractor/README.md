# Module Clip Extractor

Module này nhận **1 record Shot** và chia Shot dài thành nhiều record Clip nhỏ để
đưa vào pipeline Embedding/Caption. Cấu trúc tuân theo quy ước chung:

- code: `BackEnd/app/clip_extractor/`;
- test: `BackEnd/test/clip_extractor/`;
- tổ chức dưới dạng class, hàm ghép pipeline là `ClipExtractor.run()`;
- input/output là record theo data contract;
- có demo và không phụ thuộc package Python bên ngoài.

## Hành vi chuẩn

- Shot dài **không quá 10 giây**: tạo đúng một Clip loại `full_shot` bao phủ toàn
  bộ Shot.
- Shot dài **hơn 10 giây**: tạo các cửa sổ 10 giây, mặc định stride 8 giây nên
  hai cửa sổ lân cận có phần ngữ cảnh chồng nhau.
- Mốc `start_ms`/`end_ms` của Clip là mốc tuyệt đối trên video gốc.
- Mốc `start_frame_idx`/`end_frame_idx` được nội suy từ đúng record Shot.
- Mặc định chỉ tạo metadata, **không lưu thêm file video**, để tránh dư dữ liệu.
- Khi thật sự cần file MP4 cho demo/model ngoài, bật `materialize_files=True`;
  FFmpeg sẽ cắt chính xác từng Clip.

Ví dụ Shot `80000-105000 ms` (25 giây) tạo thành 3 cửa sổ:

```text
80000-90000, 88000-98000, 95000-105000
```

Cửa sổ cuối được căn theo cuối Shot. Module chỉ gộp một cửa sổ cuối gần-trùng
khi các cửa sổ trước đó vẫn bảo đảm toàn bộ Shot được phủ, nên không làm mất dữ
liệu ở đầu hoặc cuối Shot.

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
