# OCR module

The canonical OCR API consumes and returns shared pipeline contracts only:

```python
from BackEnd.app.ocr import OCRService
from BackEnd.app.contracts.pipeline import FrameMetadata, OCRResult

service = OCRService()

single_results: list[OCRResult] = service.process_frame(frame)
batch_results: list[OCRResult] = service.process_batch(frames)
```

`process_batch()` performs detection for the image batch, perspective-corrects
the detected polygons in memory, then recognizes all valid crops through one
shared recognition batch. Results remain grouped by input frame order through
`OCRResult.frame_id`; `OCRResult.n` is the zero-based reading-order index within
that frame.

## Default profile

- Detector: `PP-OCRv5_mobile_det`
- Recognizer: VietOCR `vgg_transformer`
- Language: Vietnamese (`vi`)
- Detection input limit: 1280 pixels on the longest side
- Recognition batch size: 32
- Device: GPU when PaddlePaddle can access one, otherwise CPU

The legacy Paddle IR executor is selected by default because the currently
installed PaddlePaddle 3.3.1 raises `ConvertPirAttribute2RuntimeAttribute` for
the cached OCR detector under the new IR executor. This workaround is explicit
through `OCRConfig.disable_new_ir` and can be benchmarked again after a runtime
upgrade.

All settings are explicit in `OCRConfig`. Setting `device="gpu:0"` fails clearly
when PaddlePaddle cannot access CUDA; an explicitly requested GPU never falls
back silently.

Model objects are initialized lazily and reused for the lifetime of the service.
If local model directories are not configured, PaddleOCR may download its
official weights on first inference. Production deployments should provide
versioned local model directories to make runs reproducible.

VietOCR is the default recognition backend because a local comparison on the
same news-frame crops preserved Vietnamese accents substantially better than
`latin_PP-OCRv5_mobile_rec`. The lighter Paddle recognizer remains available:

```python
config = OCRConfig(
    recognition_backend="paddleocr",
    recognition_model_name="latin_PP-OCRv5_mobile_rec",
)
```

For reproducible VietOCR inference, set `recognition_config_path` to a local
YAML configuration and `recognition_model_dir` to the local `.pth` weight file.

The old dictionary-based scripts under `detection/` and `recognition/` are not
the pipeline boundary. They remain only for compatibility with earlier manual
experiments; new code should use `OCRService`, `run_ocr()`, or
`run_ocr_batch()`.
