# Online VQA Handler

`VQAHandler` receives frame evidence already selected by the online retrieval
pipeline and an injected vision-language model client. It does not rerun OCR,
detection, tracking, or whole-dataset retrieval.

The handler validates local evidence paths, removes duplicate
`(video_id, timestamp_ms)` images, limits prompt size, and preserves the model,
prompt, and evidence versions in `VQAResponse`. The injected `VQAModelClient`
protocol keeps orchestration independent from one VLM provider or checkpoint.

```python
handler = VQAHandler(vlm_client)
response = handler.answer(
    VQARequest(
        question="What vehicle is visible?",
        evidence=(
            VQAEvidence(
                image_path="data/keyframes/L01_V001/001.jpg",
                video_id="L01_V001",
                frame_id="F001",
                timestamp_ms=1200,
            ),
        ),
    )
)
```
