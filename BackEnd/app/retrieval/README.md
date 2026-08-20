# Online Text/Object Retrieval and VQA

This package implements the online branches assigned to text/object tracking
retrieval. It only reads artifacts created by the offline pipeline; it never
reruns OCR, object detection, or tracking.

## Text retrieval

`TextRetrievalTool` accepts the existing `TextSearchQuery`, calls
`TextSearchService` (Elasticsearch), and converts OCR, transcript, caption, and
metadata hits into `RetrievalCandidate` records for the fusion stage.

```python
tool = TextRetrievalTool()
candidates = tool.search(TextSearchQuery(query_text="Circle K", top_k=20))
```

## Object and tracking retrieval

`ObjectTrackingRetrievalTool` queries persisted PostgreSQL detections and
ByteTrack summaries. Each object constraint is an independently usable signal;
the candidate-fusion/temporal stage is responsible for enforcing combinations
such as `person AND motorbike` across a shared shot or time window.

```python
tool = ObjectTrackingRetrievalTool(session_factory)
candidates = tool.search(
    ObjectRetrievalRequest(
        objects=(
            ObjectConstraint("Person", minimum_confidence=0.5),
            ObjectConstraint("Motorcycle", minimum_confidence=0.5),
        ),
        top_k=50,
    )
)
```

Official detections may not belong to an extracted shot, so detection results
use `frame_id` and `timestamp_ms`. Track results use `shot_id`, `start_ms`, and
`end_ms`. This distinction is intentional.
