from __future__ import annotations

from sqlalchemy import create_engine, text

from BackEnd.app.text_embedding.sources import asr_documents


def test_loads_only_non_empty_asr_segments_in_time_order() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE transcriptsegment (segment_id TEXT PRIMARY KEY, "
            "video_id TEXT, start_ms INTEGER, end_ms INTEGER, text TEXT)"
        ))
        connection.execute(text(
            "INSERT INTO transcriptsegment VALUES "
            "('S2', 'V1', 1000, 2000, 'thứ hai'), "
            "('S1', 'V1', 0, 1000, 'xin chào'), "
            "('S3', 'V1', 2000, 3000, '   '), "
            "('S4', 'V2', 0, 1000, 'video khác')"
        ))

    documents = asr_documents(engine, video_ids=["V1"])

    assert [document.segment_id for document in documents] == ["S1", "S2"]
    assert documents[0].start_ms == 0
    assert documents[0].text == "xin chào"
