from __future__ import annotations

from sqlalchemy import create_engine, text

from BackEnd.app.frame_context.postgres_source import load_frame_evidence


def test_loads_shot_caption_ocr_and_object_evidence() -> None:
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE video (video_id TEXT PRIMARY KEY)"))
        connection.execute(text(
            "CREATE TABLE frame (frame_id TEXT PRIMARY KEY, video_id TEXT, "
            "shot_id TEXT, frame_idx INTEGER, timestamp_ms INTEGER)"
        ))
        connection.execute(text(
            "CREATE TABLE shot (shot_id TEXT PRIMARY KEY, video_id TEXT, "
            "start_ms INTEGER, end_ms INTEGER)"
        ))
        connection.execute(text(
            "CREATE TABLE caption (caption_id TEXT PRIMARY KEY, shot_id TEXT, caption_text TEXT)"
        ))
        connection.execute(text("CREATE TABLE ocr (frame_id TEXT, n INTEGER, text TEXT)"))
        connection.execute(text(
            "CREATE TABLE classid (class_id TEXT PRIMARY KEY, class_name TEXT)"
        ))
        connection.execute(text(
            "CREATE TABLE objectdetection (frame_id TEXT, class_id TEXT, confidence REAL)"
        ))
        connection.execute(text("INSERT INTO video VALUES ('L01_V001')"))
        connection.execute(text("INSERT INTO shot VALUES ('S1', 'L01_V001', 0, 5000)"))
        connection.execute(text(
            "INSERT INTO frame VALUES ('F1', 'L01_V001', NULL, 90, 3000)"
        ))
        connection.execute(text("INSERT INTO caption VALUES ('C1', 'S1', 'Sửa xe')"))
        connection.execute(text("INSERT INTO ocr VALUES ('F1', 0, 'HONDA')"))
        connection.execute(text("INSERT INTO classid VALUES ('c1', 'motorcycle')"))
        connection.execute(text(
            "INSERT INTO objectdetection VALUES ('F1', 'c1', 0.9)"
        ))

    records = load_frame_evidence(engine, video_ids=["L01_V001"])

    assert len(records) == 1
    assert records[0].captions == ("Sửa xe",)
    assert records[0].ocr_texts == ("HONDA",)
    assert records[0].object_labels == ("motorcycle",)
