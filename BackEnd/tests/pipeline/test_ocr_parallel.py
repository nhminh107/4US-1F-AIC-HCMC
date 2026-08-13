from BackEnd.app.pipeline.ocr_parallel import (
    _load_completed_video_ids,
    _split_round_robin,
)


def test_split_round_robin_produces_non_overlapping_shards() -> None:
    shards = _split_round_robin(["v1", "v2", "v3", "v4", "v5"], 2)

    assert shards == [["v1", "v3", "v5"], ["v2", "v4"]]
    assert set(shards[0]).isdisjoint(shards[1])


def test_load_completed_video_ids_combines_resume_logs(tmp_path) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text(
        "[ocr-worker] 1/2 committed V001, rows=3, source=official\n",
        encoding="utf-8",
    )
    second.write_text(
        "[ocr-worker] 2/2 committed V002, rows=4, source=official\n"
        "[ocr-worker] 1/1 committed V001, rows=3, source=official\n",
        encoding="utf-8",
    )

    completed = _load_completed_video_ids([first, second, tmp_path / "missing.log"])

    assert completed == {"V001", "V002"}
