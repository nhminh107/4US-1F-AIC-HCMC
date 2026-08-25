"""Script nạp ĐẦY ĐỦ (FULL DATA) từ các file JSON dữ liệu thực tế vào PostgreSQL Database.

Official location: BackEnd/tests/seed_full_postgres.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert

# Fix Windows terminal stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (SCRIPT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from BackEnd.app.database.models import ClassID, Frame, ObjectDetection, Shot, Video
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.pipeline.official_keyframe_ingestion import (
    ingest_official_keyframes,
    load_official_keyframes,
)
from BackEnd.app.pipeline.video_ingestion import ingest_videos

OBJECTS_DIR = PROJECT_ROOT / "data/objects-aic25-b1/objects"


def seed_objects_to_postgres(
    manager: PostgreManager,
    max_videos: int = 0,
    max_frames_per_video: int = 300,
) -> dict[str, int]:
    """Ingest Object Detection JSON files into PostgreSQL classid & objectdetection tables.

    Args:
        max_videos: Maximum number of video directories to process. 0 = unlimited (all videos).
        max_frames_per_video: Maximum number of frame JSON files to process per video (default 300).
                              0 = unlimited.
    """
    if not OBJECTS_DIR.exists():
        print(f"⚠️ Thư mục '{OBJECTS_DIR.name}' không tồn tại.")
        return {"classes_inserted": 0, "detections_inserted": 0}

    print(f"📦 Đang nạp Object Detection JSONs từ '{OBJECTS_DIR.name}' vào PostgreSQL...")
    video_dirs = [d for d in OBJECTS_DIR.iterdir() if d.is_dir()]

    class_map: dict[str, str] = {}  # class_name -> class_id e.g. "Car" -> "c001"
    class_counter = 1

    total_detections = 0
    total_classes = 0

    with manager.session_factory() as session:
        # Pre-load existing frames, shots & classes from DB
        existing_frame_ids = set(session.scalars(select_frame_ids()).all())
        existing_shot_ids = set(session.scalars(select_shot_ids()).all())
        existing_classes = session.query(ClassID).all()
        for c in existing_classes:
            class_map[c.class_name] = c.class_id
            if c.class_id.startswith("c"):
                try:
                    num = int(c.class_id[1:])
                    class_counter = max(class_counter, num + 1)
                except ValueError:
                    pass

        # Scan video folders
        video_list = video_dirs if max_videos <= 0 else video_dirs[:max_videos]
        print(f"   📂 Tổng cộng {len(video_list)} thư mục video sẽ được xử lý (tối đa {max_frames_per_video} frames/video).")
        for v_idx, vdir in enumerate(video_list, 1):
            video_id = vdir.name
            shot_id = f"{video_id}_s1"

            # Ensure Shot exists in DB
            if shot_id not in existing_shot_ids:
                new_shot = Shot(
                    shot_id=shot_id,
                    video_id=video_id,
                    shot_index=1,
                    start_ms=0,
                    end_ms=3600000,
                    start_frame_idx=0,
                    end_frame_idx=90000,
                )
                session.add(new_shot)
                existing_shot_ids.add(shot_id)

            json_files = list(vdir.glob("*.json"))
            if max_frames_per_video > 0:
                json_files = json_files[:max_frames_per_video]

            for jfile in json_files:
                frame_id = f"{video_id}_{jfile.stem}"

                # Ensure frame exists in PostgreSQL DB
                if frame_id not in existing_frame_ids:
                    try:
                        frame_idx = int(jfile.stem)
                    except ValueError:
                        frame_idx = 0
                    new_frame = Frame(
                        frame_id=frame_id,
                        video_id=video_id,
                        shot_id=shot_id,
                        timestamp_ms=frame_idx * 1000,
                        fps=25.0,
                        frame_idx=frame_idx,
                        source="extracted",
                        frame_path=f"data/keyframes/{video_id}/{jfile.stem}.jpg",
                    )
                    session.add(new_frame)
                    existing_frame_ids.add(frame_id)

                try:
                    with jfile.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                entities = data.get("detection_class_entities", [])
                scores = data.get("detection_scores", [])
                boxes = data.get("detection_boxes", [])

                for entity, score, box in zip(entities, scores, boxes):
                    try:
                        score_val = float(score)
                    except (ValueError, TypeError):
                        continue
                    if score_val < 0.3:
                        continue

                    # Ensure class_id exists for this entity name
                    if entity not in class_map:
                        cid = f"c{class_counter:04d}"
                        class_counter += 1
                        new_cls = ClassID(class_id=cid, class_name=entity)
                        session.add(new_cls)
                        class_map[entity] = cid
                        total_classes += 1

                    cid = class_map[entity]
                    y_min, x_min, y_max, x_max = box
                    # Clamp boundary constraints
                    y_min = max(0.0, min(0.99, float(y_min)))
                    x_min = max(0.0, min(0.99, float(x_min)))
                    y_max = max(y_min + 0.01, min(1.0, float(y_max)))
                    x_max = max(x_min + 0.01, min(1.0, float(x_max)))

                    obj_rec = ObjectDetection(
                        frame_id=frame_id,
                        class_id=cid,
                        confidence=score_val,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        model_name="OpenImages_Detector",
                    )
                    session.add(obj_rec)
                    total_detections += 1

            # Batch commit every 50 videos for safety and progress visibility
            if v_idx % 50 == 0:
                session.commit()
                print(f"   ✅ Đã xử lý {v_idx}/{len(video_list)} videos ({total_detections} detections)...")

        session.commit()

    return {"classes_inserted": total_classes, "detections_inserted": total_detections}


def select_frame_ids():
    from sqlalchemy import select

    return select(Frame.frame_id)


def select_shot_ids():
    from sqlalchemy import select

    return select(Shot.shot_id)


def main() -> None:
    print("=" * 60)
    print("🚀 PHÂN ĐOẠN 1: NẠP DỮ LIỆU VÀO POSTGRESQL DATABASE (MAX 300 FRAMES/VIDEO)")
    print("=" * 60)

    try:
        manager = PostgreManager()
        manager.init_db()  # Ensure all tables exist

        # 1. Ingest 873 Video Metadata
        print("▶️ Step 1/3: Nạp Video Metadata (873 videos)...")
        v_res = ingest_videos()
        print(f"   └─ Metadata Videos Inserted: {v_res.get('inserted', 0)} | Skipped: {v_res.get('skipped', 0)}")

        # 2. Ingest Official Keyframes & Shots
        print("▶️ Step 2/3: Nạp Official Keyframes & Shots...")
        try:
            official_frames = load_official_keyframes()
            k_res = ingest_official_keyframes(official_frames)
            print(f"   └─ Keyframes Discovered: {k_res.get('discovered', 0)} | Inserted: {k_res.get('inserted', 0)} | Unchanged: {k_res.get('unchanged', 0)}")
        except Exception as e:
            print(f"   └─ Thông báo Keyframe Ingestion: {e}")

        # 3. Ingest Object Detections (300 frames max per video)
        print("▶️ Step 3/3: Nạp Object Detection JSONs (tối đa 300 frames/video)...")
        o_res = seed_objects_to_postgres(manager, max_videos=0, max_frames_per_video=300)
        print(f"   └─ Classes Inserted: {o_res.get('classes_inserted', 0)} | Detections Inserted: {o_res.get('detections_inserted', 0)}")

        print("\n🎉 HOÀN THÀNH NẠP DỮ LIỆU VÀO POSTGRESQL!")
        print("=" * 60)
        print("👉 Bây giờ bạn có thể đồng bộ Postgres sang Elasticsearch bằng lệnh:")
        print("   python BackEnd/tests/sync_postgres_to_elasticsearch.py")

    except Exception as err:
        print(f"❌ Lỗi nạp dữ liệu PostgreSQL: {err}")



if __name__ == "__main__":
    main()
