from BackEnd.app.keyframe_extractor import KeyframeExtractor
from BackEnd.app.database.postgre_db import PostgreManager

def extract_keyframes(video_id: str, db: PostgreManager, extractor: KeyframeExtractor): 
    kfs = extractor.extract_for_video(
        video_id=video_id,
        shots=db.get_list_shot_in_video(video_id)
    )

    for kf in kfs: 
        db.add_frame(
            frame_id=kf.frame_id, 
            video_id=video_id, 
            shot_id=kf.shot_id,
            timestamp_ms=kf.timestamp_ms,
            fps = kf.fps, 
            frame_idx=kf.frame_idx,
            source=kf.source,
            n = kf.n, 
            pts_time=kf.pts_time, 
            frame_path=str(kf.frame_path),
            width=kf.width,
            height=kf.height
        )

if __name__ == "__main__": 
    db = PostgreManager()
    extractor = KeyframeExtractor()

    extract_keyframes('L21_V005', db, extractor)