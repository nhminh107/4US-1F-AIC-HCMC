from BackEnd.app.shot_extractor import ShotExtractor
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.contracts.pipeline import ShotMetadata


def extract_shot(video_id: str, db: PostgreManager, extractor: ShotExtractor): 
    shots = extractor.extract(video_id)
    for shot in shots:
        db.add_shot(
            shot_id=shot.shot_id, 
            video_id=video_id, 
            shot_index=shot.shot_index, 
            start_ms=shot.start_ms, 
            end_ms=shot.end_ms, 
            start_frame_idx=shot.start_frame_idx, 
            end_frame_idx=shot.end_frame_idx
        )

if __name__ == "__main__": 
    db = PostgreManager()
    extractor = ShotExtractor()

    videos = db.get_list_video()
    extract_shot(video_id='L23_V005', db=db, extractor=extractor)