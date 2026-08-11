"""NOTE:
File này dùng để chạy Sound Extractor và ASR. Bởi vì nó là 1 quá trình song song, không phụ thuộc 
vào các quá trình khác. File đồng thời làm nhiệm vụ ghi vào database Postgre. 
Chưa lưu vào ElasticSearch vì có thể thực hiện sau
"""

from BackEnd.app.ASR.asr_transcript import ASR_Model
from BackEnd.app.database.postgre_db import PostgreManager
from BackEnd.app.contracts.pipeline import VideoMetadata, TranscriptSegmentResult
from BackEnd.CONFIG import ASR_BATCH_SIZE



def run_asr_model(video: VideoMetadata, db:PostgreManager, asr_model: ASR_Model): 
    trans_list = asr_model.transcript_segment(video)
    for item in trans_list: 
        db.add_transcript_segment(
            segment_id=item.segment_id, 
            video_id=item.video_id, 
            start_ms=item.start_ms, 
            end_ms=item.end_ms,
            language=item.language, 
            text=item.text
        )


if __name__ == "__main__": 
    db = PostgreManager()
    asr_model = ASR_Model()

    videos = db.get_list_video()
    run_asr_model(videos[69], db, asr_model)