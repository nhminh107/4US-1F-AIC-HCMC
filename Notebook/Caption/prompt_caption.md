## YÊU CẦU TRIỂN KHAI NOTEBOOK CAPTION 
Description: Hoàn thiện 1 Notebook caption theo yêu cầu dưới đây 

### File Input của Notebook
- video.txt: Bao gồm: <video_id>, URL R2 cloudflare video
- Bảng Shot trong database: Bao gồm: shot_id, video_id, shot_index, start_ms, end_ms, start_frame_idx, end_frame_idx
- API Key của FPT Cloud AI, mô hình VLM 

### Output của Notebook
- 1 file SQL duy nhất, chứa các script insert vào bảng Caption, bao gồm (caption_id, shot_id, caption_text)
- Bạn có thể tạo các checkpoint thoải mái, miễn là sau cùng có file SQL là được
### Yêu cầu triển khai
- Có hỗ trợ cơ chế tải video song song (n worker), n worker gọi API và đọc ghi SQL, n worker cắt shot từ video. Tuỳ bạn sắp xếp. Cái này để tối ưu hoá hiệu năng và tốc độ 
- 2 file input tôi sẽ sort theo shot_id và video_id. Bạn cần triển khai thêm cơ chế chọn khoảng để chạy: Ví dụ tôi chọn 0:8000, chạy hết 8000 shot đầu tiên là xong. Sỡ dĩ cần cơ chế này vì Kaggle chỉ cho chạy 12 tiếng / notebook. Nên tôi sẽ chia làm nhiều notebook để tách ra
- Prompt cho VLM cần hướng tới mô tả chi tiết khung cảnh, sự vật, hiện tượng, không cần OCR 

### Yêu cầu cell test
Hiện tại tôi chưa có file data nên bạn hãy tạo 1 cell mock test ở dưới cùng. Có download video và caption shot thật cho tôi kiểm tra 