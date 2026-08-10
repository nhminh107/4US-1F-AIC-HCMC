4. Các Lỗi & Rủi Ro Tiềm Ẩn Khi Chạy Với Big Data Trong Cuộc Thi AIC
Trong cuộc thi AIC (AI Challenge HCMC), quy mô dữ liệu cực kỳ lớn: hàng nghìn video, hàng triệu keyframes, OCR, ASR segments và VLM Captions. Qua kiểm tra codebase, có 5 NGUY CƠ NGHẼN MẠCH VÀ TRỤY HỆ THỐNG sau:

🚨 Rủi ro 1: Out-Of-Memory (OOM) ở Python App Server khi Sync Dữ Liệu
Vị trí code: 

text_search_service.py:L52-L102
Nguyên nhân:
python
frames = session.scalars(select(Frame).options(selectinload(Frame.ocr_records))).all()
for frame in frames: ... documents.append(doc)
transcripts = session.scalars(select(TranscriptSegment)).all()
captions = session.scalars(select(Caption)...).all()
Hàm sync_from_postgres dùng .all() để tải toàn bộ triệu bản ghi Postgres vào RAM của Python, sau đó dồn tất cả TextIndexDocument vào mảng documents trước khi ghi vào ES.
Hậu quả với BigData: Với dataset hàng triệu frames/captions, mảng documents sẽ chiếm tới vài chục GB RAM, gây ra lỗi MemoryError hoặc tiến trình Python bị OS KILL (OOM Killed).
🚨 Rủi ro 2: ES Node cạn kiệt bộ nhớ Heap & RAM do Shingle Analyzer & Nested Objects
Vị trí code: 

elasticsearch_db.py:L86-L91, L128-L139
Nguyên nhân:
aic_shingle (min:2, max:3 + preserve_original: True ở ASCII folding) nhân số lượng Term lên gấp nhiều lần cho mỗi từ tiếng Việt.
Mẫu document OCR chứa regions khai báo kiểu nested. Trong Lucene/ES, mỗi nested object được lưu thành 1 hidden document riêng biệt. Nếu 1 frame có 50 đoạn OCR text $\rightarrow$ 1 doc tạo ra 51 internal Lucene documents.
Hậu quả với BigData: Inverted Index của Elasticsearch bị phình to cực nhanh (Index Bloat), ngốn bộ nhớ RAM/Heap của ES Cluster. ES node có thể bị sập do java.lang.OutOfMemoryError: Java heap space.
🚨 Rủi ro 3: Tốc độ Tìm Kiếm (Query Latency) Bị Trễ Nặng Lúc Thi (Real-time Search Bottleneck)
Vị trí code: 

elasticsearch_db.py:L312-L344
Nguyên nhân:
Fuzzy Search (use_fuzzy=True): Khi chạy trên index có shingle n-gram cực lớn, query fuzzy AUTO sẽ ép ES tạo Levenshtein Automata kiểm tra hàng triệu terms.
Highlighting (use_highlight=True): Được bật mặc định cho tất cả queries trên field content và regions.text.
Nested Query: Query regions.text lồng trong nested path đòi hỏi ES phải join các hidden documents tại runtime.
Hậu quả với BigData: Thời gian phản hồi cho 1 câu search từ 10ms có thể vọt lên >3-10 giây, gây đơ giao diện tìm kiếm của đội khi đang thi đấu bấm thời gian.
🚨 Rủi ro 4: Nghẽn Bulk Indexing & Lỗi HTTP Payload Too Large
Vị trí code: 

elasticsearch_db.py:L224-L235
, 

text_search_service.py:L105-L110
Nguyên nhân:
Khi gọi index_documents(..., refresh=True), ES phải flush và refresh Lucene segment sau mỗi chunk 500 docs.
Batch size chỉ đếm theo số lượng bản ghi chunk_size = 500 mà không giới hạn dung lượng byte. Bản ghi OCR kèm hàng chục regions tọa độ có dung lượng lớn gấp nhiều lần bản ghi metadata.
Hậu quả với BigData: refresh=True làm tốc độ đẩy data chậm đi 10x - 50x. Nếu batch payload quá nặng sẽ bị ES chối từ với lỗi 413 Request Entity Too Large hoặc ConnectionTimeout.
🚨 Rủi ro 5: Hủy Toàn Bộ Quá Trình Sync Khi Có Duplicate Document ID hoặc Lỗi Nhỏ
Vị trí code: 

elasticsearch_db.py:L205-L207
Nguyên nhân:
python
if len(set(doc_ids)) != len(doc_ids):
    raise ValueError("Duplicate doc_id values are not allowed in one batch.")
Hậu quả: Khi ingest Big Data ngẫu nhiên từ nhiều nguồn, nếu DB có 2 bản ghi trùng ID trong 1 batch, hàm lập tức throw ValueError và hủy toàn bộ tiến trình Sync, bỏ dở việc đánh index.