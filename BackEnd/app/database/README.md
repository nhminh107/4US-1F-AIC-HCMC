## Hướng dẫn kết nối Database

Tạo 1 user mới trong PostgreDB, chạy script, thay tên phù hợp vào .env
Chạy test để đảm bảo khởi tạo thành công 

Test elasticsearch
python -m pytest BackEnd/tests/database/test_elasticsearch.py BackEnd/tests/database/test_elasticsearch_documents.py -q

