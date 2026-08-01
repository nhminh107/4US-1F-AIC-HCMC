# PostgreSQL database

## 1. Cài và khởi động PostgreSQL (Ubuntu)

```bash
sudo apt update
sudo apt install postgresql
sudo systemctl enable --now postgresql
```

Trong Conda `DL_Env`, cài dependency của project và PostgreSQL driver nếu máy chưa có:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_Env
python -m pip install -r requirements.txt
python -m pip install "psycopg[binary]"
```

## 2. Tạo user và database

```bash
sudo -u postgres psql
```

```sql
CREATE USER nhminh107 WITH PASSWORD 'CHANGE_ME';
CREATE DATABASE "4US-1F-AIC" OWNER nhminh107;
\q
```

Khởi tạo các bảng bằng chính application user:

```bash
psql -h 127.0.0.1 -U nhminh107 -d "4US-1F-AIC" \
  -f BackEnd/app/database/postgre_script.sql
```

Nếu bảng đã được tạo bởi user `postgres`, cấp quyền lại:

```bash
sudo -u postgres psql -d "4US-1F-AIC" -c \
  'GRANT USAGE ON SCHEMA public TO nhminh107; GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nhminh107;'
```

## 3. Cấu hình kết nối

Tạo `.env` tại thư mục gốc project:

```env
DATABASE_URL=postgresql+psycopg://nhminh107:CHANGE_ME@127.0.0.1:5432/4US-1F-AIC
```

Không commit `.env`. Nếu mật khẩu có ký tự đặc biệt, cần URL-encode mật khẩu.

## 4. Kiểm tra kết nối

Chạy từ thư mục gốc project trong `DL_Env`:

```bash
python -c "from BackEnd.app.database.postgre_db import Postgre_Manager; print(Postgre_Manager().engine.connect().exec_driver_sql('SELECT 1').scalar_one())"
```

Kết quả thành công là `1`.
