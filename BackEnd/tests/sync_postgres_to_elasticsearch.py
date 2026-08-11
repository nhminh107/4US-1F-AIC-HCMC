"""Script đồng bộ dữ liệu từ PostgreSQL Database sang Elasticsearch Index (Wrapper cho Pipeline Module).

Wrapper location: BackEnd/tests/sync_postgres_to_elasticsearch.py
Official pipeline module: BackEnd/app/pipeline/sync_elasticsearch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Reconfigure stdout for Windows terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (SCRIPT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from BackEnd.app.pipeline.sync_elasticsearch import run_elasticsearch_sync


def main() -> None:
    try:
        run_elasticsearch_sync()
    except Exception as err:
        print(f"❌ Lỗi trong quá trình đồng bộ: {err}")


if __name__ == "__main__":
    main()
