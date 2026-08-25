"""Interactive CLI search tool for manual query testing against Elasticsearch.

Official repository location: BackEnd/tests/cli_search.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Fix Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (SCRIPT_DIR / "../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from BackEnd.app.contracts.search import TextSearchHit, TextSearchQuery
from BackEnd.app.database.elasticsearch_db import ElasticsearchManager
from BackEnd.app.database.elasticsearch_documents import ElasticsearchDocumentBuilder
from BackEnd.app.service.text_search_service import TextSearchService

MEDIA_INFO_DIR = PROJECT_ROOT / "data/media-info-aic25-b1/media-info"
OBJECTS_DIR = PROJECT_ROOT / "data/objects-aic25-b1/objects"

SOURCE_NAMES: dict[str, str] = {
    "video_metadata": "🎬 VIDEO METADATA",
    "ocr": "📝 OCR TEXT",
    "transcript": "💬 TRANSCRIPT",
    "caption": "🖼️ CAPTION",
    "object": "🎯 OBJECT DETECTION",
}


def setup_demo_index_if_needed(manager: ElasticsearchManager, index_name: str = "demo_aic_text") -> None:
    """Index the real dataset metadata and object detection JSON files if not present."""
    builder = ElasticsearchDocumentBuilder()

    if not manager.client.indices.exists(index=index_name):
        print(f"📦 Index '{index_name}' chưa tồn tại. Đang nạp dữ liệu Video Metadata...")
        manager.create_index(index_name)

        json_paths = sorted(MEDIA_INFO_DIR.glob("*.json"))
        docs = []

        for path in json_paths:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            from types import SimpleNamespace
            video_obj = SimpleNamespace(
                video_id=path.stem,
                title=raw.get("title"),
                description=raw.get("description"),
                keywords=raw.get("keywords"),
                author=raw.get("author"),
            )
            doc = builder.build_video_metadata_document(video_obj, index_build_id="demo-cli-build")
            if doc:
                docs.append(doc)

        manager.index_documents(docs, index_name=index_name, refresh=True)
        print(f"🎉 Nạp xong {len(docs)} bản ghi Video Metadata vào Elasticsearch!\n")

    # Check if object detection documents are present in index
    obj_count = 0
    try:
        res = manager.client.count(
            index=index_name,
            body={"query": {"term": {"source_type": "object"}}},
        )
        obj_count = res.get("count", 0)
    except Exception:
        obj_count = 0

    if obj_count == 0 and OBJECTS_DIR.exists():
        print(f"📦 Đang tự động nạp các bản ghi Object Detection từ '{OBJECTS_DIR.name}'...")
        obj_docs = []

        # Index object frame JSONs across video folders
        for video_dir in list(OBJECTS_DIR.iterdir())[:300]:
            if video_dir.is_dir():
                for jfile in list(video_dir.glob("*.json"))[:10]:
                    try:
                        with jfile.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        entities = data.get("detection_class_entities", [])
                        if entities:
                            video_id = video_dir.name
                            frame_id = f"{video_id}_{jfile.stem}"
                            from types import SimpleNamespace
                            record = SimpleNamespace(video_id=video_id, frame_id=frame_id)
                            odoc = builder.build_object_document(record, entities, index_build_id="demo-cli-build")
                            if odoc:
                                obj_docs.append(odoc)
                    except Exception:
                        continue

        if obj_docs:
            manager.index_documents(obj_docs, index_name=index_name, refresh=True)
            print(f"🎉 Nạp xong {len(obj_docs)} bản ghi Object Detection vào Elasticsearch!\n")

    # Always ensure active aliases point to this index
    manager.publish_source_aliases(index_name)
    print(f"✅ Đã kết nối Aliases tới Index: '{index_name}'")


class SearchResultFormatter:
    """Format Elasticsearch hits into rich card layouts per source type."""

    @staticmethod
    def _format_ms(ms: int | None) -> str | None:
        if ms is None:
            return None
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:04.1f}s"

    @classmethod
    def format_hit(cls, hit: TextSearchHit, rank: int, total: int) -> str:
        stype = hit.source_type
        stype_label = SOURCE_NAMES.get(stype, stype.upper())

        header = f"┌── [Top {rank}/{total}] Score: {hit.score:.2f} | Nguồn: {stype_label} "
        header = header.ljust(75, "─")

        lines = [header]
        lines.append(f"│ 🎬 Video ID  : {hit.video_id}")

        # Position granularity
        pos_parts = []
        if hit.shot_id:
            pos_parts.append(f"Shot {hit.shot_id}")
        if hit.frame_id:
            pos_parts.append(f"Keyframe {hit.frame_id}")
        if hit.segment_id:
            pos_parts.append(f"Segment {hit.segment_id}")

        time_str = None
        if hit.start_ms is not None or hit.end_ms is not None:
            start = cls._format_ms(hit.start_ms) or "00:00.0s"
            end = cls._format_ms(hit.end_ms) or "?"
            time_str = f"Time Window: {start} -> {end}"
        elif hit.timestamp_ms is not None:
            time_str = f"Timestamp: {cls._format_ms(hit.timestamp_ms)}"

        if time_str:
            pos_parts.append(time_str)

        if pos_parts:
            lines.append(f"│ 📍 Position  : {' | '.join(pos_parts)}")

        # Source specific details
        if stype == "object" and hit.objects:
            lines.append(f"│ 🎯 Matched Objs: {list(hit.objects)}")
        if hit.highlights:
            hl_str = " | ".join(f'"{h}"' for h in hit.highlights[:2])
            lines.append(f"│ 💬 Highlights  : {hl_str}")

        if hit.content:
            preview = hit.content[:140].replace("\n", " ")
            if len(hit.content) > 140:
                preview += "..."
            lines.append(f"│ 📝 Content   : {preview}")

        lines.append("└" + "─" * 74)
        return "\n".join(lines)


class CLISearchSession:
    """Manage interactive CLI state, source selection, and search query loop."""

    MENU_MAP: dict[str, tuple[str, ...]] = {
        "1": ("video_metadata",),
        "2": ("ocr",),
        "3": ("transcript",),
        "4": ("caption",),
        "5": ("object",),
        "6": (),  # All sources
    }

    def __init__(self, service: TextSearchService) -> None:
        self.service = service
        self.selected_source_types: tuple[str, ...] = ()
        self.state: str = "menu"

    def print_menu(self) -> None:
        print("\n" + "=" * 60)
        print("🎯 CHỌN NGUỒN DỮ LIỆU TÌM KIẾM (SEARCH SOURCE MENU)")
        print("=" * 60)
        print(" 1. 🎬 Video Metadata (Tiêu đề, mô tả, từ khóa video)")
        print(" 2. 📝 OCR Text (Chữ xuất hiện trên màn hình video)")
        print(" 3. 💬 Transcript (Lời nói / Phụ đề hội thoại)")
        print(" 4. 🖼️ Caption (Mô tả nội dung khung hình / Clip AI)")
        print(" 5. 🎯 Object Detection (Vật thể thị giác: Người, xe, tòa nhà...)")
        print(" 6. 🚀 TỔ HỢP TẤT CẢ NGUỒN (Mặc định)")
        print("-" * 60)
        print("👉 Nhập số (1-6) hoặc nhập tổ hợp (Ví dụ: '2,5' cho OCR + Object):")

    def process_menu_input(self, raw_choice: str) -> bool:
        choice = raw_choice.strip()
        if not choice or choice == "6":
            self.selected_source_types = ()
            print("✅ Đã chọn: TỔ HỢP TẤT CẢ NGUỒN (All Sources)")
        elif choice in self.MENU_MAP:
            self.selected_source_types = self.MENU_MAP[choice]
            sname = SOURCE_NAMES.get(self.selected_source_types[0], choice)
            print(f"✅ Đã chọn nguồn đơn: {sname}")
        else:
            # Multi-choice selection e.g. "2,5"
            parts = [p.strip() for p in choice.split(",") if p.strip()]
            selected = []
            for p in parts:
                if p in self.MENU_MAP and self.MENU_MAP[p]:
                    selected.extend(self.MENU_MAP[p])
            if selected:
                self.selected_source_types = tuple(set(selected))
                labels = [SOURCE_NAMES.get(s, s) for s in self.selected_source_types]
                print(f"✅ Đã chọn tổ hợp nguồn: {', '.join(labels)}")
            else:
                print("⚠️ Lựa chọn không hợp lệ! Đặt về mặc định Tất cả nguồn.")
                self.selected_source_types = ()

        self.state = "searching"
        print("\n💡 Mẹo: Bạn có thể nhập '-1' bất kỳ lúc nào để đổi lại nguồn tìm kiếm!")
        return True

    @property
    def source_label(self) -> str:
        if not self.selected_source_types:
            return "Tất cả nguồn"
        labels = [SOURCE_NAMES.get(s, s) for s in self.selected_source_types]
        return " + ".join(labels)


def main():
    print("=" * 60)
    print("🔍 CÔNG CỤ TÌM KIẾM TƯƠNG TÁC REAL-TIME (CLI SEARCH 2.0)")
    print("=" * 60)

    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    print(f"🔗 Elasticsearch Server: {es_url}")

    try:
        manager = ElasticsearchManager(elasticsearch_url=es_url)
        if not manager.health_check().get("info"):
            print("❌ Không thể kết nối tới Elasticsearch Server. Hãy đảm bảo Docker ES đang bật!")
            return

        setup_demo_index_if_needed(manager, index_name="demo_aic_text")
        service = TextSearchService(manager=manager)
        session = CLISearchSession(service=service)

    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        return

    while True:
        try:
            if session.state == "menu":
                session.print_menu()
                choice = input("👉 Nhập lựa chọn nguồn: ").strip()
                if choice.lower() in ("exit", "q", "quit"):
                    print("👋 Cảm ơn bạn đã sử dụng CLI Search! Tạm biệt.")
                    break
                session.process_menu_input(choice)
                continue

            # State = searching
            prompt = f"\n🔍 [{session.source_label}] Nhập từ khóa (gõ -1 để đổi nguồn): "
            query_str = input(prompt).strip()

            if not query_str:
                continue
            if query_str == "-1":
                print("🔄 Đã reset! Chuyển về menu chọn nguồn tìm kiếm...")
                session.state = "menu"
                continue
            if query_str.lower() in ("exit", "q", "quit"):
                print("👋 Cảm ơn bạn đã sử dụng CLI Search! Tạm biệt.")
                break

            # Execute fast-path search query
            query = TextSearchQuery(
                query_text=query_str,
                top_k=5,
                source_types=session.selected_source_types,  # type: ignore[arg-type]
                use_highlight=True,
            )
            hits = service.search(query)

            print(f"\n🎯 KẾT QUẢ TÌM KIẾM CHO: '{query_str}' (Top {len(hits)} kết quả)")
            print("=" * 75)

            if not hits:
                print("❌ Không tìm thấy kết quả phù hợp cho nguồn đã chọn.")
            else:
                for i, hit in enumerate(hits, 1):
                    print(SearchResultFormatter.format_hit(hit, i, len(hits)))

            print("=" * 75)

        except KeyboardInterrupt:
            print("\n👋 Đã thoát ứng dụng.")
            break
        except Exception as err:
            print(f"⚠️ Lỗi xử lý truy vấn: {err}\n")


if __name__ == "__main__":
    main()
