"""Interactive terminal CLI for testing Elasticsearch text retrieval in AIC HCMC 2026."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BackEnd.CONFIG import DATA_ROOT
from BackEnd.app.contracts.search import TextSearchHit, TextSearchQuery, TextSourceType
from BackEnd.app.service.text_search_service import TextSearchService

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def format_timestamp(ms: int | None) -> str:
    """Format milliseconds into MM:SS.mmm format."""
    if ms is None:
        return "N/A"
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:06.3f} ({seconds:.2f}s)"


def find_local_video(video_id: str | None) -> Path | None:
    if not video_id:
        return None
    for folder_name in ("video", "Videos_L21_a", "Videos_L23_a", "Videos_L21_b", "Videos_L23_b"):
        for sub in ("", "video"):
            base_dir = DATA_ROOT / folder_name / sub if sub else DATA_ROOT / folder_name
            if base_dir.is_dir():
                cand = base_dir / f"{video_id}.mp4"
                if cand.is_file():
                    return cand
    return None


def find_local_keyframe(video_id: str | None, frame_id: str | None) -> Path | None:
    if not video_id or not frame_id:
        return None
    parts = frame_id.split("_")
    num_str = parts[-1] if len(parts) >= 3 and parts[-1].isdigit() else frame_id
    candidates_names = [f"{num_str}.jpg", f"{frame_id}.jpg"]
    if num_str.isdigit():
        candidates_names.extend([f"{int(num_str):03d}.jpg", f"{int(num_str):04d}.jpg", f"{int(num_str):05d}.jpg"])

    for folder_name in ("keyframes", "Keyframes_L21", "Keyframes_L23"):
        for sub in (video_id, f"keyframes/{video_id}", ""):
            base_dir = DATA_ROOT / folder_name / sub if sub else DATA_ROOT / folder_name
            if base_dir.is_dir():
                for name in candidates_names:
                    cand = base_dir / name
                    if cand.is_file():
                        return cand
    return None


def get_online_watch_url(video_id: str | None, timestamp_ms: int | None = None) -> str | None:
    if not video_id:
        return None
    json_path = DATA_ROOT / "media-info-aic25-b1" / "media-info" / f"{video_id}.json"
    if json_path.is_file():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                watch_url = data.get("watch_url")
                if watch_url:
                    if timestamp_ms is not None and timestamp_ms > 0:
                        sec = int(timestamp_ms / 1000)
                        separator = "&" if "?" in watch_url else "?"
                        return f"{watch_url}{separator}t={sec}s"
                    return watch_url
        except Exception:
            pass
    return None


def resolve_media_paths(hit: TextSearchHit) -> dict[str, Any]:
    """Find local video, local keyframe image, and online watch url."""
    video_path = find_local_video(hit.video_id)
    keyframe_path = find_local_keyframe(hit.video_id, hit.frame_id)
    watch_url = get_online_watch_url(hit.video_id, hit.timestamp_ms or hit.start_ms)

    return {
        "video": video_path,
        "keyframe": keyframe_path,
        "watch_url": watch_url,
    }


def display_hits(hits: list[TextSearchHit], elapsed_ms: float = 0.0) -> None:
    """Print search hits in a formatted, human-readable terminal table."""
    if not hits:
        print(f"\n  [x] Khong tim thay ket qua phu hop nao ({elapsed_ms:.2f}ms).\n")
        return

    print(f"\n[+] Tim thay {len(hits)} ket qua phu hop (Thoi gian truy van: {elapsed_ms:.2f}ms):\n" + "=" * 80)
    for idx, hit in enumerate(hits, start=1):
        source_tag = f"[{hit.source_type.upper()}]"
        media = resolve_media_paths(hit)

        print(f"Top {idx:02d} | {source_tag:<16} | Video: {hit.video_id:<12} | Score: {hit.score:.3f}")
        print(f"       Entity ID : {hit.entity_id}")

        if hit.timestamp_ms is not None:
            print(f"       [TIME] Thoi diem: {format_timestamp(hit.timestamp_ms)}")
        elif hit.start_ms is not None and hit.end_ms is not None:
            print(f"       [TIME] Khoang tg: {format_timestamp(hit.start_ms)} -> {format_timestamp(hit.end_ms)}")

        if hit.highlights:
            clean_hl = " | ".join(
                h.replace("<em>", ">>").replace("</em>", "<<") for h in hit.highlights
            )
            print(f"       Khop chu  : {clean_hl}")
        else:
            snippet = hit.content[:100] + ("..." if len(hit.content) > 100 else "")
            print(f"       Noi dung  : {snippet}")

        # Media links
        if media["keyframe"]:
            print(f"       [IMG]  Anh Frame: {media['keyframe']}")
        if media["video"]:
            print(f"       [VID]  File Video: {media['video']}")
        if media["watch_url"]:
            print(f"       [URL]  Link Xem  : {media['watch_url']}")

        print("-" * 80)
    print()


def execute_search(
    service: TextSearchService,
    query_text: str,
    *,
    source_types: tuple[TextSourceType, ...] = (),
    video_id: str | None = None,
    top_k: int = 10,
) -> list[TextSearchHit]:
    """Perform text search and display the formatted output."""
    video_ids = (video_id,) if video_id else ()
    query = TextSearchQuery(
        query_text=query_text,
        source_types=source_types,
        video_ids=video_ids,
        top_k=top_k,
    )
    t0 = time.perf_counter()
    hits = service.search(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    display_hits(hits, elapsed_ms=elapsed_ms)
    return hits


def open_hit_media(hit: TextSearchHit) -> None:
    """Open keyframe image, local video, or online YouTube/watch URL."""
    media = resolve_media_paths(hit)
    opened = False

    # 1. Try local keyframe image
    if media["keyframe"] and media["keyframe"].exists():
        print(f"[*] Dang mo anh keyframe: {media['keyframe']}")
        if hasattr(os, "startfile"):
            os.startfile(str(media["keyframe"]))
        else:
            subprocess.Popen(["xdg-open", str(media["keyframe"])])
        opened = True

    # 2. Try local video
    if media["video"] and media["video"].exists():
        print(f"[*] Dang mo video offline: {media['video']}")
        if hasattr(os, "startfile"):
            os.startfile(str(media["video"]))
        else:
            subprocess.Popen(["xdg-open", str(media["video"])])
        opened = True

    # 3. Try online watch URL (YouTube/HTV)
    if not opened and media["watch_url"]:
        print(f"[*] Dang mo video online tren trinh duyet: {media['watch_url']}")
        webbrowser.open(media["watch_url"])
        opened = True

    if not opened:
        target_time = format_timestamp(hit.timestamp_ms or hit.start_ms)
        print(f"[!] Video ID: '{hit.video_id}', Thoi diem: {target_time}")
        print("    (Khong tim thay file media offline va khong co watch_url)")


def interactive_mode(service: TextSearchService) -> None:
    """Run an interactive prompt loop in terminal."""
    print("=" * 80)
    print("=== AIC HCMC 2026 - TERMINAL TEXT SEARCH & VIDEO LOCATOR ===")
    print("   - Nhap tu khoa de tim kiem (vd: 'vo dich', 'Integration', 'Hakuhodo').")
    print("   - Nhap 'open <so_thu_tu>' (vd: 'open 1') de mo ngay anh frame / link video.")
    print("   - Nhap 'exit' hoac 'quit' de thoat.")
    print("=" * 80)

    last_hits: list[TextSearchHit] = []

    while True:
        try:
            query = input("\n>> Nhap tu khoa tim kiem: ").strip()
            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                print("Tam biet!")
                break
            if query.lower().startswith("open "):
                parts = query.split()
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(last_hits):
                        open_hit_media(last_hits[idx])
                    else:
                        print(f"So thu tu khong hop le (Chon tu 1 den {len(last_hits)})")
                continue

            last_hits = execute_search(service, query)
        except (KeyboardInterrupt, EOFError):
            print("\nDa thoat.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search text (OCR, Metadata, Transcript, Caption) and locate video frames directly in terminal."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Search query text. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by comma-separated source types: video_metadata, ocr, transcript, caption",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Filter results by specific video_id (e.g. L21_V001).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10).",
    )
    args = parser.parse_args()

    service = TextSearchService()

    source_types: tuple[TextSourceType, ...] = ()
    if args.source:
        valid_sources = {"video_metadata", "ocr", "transcript", "caption"}
        requested = [s.strip() for s in args.source.split(",") if s.strip()]
        for s in requested:
            if s not in valid_sources:
                print(f"Lỗi: Nguồn '{s}' không hợp lệ. Chọn trong {valid_sources}")
                sys.exit(1)
        source_types = tuple(requested)  # type: ignore[assignment]

    if args.query:
        execute_search(
            service,
            args.query,
            source_types=source_types,
            video_id=args.video_id,
            top_k=args.top_k,
        )
    else:
        interactive_mode(service)


if __name__ == "__main__":
    main()
