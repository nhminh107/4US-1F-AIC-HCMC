REQUIRED_SHOT_COLUMNS = {
    "shot_id", "video_id", "shot_index", "start_ms", "end_ms",
    "start_frame_idx", "end_frame_idx",
}

@dataclass(frozen=True)
class Shot:
    global_index: int
    shot_id: str
    video_id: str
    shot_index: int
    start_ms: int
    end_ms: int
    start_frame_idx: int | None
    end_frame_idx: int | None


def _optional_int(value: Any) -> int | None:
    text = "" if value is None else str(value).strip()
    return None if not text else int(text)


def load_video_urls(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy videos file: {path}")
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 1:
            parts = line.split(",", 1)
        if len(parts) == 1:
            parts = line.split(maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(f"{path}:{line_number}: cần '<video_id><TAB><URL>'")
        video_id, url = (part.strip() for part in parts)
        if video_id in result:
            raise ValueError(f"video_id bị lặp trong {path}: {video_id}")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"URL không hợp lệ tại {path}:{line_number}: {url}")
        result[video_id] = url
    if not result:
        raise ValueError(f"{path} không có video hợp lệ")
    return result


def load_shot_slice(path: Path, start: int, end: int | None) -> list[Shot]:
    if start < 0 or (end is not None and end < start):
        raise ValueError(f"Khoảng không hợp lệ: [{start}:{end})")
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy shots file: {path}")
    shots: list[Shot] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_SHOT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} thiếu cột: {sorted(missing)}")
        for global_index, row in enumerate(reader):
            if global_index < start:
                continue
            if end is not None and global_index >= end:
                break
            shot = Shot(
                global_index=global_index,
                shot_id=row["shot_id"].strip(),
                video_id=row["video_id"].strip(),
                shot_index=int(row["shot_index"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                start_frame_idx=_optional_int(row["start_frame_idx"]),
                end_frame_idx=_optional_int(row["end_frame_idx"]),
            )
            if not shot.shot_id or not shot.video_id:
                raise ValueError(f"Dòng shot {global_index + 2}: ID rỗng")
            if shot.shot_id in seen_ids:
                raise ValueError(f"shot_id bị lặp trong lát cắt: {shot.shot_id}")
            if shot.shot_index < 0 or shot.start_ms < 0 or shot.end_ms <= shot.start_ms:
                raise ValueError(f"Shot không hợp lệ: {shot}")
            if (shot.start_frame_idx is None) != (shot.end_frame_idx is None):
                raise ValueError(f"Shot phải có cả hai frame index hoặc cùng để trống: {shot.shot_id}")
            if shot.start_frame_idx is not None and shot.end_frame_idx < shot.start_frame_idx:
                raise ValueError(f"Frame range không hợp lệ: {shot.shot_id}")
            seen_ids.add(shot.shot_id)
            shots.append(shot)
    if not shots:
        raise ValueError(f"Không có shot trong lát cắt [{start}:{end})")
    return shots


def safe_video_path(video_dir: Path, video_id: str, url: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id).strip("._")
    if not safe_id:
        raise ValueError(f"video_id không thể tạo filename an toàn: {video_id!r}")
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):
        suffix = ".mp4"
    return video_dir / f"{safe_id}{suffix}"


def download_video(
    video_id: str, url: str, target: Path, overwrite: bool = False, max_retries: int = 5
) -> Path:
    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=(20, 300)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size == 0:
                raise RuntimeError(f"Video tải về rỗng: {video_id}")
            partial.replace(target)
            return target
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt + 1 == max_retries:
                break
            # Cap cao hơn để vượt qua đợt DNS/network chưa sẵn sàng lúc kernel Kaggle mới khởi động (~30-45s).
            time.sleep(min(45.0, 3 * (2**attempt)) + random.uniform(0.0, 1.0))
    raise RuntimeError(f"Tải video thất bại sau {max_retries} lần: {video_id}: {last_error}") from last_error


def download_required_videos(
    shots: list[Shot], video_urls: dict[str, str], config: PipelineConfig
) -> dict[str, Path]:
    video_ids = list(dict.fromkeys(shot.video_id for shot in shots))
    missing = [video_id for video_id in video_ids if video_id not in video_urls]
    if missing:
        raise KeyError(f"Thiếu URL cho video_id: {missing[:10]}")
    video_dir = config.work_dir / "videos"
    paths = {
        video_id: safe_video_path(video_dir, video_id, video_urls[video_id])
        for video_id in video_ids
    }
    with ThreadPoolExecutor(max_workers=config.download_workers) as executor:
        futures = {
            executor.submit(
                download_video, video_id, video_urls[video_id], paths[video_id],
                config.overwrite_videos, config.max_retries,
            ): video_id
            for video_id in video_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Download videos"):
            future.result()
    return paths