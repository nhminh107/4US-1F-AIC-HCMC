@dataclass(frozen=True)
class FrameCandidate:
    timestamp_ms: int
    frame: Any
    sharpness: float
    brightness: float


def adaptive_candidate_count(shot: Shot, config: PipelineConfig) -> int:
    duration_ms = shot.end_ms - shot.start_ms
    if duration_ms <= config.short_shot_ms:
        return min(3, config.max_candidate_frames)
    # Xấp xỉ một ứng viên mỗi 1,5 giây, có chặn để thời gian decode ổn định.
    return min(config.max_candidate_frames, max(6, (duration_ms + 1499) // 1500))


def sample_frame_times_ms(shot: Shot, count: int) -> list[int]:
    if count < 1:
        raise ValueError("candidate frame count phải >= 1")
    duration = shot.end_ms - shot.start_ms
    # Tâm của các đoạn đều nhau, không chạm end_ms (thường là biên exclusive).
    return [shot.start_ms + int(duration * (index + 0.5) / count) for index in range(count)]


def frame_quality(frame: Any) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharpness, float(gray.mean())


def extract_frame_candidates(
    video_path: Path, shot: Shot, config: PipelineConfig
) -> list[FrameCandidate]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được video {video_path} cho shot {shot.shot_id}")
    candidates: list[FrameCandidate] = []
    try:
        count = adaptive_candidate_count(shot, config)
        for timestamp_ms in sample_frame_times_ms(shot, count):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            sharpness, brightness = frame_quality(frame)
            candidates.append(FrameCandidate(timestamp_ms, frame, sharpness, brightness))
    finally:
        capture.release()
    if not candidates:
        raise RuntimeError(f"Không trích được frame nào cho shot {shot.shot_id}")
    return candidates


def visual_distance(first: Any, second: Any) -> float:
    size = (160, 90)
    first_small = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
    second_small = cv2.resize(second, size, interpolation=cv2.INTER_AREA)
    first_hsv = cv2.cvtColor(first_small, cv2.COLOR_BGR2HSV)
    second_hsv = cv2.cvtColor(second_small, cv2.COLOR_BGR2HSV)
    hist_first = cv2.calcHist([first_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist_second = cv2.calcHist([second_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_first, hist_first)
    cv2.normalize(hist_second, hist_second)
    histogram_distance = float(cv2.compareHist(hist_first, hist_second, cv2.HISTCMP_BHATTACHARYYA))
    pixel_distance = min(1.0, float(cv2.absdiff(first_small, second_small).mean()) / 32.0)
    return 0.55 * histogram_distance + 0.45 * pixel_distance


def filter_low_quality_and_duplicates(
    candidates: list[FrameCandidate], config: PipelineConfig
) -> list[FrameCandidate]:
    usable = [
        candidate for candidate in candidates
        if config.min_blur_score <= candidate.sharpness and 8.0 <= candidate.brightness <= 247.0
    ]
    if not usable:
        usable = candidates
    deduplicated: list[FrameCandidate] = []
    for candidate in usable:
        if not deduplicated:
            deduplicated.append(candidate)
            continue
        if visual_distance(deduplicated[-1].frame, candidate.frame) < config.duplicate_distance:
            if candidate.sharpness > deduplicated[-1].sharpness:
                deduplicated[-1] = candidate
        else:
            deduplicated.append(candidate)
    return deduplicated


def select_diverse_keyframes(
    candidates: list[FrameCandidate], shot: Shot, config: PipelineConfig
) -> list[FrameCandidate]:
    candidates = filter_low_quality_and_duplicates(candidates, config)
    duration_ms = shot.end_ms - shot.start_ms
    if len(candidates) == 1:
        return candidates
    if duration_ms <= config.short_shot_ms:
        midpoint = (shot.start_ms + shot.end_ms) / 2
        return [min(candidates, key=lambda item: abs(item.timestamp_ms - midpoint))]
    if duration_ms <= config.medium_shot_ms:
        best_pair = max(
            ((first, second) for index, first in enumerate(candidates) for second in candidates[index + 1:]),
            key=lambda pair: visual_distance(pair[0].frame, pair[1].frame)
            + 0.2 * abs(pair[1].timestamp_ms - pair[0].timestamp_ms) / duration_ms,
        )
        return list(best_pair)

    target = min(config.max_keyframes, len(candidates))
    selected = [candidates[0], candidates[-1]]
    while len(selected) < target:
        selected_timestamps = {item.timestamp_ms for item in selected}
        remaining = [item for item in candidates if item.timestamp_ms not in selected_timestamps]
        if not remaining:
            break
        next_item = max(
            remaining,
            key=lambda item: min(visual_distance(item.frame, chosen.frame) for chosen in selected)
            + 0.2 * min(abs(item.timestamp_ms - chosen.timestamp_ms) for chosen in selected) / duration_ms,
        )
        selected.append(next_item)
    return sorted(selected, key=lambda item: item.timestamp_ms)


def letterbox_frame(frame: Any, width: int, height: int) -> Any:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    left = (width - resized_width) // 2
    right = width - resized_width - left
    top = (height - resized_height) // 2
    bottom = height - resized_height - top
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def build_storyboard(keyframes: list[FrameCandidate], config: PipelineConfig) -> Any:
    columns = min(config.storyboard_columns, len(keyframes))
    tiles = [
        letterbox_frame(item.frame, config.storyboard_tile_width, config.storyboard_tile_height)
        for item in keyframes
    ]
    blank = tiles[0].copy()
    blank[:] = 0
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [cv2.hconcat(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    return cv2.vconcat(rows)


def prepare_vlm_images(
    video_path: Path, shot: Shot, config: PipelineConfig
) -> tuple[list[Any], list[FrameCandidate]]:
    keyframes = select_diverse_keyframes(extract_frame_candidates(video_path, shot, config), shot, config)
    if len(keyframes) <= 2:
        return [item.frame for item in keyframes], keyframes
    storyboard = build_storyboard(keyframes, config)
    detail = max(keyframes, key=lambda item: item.sharpness).frame
    return [storyboard, detail], keyframes


def frame_to_data_url(frame: Any, jpeg_quality: int) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    if not ok:
        raise RuntimeError("Không encode được frame sang JPEG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def extract_caption_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Response FPT không đúng Chat Completions: {payload}") from exc
    if isinstance(content, str):
        caption = content.strip()
    elif isinstance(content, list):
        caption = "\n".join(
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ).strip()
    else:
        caption = ""
    if not caption:
        raise RuntimeError(f"FPT trả caption rỗng: {payload}")
    return caption


def call_fpt_vlm(
    frames: list[Any], api_key: str, config: PipelineConfig, api_semaphore: threading.Semaphore
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": CAPTION_PROMPT}]
    content.extend(
        {"type": "image_url", "image_url": {"url": frame_to_data_url(frame, config.jpeg_quality)}}
        for frame in frames
    )
    request_payload = {
        "model": config.fpt_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            with api_semaphore:
                response = requests.post(
                    config.fpt_endpoint,
                    headers=headers,
                    json=request_payload,
                    timeout=(20, config.request_timeout_s),
                )
            if response.status_code in {408, 429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"FPT HTTP {response.status_code}: {response.text[:500]}", response=response
                )
            response.raise_for_status()
            return extract_caption_content(response.json())
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 == config.max_retries:
                break
            retry_after = None
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2**attempt)
            time.sleep(delay + random.uniform(0.0, 0.5))
    raise RuntimeError(f"Gọi FPT thất bại sau {config.max_retries} lần: {last_error}") from last_error


def process_shot(
    shot: Shot,
    video_path: Path,
    api_key: str,
    config: PipelineConfig,
    api_semaphore: threading.Semaphore,
) -> dict[str, Any]:
    api_images, keyframes = prepare_vlm_images(video_path, shot, config)
    caption = call_fpt_vlm(api_images, api_key, config, api_semaphore)
    return {
        "global_index": shot.global_index,
        "caption_id": config.caption_id_offset + shot.global_index,
        "shot_id": shot.shot_id,
        "caption_text": caption,
        "model": config.fpt_model,
        "keyframe_count": len(keyframes),
        "api_image_count": len(api_images),
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                results[item["shot_id"]] = item
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Checkpoint hỏng tại {path}:{line_number}") from exc
    return results


def append_checkpoint(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sql_literal(value: str) -> str:
    # PostgreSQL standard string literal: nháy đơn được nhân đôi; NUL bị cấm.
    if "\x00" in value:
        raise ValueError("SQL text không được chứa NUL")
    return "'" + value.replace("'", "''") + "'"


def write_sql(path: Path, shots: list[Shot], results: dict[str, dict[str, Any]]) -> None:
    missing = [shot.shot_id for shot in shots if shot.shot_id not in results]
    if missing:
        raise RuntimeError(f"Chưa đủ caption nên không xuất SQL; thiếu {len(missing)} shot")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("BEGIN;\n")
        for shot in shots:
            item = results[shot.shot_id]
            handle.write(SQL_INSERT_TEMPLATE.format(
                caption_id=int(item["caption_id"]),
                shot_id=sql_literal(item["shot_id"]),
                caption_text=sql_literal(item["caption_text"]),
            ) + "\n")
        handle.write("COMMIT;\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def get_fpt_api_key(secret_name: str) -> str:
    key = os.getenv(secret_name, "").strip()
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret(secret_name).strip()
        except (ImportError, Exception) as exc:
            raise RuntimeError(
                f"Không lấy được secret {secret_name!r}. Hãy tạo Kaggle Secret hoặc env var cùng tên."
            ) from exc
    if not key:
        raise RuntimeError(f"Secret {secret_name!r} rỗng")
    return key


def validate_config(config: PipelineConfig) -> None:
    positive = {
        "download_workers": config.download_workers,
        "shot_workers": config.shot_workers,
        "api_workers": config.api_workers,
        "request_timeout_s": config.request_timeout_s,
        "max_retries": config.max_retries,
        "max_candidate_frames": config.max_candidate_frames,
        "max_keyframes": config.max_keyframes,
        "short_shot_ms": config.short_shot_ms,
        "medium_shot_ms": config.medium_shot_ms,
        "storyboard_columns": config.storyboard_columns,
        "storyboard_tile_width": config.storyboard_tile_width,
        "storyboard_tile_height": config.storyboard_tile_height,
    }
    invalid = {name: value for name, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"Cấu hình phải dương: {invalid}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config.pipeline_version):
        raise ValueError("pipeline_version chỉ được chứa chữ, số, '.', '_' hoặc '-'")
    if config.max_keyframes < 2:
        raise ValueError("max_keyframes phải >= 2")
    if config.short_shot_ms > config.medium_shot_ms:
        raise ValueError("short_shot_ms không được lớn hơn medium_shot_ms")
    if not 0.0 <= config.duplicate_distance <= 1.0:
        raise ValueError("duplicate_distance phải trong [0, 1]")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("jpeg_quality phải trong [1, 100]")
    if "REPLACE_WITH" in config.fpt_model:
        raise ValueError("Hãy điền đúng fpt_model trước khi chạy")
    parsed = urlparse(config.fpt_endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("fpt_endpoint phải là HTTPS URL hợp lệ")


def run_pipeline(config: PipelineConfig, api_key: str | None = None) -> Path:
    validate_config(config)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    shots = load_shot_slice(config.shots_file, config.slice_start, config.slice_end)
    video_urls = load_video_urls(config.videos_file)
    video_paths = download_required_videos(shots, video_urls, config)
    api_key = api_key or get_fpt_api_key(config.api_key_secret_name)

    checkpoint = config.work_dir / (
        f"checkpoint_{config.pipeline_version}_{config.slice_start}_{config.slice_end}.jsonl"
    )
    results = load_checkpoint(checkpoint) if config.resume else {}
    selected_ids = {shot.shot_id for shot in shots}
    results = {shot_id: item for shot_id, item in results.items() if shot_id in selected_ids}
    pending = [shot for shot in shots if shot.shot_id not in results]
    print(f"Selected={len(shots)}, resumed={len(results)}, pending={len(pending)}")

    failures: list[tuple[str, str]] = []
    api_semaphore = threading.Semaphore(config.api_workers)
    with ThreadPoolExecutor(max_workers=config.shot_workers) as executor:
        futures = {
            executor.submit(
                process_shot, shot, video_paths[shot.video_id], api_key, config, api_semaphore
            ): shot
            for shot in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Caption shots"):
            shot = futures[future]
            try:
                item = future.result()
                results[shot.shot_id] = item
                append_checkpoint(checkpoint, item)
            except Exception as exc:
                failures.append((shot.shot_id, repr(exc)))

    if failures:
        failure_path = config.work_dir / (
            f"failures_{config.pipeline_version}_{config.slice_start}_{config.slice_end}.json"
        )
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        preview = "\n".join(f"- {shot_id}: {error}" for shot_id, error in failures[:5])
        raise RuntimeError(
            f"{len(failures)} shot thất bại; checkpoint đã giữ kết quả thành công.\n{preview}\n"
            f"Chi tiết: {failure_path}"
        )

    write_sql(config.output_sql, shots, results)
    print(f"Hoàn tất {len(shots)} shot -> {config.output_sql}")
    return config.output_sql