-- Khởi động PostgreSQL, chạy script này, đọc readme để hướng dẫn kết nối
BEGIN;

-- =====================================================
-- VIDEO
-- =====================================================
CREATE TABLE video (
    video_id        VARCHAR(32) PRIMARY KEY,

    fps             DOUBLE PRECISION NOT NULL
                        CHECK (fps > 0),

    duration_ms     BIGINT NOT NULL
                        CHECK (duration_ms >= 0),

    video_path      TEXT NOT NULL
);


-- =====================================================
-- SCENE
-- =====================================================
CREATE TABLE scene (
    scene_id        VARCHAR(48) PRIMARY KEY,

    video_id        VARCHAR(32) NOT NULL,

    start_ms        BIGINT NOT NULL
                        CHECK (start_ms >= 0),

    end_ms          BIGINT NOT NULL
                        CHECK (end_ms > start_ms),

    CONSTRAINT fk_scene_video
        FOREIGN KEY (video_id)
        REFERENCES video(video_id)
        ON DELETE CASCADE
);


-- =====================================================
-- KEYFRAME
-- Chỉ giữ metadata cốt lõi của keyframe
-- =====================================================
CREATE TABLE keyframe (
    keyframe_id     VARCHAR(64) PRIMARY KEY,

    scene_id        VARCHAR(48) NOT NULL,

    -- ID int64 do code của bạn tự cấp cho FAISS
    visual_index_id BIGINT NOT NULL UNIQUE,

    timestamp_ms    BIGINT NOT NULL
                        CHECK (timestamp_ms >= 0),

    image_path      TEXT NOT NULL,

    CONSTRAINT fk_keyframe_scene
        FOREIGN KEY (scene_id)
        REFERENCES scene(scene_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_keyframe_scene_time
        UNIQUE (scene_id, timestamp_ms)
);


-- =====================================================
-- OCR
-- Mỗi vùng chữ là một record riêng
-- Bounding box là hình chữ nhật không xoay
--
-- bbox_x, bbox_y: góc trên bên trái
-- bbox_width, bbox_height: chiều rộng và chiều cao
--
-- Tất cả tọa độ được chuẩn hóa trong khoảng 0 đến 1
-- =====================================================
CREATE TABLE ocr (
    ocr_id          VARCHAR(80) PRIMARY KEY,

    keyframe_id     VARCHAR(64) NOT NULL,

    text            TEXT NOT NULL,

    confidence      REAL
                        CHECK (
                            confidence IS NULL
                            OR confidence BETWEEN 0 AND 1
                        ),

    bbox_x          DOUBLE PRECISION NOT NULL,
    bbox_y          DOUBLE PRECISION NOT NULL,
    bbox_width      DOUBLE PRECISION NOT NULL,
    bbox_height     DOUBLE PRECISION NOT NULL,

    CONSTRAINT fk_ocr_keyframe
        FOREIGN KEY (keyframe_id)
        REFERENCES keyframe(keyframe_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_ocr_bbox
        CHECK (
            bbox_x >= 0
            AND bbox_y >= 0
            AND bbox_width > 0
            AND bbox_height > 0
            AND bbox_x + bbox_width <= 1
            AND bbox_y + bbox_height <= 1
        )
);


-- =====================================================
-- CAPTION
-- Một keyframe có thể có một hoặc nhiều caption
-- =====================================================
CREATE TABLE caption (
    caption_id      VARCHAR(80) PRIMARY KEY,

    keyframe_id     VARCHAR(64) NOT NULL,

    text            TEXT NOT NULL,

    CONSTRAINT fk_caption_keyframe
        FOREIGN KEY (keyframe_id)
        REFERENCES keyframe(keyframe_id)
        ON DELETE CASCADE
);


-- =====================================================
-- ASR / TRANSCRIPT
-- Transcript thuộc một khoảng thời gian trong video
-- =====================================================
CREATE TABLE transcript_segment (
    segment_id      VARCHAR(80) PRIMARY KEY,

    video_id        VARCHAR(32) NOT NULL,

    start_ms        BIGINT NOT NULL
                        CHECK (start_ms >= 0),

    end_ms          BIGINT NOT NULL
                        CHECK (end_ms > start_ms),

    text            TEXT NOT NULL,

    CONSTRAINT fk_transcript_video
        FOREIGN KEY (video_id)
        REFERENCES video(video_id)
        ON DELETE CASCADE
);

CREATE TABLE embedding_record (
    faiss_id        BIGINT,
    keyframe_id     VARCHAR(64),
    index_version   VARCHAR(50),
    model_name      VARCHAR(100),
    PRIMARY KEY (faiss_id, index_version),
    foreign key (keyframe_id) references keyframe(keyframe_id)
);

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX idx_scene_video
    ON scene(video_id);

CREATE INDEX idx_scene_video_time
    ON scene(video_id, start_ms, end_ms);

CREATE INDEX idx_keyframe_scene
    ON keyframe(scene_id);

CREATE INDEX idx_keyframe_scene_time
    ON keyframe(scene_id, timestamp_ms);

CREATE INDEX idx_ocr_keyframe
    ON ocr(keyframe_id);

CREATE INDEX idx_transcript_video_time
    ON transcript_segment(video_id, start_ms, end_ms);

COMMIT;
