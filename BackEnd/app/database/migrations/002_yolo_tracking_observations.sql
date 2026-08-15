-- Review and back up the target database before running this migration.
-- This script is intentionally not executed by the application or test suite.

BEGIN;

-- Preserve the legacy detection-to-track associations as data-only backup.
-- The migration fails instead of overwriting a previous backup with this name.
CREATE TABLE trackobservation_legacy_002 AS
TABLE trackobservation WITH DATA;

COMMENT ON TABLE trackobservation_legacy_002 IS
    'Backup created by migration 002 before TrackObservation was decoupled from ObjectDetection.';

-- A track should have at most one observation per source frame. Stop for manual
-- review instead of silently discarding duplicates that violate the new key.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM trackobservation_legacy_002 legacy
        JOIN objectdetection detection
            ON detection.detection_id = legacy.detection_id
        JOIN frame source_frame
            ON source_frame.frame_id = detection.frame_id
        GROUP BY legacy.track_id, source_frame.frame_idx
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION
            'Legacy TrackObservation contains duplicate (track_id, frame_idx) values.';
    END IF;
END
$$;

DROP TABLE trackobservation;

ALTER TABLE objecttrack
    ADD COLUMN model_name varchar(100) NOT NULL DEFAULT 'legacy',
    ADD COLUMN model_version varchar(100) NOT NULL DEFAULT 'unknown',
    ADD COLUMN sampling_fps float NOT NULL DEFAULT 2.0,
    ADD COLUMN mapping_version varchar(50) NOT NULL DEFAULT 'legacy-unmapped';

UPDATE objecttrack
SET tracker_name = 'legacy'
WHERE tracker_name IS NULL;

UPDATE objecttrack
SET tracker_version = 'unknown'
WHERE tracker_version IS NULL;

ALTER TABLE objecttrack
    ALTER COLUMN tracker_name SET NOT NULL,
    ALTER COLUMN tracker_version SET NOT NULL,
    ADD CONSTRAINT objecttrack_sampling_fps_check CHECK (sampling_fps > 0);

ALTER TABLE objecttrack
    ALTER COLUMN model_name DROP DEFAULT,
    ALTER COLUMN model_version DROP DEFAULT,
    ALTER COLUMN sampling_fps DROP DEFAULT,
    ALTER COLUMN mapping_version DROP DEFAULT;

CREATE TABLE trackobservation (
    track_id bigint NOT NULL,
    frame_idx bigint NOT NULL CHECK (frame_idx >= 0),
    timestamp_ms bigint NOT NULL CHECK (timestamp_ms >= 0),
    confidence float NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    x_min float NOT NULL CHECK (x_min BETWEEN 0 AND 1),
    x_max float NOT NULL CHECK (x_max BETWEEN 0 AND 1),
    y_min float NOT NULL CHECK (y_min BETWEEN 0 AND 1),
    y_max float NOT NULL CHECK (y_max BETWEEN 0 AND 1),

    PRIMARY KEY (track_id, frame_idx),
    CHECK (x_min < x_max),
    CHECK (y_min < y_max),

    FOREIGN KEY (track_id) REFERENCES objecttrack(track_id)
);

-- Preserve every legacy association that can be represented by the new
-- observation schema. The legacy backup remains available for auditing.
INSERT INTO trackobservation (
    track_id,
    frame_idx,
    timestamp_ms,
    confidence,
    x_min,
    x_max,
    y_min,
    y_max
)
SELECT
    legacy.track_id,
    source_frame.frame_idx,
    source_frame.timestamp_ms,
    detection.confidence,
    detection.x_min,
    detection.x_max,
    detection.y_min,
    detection.y_max
FROM trackobservation_legacy_002 legacy
JOIN objectdetection detection
    ON detection.detection_id = legacy.detection_id
JOIN frame source_frame
    ON source_frame.frame_id = detection.frame_id;

COMMIT;
