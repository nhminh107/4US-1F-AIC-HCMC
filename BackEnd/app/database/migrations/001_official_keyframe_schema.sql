BEGIN;

ALTER TABLE frame
    DROP COLUMN IF EXISTS frame_role;

ALTER TABLE frame
    ALTER COLUMN pts_time TYPE double precision
    USING pts_time::double precision;

ALTER TABLE frame
    DROP CONSTRAINT IF EXISTS frame_video_id_frame_idx_key;

ALTER TABLE frame
    ALTER COLUMN shot_id DROP NOT NULL;

UPDATE frame
SET shot_id = NULL
WHERE source = 'official';

ALTER TABLE frame
    DROP CONSTRAINT IF EXISTS frame_official_shot_null_check;

ALTER TABLE frame
    ADD CONSTRAINT frame_official_shot_null_check
    CHECK (source <> 'official' OR shot_id IS NULL);

CREATE UNIQUE INDEX IF NOT EXISTS uq_frame_official_video_n
    ON frame(video_id, n)
    WHERE source = 'official';

COMMIT;
