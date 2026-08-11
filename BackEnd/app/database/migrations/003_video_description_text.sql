-- Preserve full organizer video descriptions instead of truncating at 500 chars.
ALTER TABLE video
    ALTER COLUMN description TYPE TEXT;
