from __future__ import annotations

import unittest

from BackEnd.app.embedding.scripts.benchmark_decoder import benchmark_decoder


class DecoderBenchmarkScriptTests(unittest.TestCase):
    def test_missing_video_returns_failure_metrics(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = benchmark_decoder(
                Path(temp_dir) / "missing.mp4",
                timestamps_ms=(100, 200),
            )

        self.assertEqual(metrics["open_count"], 0)
        self.assertEqual(metrics["failed_frame_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

