import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_core.utils import validate_video_take


def _build_video(path: Path, width: int = 320, height: int = 256, duration_sec: float = 2.0, frame_rate: int = 24) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={frame_rate}",
            "-t",
            f"{duration_sec:.3f}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


class TakeQualityGuardTest(unittest.TestCase):
    def test_quality_guard_accepts_valid_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "valid.mp4"
            _build_video(video_path)

            report = validate_video_take(
                video_path,
                expected_width=320,
                expected_height=256,
                expected_frame_rate=24,
                expected_duration_sec=2.0,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report["validation_status"], "passed")
            self.assertEqual(report["width"], 320)
            self.assertEqual(report["height"], 256)
            self.assertAlmostEqual(report["fps"], 24.0, places=1)

    def test_quality_guard_rejects_trivially_small_or_corrupt_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "broken.mp4"
            video_path.write_bytes(b"bad")

            report = validate_video_take(
                video_path,
                expected_width=320,
                expected_height=256,
                expected_frame_rate=24,
                expected_duration_sec=2.0,
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["validation_status"], "rejected")
            self.assertTrue(report["issues"])


if __name__ == "__main__":
    unittest.main()
