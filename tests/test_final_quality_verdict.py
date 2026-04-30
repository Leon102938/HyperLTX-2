import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_core.assembler import ResultAssembler
from agent_core.schemas import ExecutionResult, JobInput
from agent_core.utils import evaluate_final_quality_verdict, probe_media_duration
from tests.test_assembler_mux import _build_plan, _build_state, _build_video


class FinalQualityVerdictTest(unittest.TestCase):
    def test_final_verdict_metadata_present_for_clean_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            _build_video(video_path, 2.0)

            result = ResultAssembler().assemble(
                JobInput(job_id="final-quality-clean", idea="clean"),
                _build_plan("final-quality-clean", 2.0),
                _build_state("final-quality-clean"),
                workspace,
                None,
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    output_path=str(video_path),
                    duration_sec=probe_media_duration(str(video_path)),
                    metadata={
                        "selected_scene_outputs": [
                            {
                                "scene_id": "scene_01",
                                "output_path": str(video_path),
                                "selected_take_id": "scene_01_take_01",
                                "review_status": "selected",
                                "validation": {"passed": True},
                                "take_visual_review_status": "passed",
                                "postability_score": 0.92,
                                "take_visual_review": {
                                    "take_visual_review_status": "passed",
                                    "postability_score": 0.92,
                                    "issues": [],
                                    "warnings": [],
                                    "provider": "heuristic",
                                },
                            }
                        ],
                    },
                ),
            )

            verdict = result.metadata["final_quality_verdict"]
            self.assertIn(verdict["final_quality_status"], {"passed", "needs_review"})
            self.assertGreaterEqual(verdict["final_postability_score"], 0.7)
            self.assertIn("quality_policy_version", verdict)
            self.assertIn("final_quality_verdict", result.metadata["assembly"])

    def test_final_verdict_needs_review_when_selected_take_needs_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            _build_video(video_path, 2.0)

            result = ResultAssembler().assemble(
                JobInput(job_id="final-quality-needs", idea="needs"),
                _build_plan("final-quality-needs", 2.0),
                _build_state("final-quality-needs"),
                workspace,
                None,
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    output_path=str(video_path),
                    duration_sec=probe_media_duration(str(video_path)),
                    metadata={
                        "selected_scene_outputs": [
                            {
                                "scene_id": "scene_01",
                                "output_path": str(video_path),
                                "selected_take_id": "scene_01_take_01",
                                "review_status": "selected",
                                "validation": {"passed": True},
                                "take_visual_review_status": "needs_review",
                                "postability_score": 0.62,
                                "take_visual_review": {
                                    "take_visual_review_status": "needs_review",
                                    "postability_score": 0.62,
                                    "issues": [],
                                    "warnings": ["missing review frames"],
                                    "provider": "heuristic",
                                },
                            }
                        ],
                    },
                ),
            )

            verdict = result.metadata["final_quality_verdict"]
            self.assertEqual(verdict["final_quality_status"], "needs_review")
            self.assertTrue(any("needs visual review" in warning for warning in verdict["warnings"]))
            self.assertEqual(verdict["recommended_next_action"], "manual_visual_review_before_publish")

    def test_final_verdict_failed_when_selected_take_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            _build_video(video_path, 2.0)

            verdict = evaluate_final_quality_verdict(
                final_output_path=video_path,
                expected_width=320,
                expected_height=240,
                expected_frame_rate=24,
                expected_duration_sec=2.0,
                selected_scene_outputs=[
                    {
                        "scene_id": "scene_01",
                        "take_visual_review_status": "rejected",
                        "postability_score": 0.15,
                        "take_visual_review": {
                            "take_visual_review_status": "rejected",
                            "postability_score": 0.15,
                            "issues": ["visible screen"],
                            "warnings": [],
                        },
                    }
                ],
                output_dir=workspace,
            )

            self.assertEqual(verdict["final_quality_status"], "failed")
            self.assertTrue(verdict["main_issues"])
            self.assertEqual(verdict["recommended_next_action"], "fix_or_rerender_problem_scenes")

    def test_final_verdict_failed_when_final_output_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            verdict = evaluate_final_quality_verdict(
                final_output_path=workspace / "missing.mp4",
                expected_width=320,
                expected_height=240,
                expected_frame_rate=24,
                expected_duration_sec=2.0,
                selected_scene_outputs=[],
                output_dir=workspace,
            )

            self.assertEqual(verdict["final_quality_status"], "failed")
            self.assertIn("final.mp4 is missing", verdict["main_issues"])

    def test_qwen_status_normalization_is_not_a_gpu_unit_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=320x240:r=24",
                    "-t",
                    "1.0",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            verdict = evaluate_final_quality_verdict(
                final_output_path=video_path,
                expected_width=320,
                expected_height=240,
                expected_frame_rate=24,
                expected_duration_sec=1.0,
                selected_scene_outputs=[],
                output_dir=workspace,
                final_frame_provider="heuristic",
            )

            self.assertIn("heuristic_final_frame_review", verdict["quality_sources"])
            self.assertIn(verdict["final_quality_status"], {"needs_review", "passed"})


if __name__ == "__main__":
    unittest.main()
