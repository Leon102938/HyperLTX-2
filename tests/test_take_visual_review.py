import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.adapters.base import VideoAdapter
from agent_core.assembler import ResultAssembler
from agent_core.backend_registry import BackendRegistry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import (
    ArtifactRef,
    BackendCapabilities,
    ExecutionResult,
    JobInput,
    ProductionPlan,
    ScenePlan,
    TakeResultRecord,
    TakeValidationReport,
)
from agent_core.state_store import StateStore
from agent_core.utils import evaluate_take_visual_review, extract_review_frames


class TakeVisualReviewFakeVideoAdapter(VideoAdapter):
    name = "take_visual_review_fake_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            supported_pipelines=["ti2vid"],
        )

    def generate_video(
        self,
        job: JobInput,
        plan: ProductionPlan,
        workspace: Path,
        voice_result: ExecutionResult | None = None,
    ) -> ExecutionResult:
        workspace.mkdir(parents=True, exist_ok=True)
        video_path = workspace / "fake_video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={plan.width}x{plan.height}:r=24",
                "-t",
                f"{plan.target_duration_sec:.3f}",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return ExecutionResult(
            step_name="video",
            success=True,
            status="succeeded",
            backend_name=self.name,
            backend_job_id=f"{job.job_id}_video",
            output_path=str(video_path),
            artifacts=[ArtifactRef(key="final_video", kind="video", path=str(video_path), origin=self.name, exists=True)],
        )


class TakeVisualReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = {
            "VISION_REVIEW_ENABLED": os.environ.get("VISION_REVIEW_ENABLED"),
            "VISION_REVIEW_PROVIDER": os.environ.get("VISION_REVIEW_PROVIDER"),
        }
        os.environ["VISION_REVIEW_ENABLED"] = "1"
        os.environ["VISION_REVIEW_PROVIDER"] = "heuristic"

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @unittest.skipIf(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, "ffmpeg/ffprobe not available")
    def test_extract_review_frames_from_short_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=160x96:r=24",
                    "-t",
                    "1.2",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = extract_review_frames(video_path, Path(tmpdir) / "review_frames", count=5)

            self.assertGreaterEqual(payload["frame_count"], 1)
            self.assertLessEqual(payload["frame_count"], 3)
            self.assertTrue(all(frame["exists"] for frame in payload["frames"] if frame["path"]))
            self.assertTrue(all(Path(frame["path"]).exists() for frame in payload["frames"] if frame["exists"]))

    def test_heuristic_review_allows_forbidden_policy_words(self) -> None:
        review = evaluate_take_visual_review(
            validation={"passed": True, "issues": [], "warnings": []},
            scene_world_contract={
                "visible_subject": "person opening curtains in soft morning light",
                "environment": "tidy bedroom with clean unlabeled surfaces",
                "action": "opening curtains",
                "allowed_props": ["glass of water", "curtains", "window light"],
                "forbidden_props": ["paper", "screens", "readable text", "logos"],
                "text_risk_policy": "No readable text, no paper, no screens, no logos.",
            },
            review_frames=[{"timestamp_sec": 0.5, "path": "/tmp/frame.jpg", "exists": True}],
            prompt_text="Forbidden visuals: paper, screens, readable text. Text risk policy: no paper, no screens.",
            prompt_variant_text="No readable text, no screens, no paper.",
        )

        self.assertEqual(review["take_visual_review_status"], "passed")
        self.assertEqual(review["issues"], [])
        self.assertGreaterEqual(review["postability_score"], 0.8)

    def test_heuristic_review_rejects_positive_allowed_or_action_risk(self) -> None:
        review = evaluate_take_visual_review(
            validation={"passed": True, "issues": [], "warnings": []},
            scene_world_contract={
                "visible_subject": "person at desk",
                "environment": "workspace",
                "action": "writing on paper",
                "allowed_props": ["open notebook", "visible screen"],
                "forbidden_props": ["paper", "screens"],
                "text_risk_policy": "No readable text.",
            },
            review_frames=[{"timestamp_sec": 0.5, "path": "/tmp/frame.jpg", "exists": True}],
            prompt_text="Scene prompt.",
        )

        self.assertEqual(review["take_visual_review_status"], "rejected")
        self.assertTrue(any("allowed_props" in issue for issue in review["issues"]))
        self.assertTrue(any("action" in issue for issue in review["issues"]))

    def test_heuristic_review_missing_frames_needs_review(self) -> None:
        review = evaluate_take_visual_review(
            validation={"passed": True, "issues": [], "warnings": []},
            scene_world_contract={
                "visible_subject": "person by window",
                "environment": "tidy room",
                "action": "breathing calmly",
                "allowed_props": ["curtains"],
                "forbidden_props": ["paper"],
                "text_risk_policy": "No readable text.",
            },
            review_frames=[],
            frame_warnings=["ffmpeg did not create review frame"],
        )

        self.assertEqual(review["take_visual_review_status"], "needs_review")
        self.assertTrue(review["warnings"])

    def test_selection_prefers_passed_then_needs_review_then_rejected(self) -> None:
        agent = VideoAgent()
        scene = ScenePlan(
            scene_id="scene_01",
            index=1,
            title="Scene",
            description="Scene",
            target_duration_sec=4.0,
            num_frames=105,
            prompt_text="Scene prompt",
        )
        records = [
            self._take_record("take_rejected", 1, "rejected", 0.2),
            self._take_record("take_needs", 2, "needs_review", 0.7),
            self._take_record("take_passed", 3, "passed", 0.85),
        ]

        selected, details = agent._select_take_record(
            scene,
            records,
            total_scene_count=1,
            previous_selected_shot_type=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.take_id, "take_passed")
        self.assertEqual(details["visual_selection_status"], "passed")

        selected_without_passed, details_without_passed = agent._select_take_record(
            scene,
            records[:2],
            total_scene_count=1,
            previous_selected_shot_type=None,
        )
        self.assertIsNotNone(selected_without_passed)
        self.assertEqual(selected_without_passed.take_id, "take_needs")
        self.assertEqual(details_without_passed["visual_selection_status"], "needs_review")

    @unittest.skipIf(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, "ffmpeg/ffprobe not available")
    def test_run_persists_take_visual_review_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([TakeVisualReviewFakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )
            result = agent.run_job(
                {
                    "job_id": "take-visual-review-metadata",
                    "idea": "A calm morning reset without text props.",
                    "script": "Scene one opens with curtains and a glass of water.",
                    "duration_sec": 4,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {
                        "force_single_scene": True,
                        "takes_per_scene": 2,
                        "variations_per_scene": 1,
                    },
                }
            )

            takes_payload = json.loads((store.job_dir("take-visual-review-metadata") / "takes.json").read_text())
            selected = takes_payload["selected_scene_outputs"][0]

            self.assertTrue(result.success)
            self.assertIn("take_visual_review", selected)
            self.assertIn("postability_score", selected)
            self.assertIn(selected["take_visual_review_status"], {"passed", "needs_review", "rejected"})
            self.assertTrue(selected["review_frames"])
            self.assertIn("take_visual_review", takes_payload["scene_outputs"][0]["selected_take"]["metadata"])

    @staticmethod
    def _take_record(take_id: str, take_index: int, visual_status: str, postability_score: float) -> TakeResultRecord:
        validation = TakeValidationReport(
            validation_status="passed",
            passed=True,
            file_exists=True,
            file_size_bytes=2048,
            ffprobe_ok=True,
            decode_ok=True,
            width=320,
            height=256,
            fps=24.0,
            duration_sec=4.0,
            duration_delta_sec=0.0,
        )
        return TakeResultRecord(
            take_id=take_id,
            scene_id="scene_01",
            take_index=take_index,
            shot_type="establishing",
            prompt_variant_text="Clean visual prompt",
            seed=take_index,
            status="succeeded",
            review_status="passed",
            output_path=f"/tmp/{take_id}.mp4",
            duration_sec=4.0,
            validation=validation,
            metadata={
                "take_visual_review_status": visual_status,
                "postability_score": postability_score,
                "take_visual_review": {
                    "take_visual_review_status": visual_status,
                    "postability_score": postability_score,
                    "provider": "heuristic",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
