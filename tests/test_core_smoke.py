import subprocess
import tempfile
import unittest
import wave
import json
from pathlib import Path

from agent_core.agent import VideoAgent
from agent_core.adapters.base import VideoAdapter, VoiceAdapter
from agent_core.assembler import ResultAssembler
from agent_core.backend_registry import BackendRegistry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import ArtifactRef, BackendCapabilities, ExecutionResult, JobInput, ProductionPlan
from agent_core.state_store import StateStore


class FakeVoiceAdapter(VoiceAdapter):
    name = "fake_voice"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="voice",
            available=True,
            phase1_enabled=True,
            transport="fake",
            supported_pipelines=["custom_voice"],
        )

    def generate_voice(self, job: JobInput, plan: ProductionPlan, workspace: Path) -> ExecutionResult:
        audio_path = workspace / "fake_voice.wav"
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\x00\x00" * 24000)
        return ExecutionResult(
            step_name="voice",
            success=True,
            status="succeeded",
            backend_name=self.name,
            backend_job_id=f"{job.job_id}_voice",
            output_path=str(audio_path),
            output_url=None,
            duration_sec=7.5,
            artifacts=[
                ArtifactRef(
                    key="voice_audio",
                    kind="audio",
                    path=str(audio_path),
                    origin=self.name,
                    exists=True,
                )
            ],
        )


class FakeVideoAdapter(VideoAdapter):
    name = "fake_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            transport="fake",
            supported_pipelines=["ti2vid", "a2vid"],
        )

    def generate_video(
        self,
        job: JobInput,
        plan: ProductionPlan,
        workspace: Path,
        voice_result: ExecutionResult | None = None,
    ) -> ExecutionResult:
        fail_take_indices = {int(value) for value in job.metadata.get("fail_take_indices", [])}
        reject_take_indices = {int(value) for value in job.metadata.get("reject_take_indices", [])}
        if int(job.metadata.get("take_index", 0) or 0) in fail_take_indices:
            return ExecutionResult(
                step_name="video",
                success=False,
                status="failed",
                backend_name=self.name,
                backend_job_id=f"{job.job_id}_video",
                error=f"forced failure for take {job.metadata.get('take_index')}",
                metadata={"forced_failure": True},
            )
        workspace.mkdir(parents=True, exist_ok=True)
        video_path = workspace / "fake_video.mp4"
        if int(job.metadata.get("take_index", 0) or 0) in reject_take_indices:
            video_path.write_bytes(b"bad")
            return ExecutionResult(
                step_name="video",
                success=True,
                status="succeeded",
                backend_name=self.name,
                backend_job_id=f"{job.job_id}_video",
                output_path=str(video_path),
                output_url=None,
                duration_sec=plan.target_duration_sec,
                artifacts=[
                    ArtifactRef(
                        key="final_video",
                        kind="video",
                        path=str(video_path),
                        origin=self.name,
                        exists=True,
                    )
                ],
                metadata={"forced_quality_rejection": True},
            )
        frame_rate = int(plan.metadata.get("frame_rate", 24) or 24)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={plan.width}x{plan.height}:r={frame_rate}",
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
            output_url=None,
            duration_sec=plan.target_duration_sec,
            artifacts=[
                ArtifactRef(
                    key="final_video",
                    kind="video",
                    path=str(video_path),
                    origin=self.name,
                    exists=True,
                )
            ],
        )


class VideoAgentSmokeTest(unittest.TestCase):
    def test_core_smoke_run_creates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-job",
                    "idea": "A modular agent composes a teaser.",
                    "script": "The agent reads the brief, speaks the narration, and renders a final video.",
                    "use_voice": True,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {"force_single_scene": True},
                }
            )

            job_dir = store.job_dir("smoke-job")
            self.assertTrue(result.success)
            self.assertEqual(result.final_phase, "assembled")
            self.assertTrue((job_dir / "input_job.json").exists())
            self.assertTrue((job_dir / "plan.json").exists())
            self.assertTrue((job_dir / "state.json").exists())
            self.assertTrue((job_dir / "result.json").exists())
            self.assertTrue((job_dir / "logs" / "agent.log").exists())
            self.assertTrue((job_dir / "final.mp4").exists())
            self.assertTrue(any(artifact.key == "voice_audio" for artifact in result.artifacts))
            self.assertTrue(any(artifact.key == "final_video" for artifact in result.artifacts))
            self.assertTrue(any(artifact.key == "final_output_mp4" for artifact in result.artifacts))
            self.assertEqual(result.output_final_path, str(job_dir / "final.mp4"))

    def test_core_smoke_run_without_voice_still_creates_final_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-no-voice",
                    "idea": "A modular agent renders a teaser without narration.",
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                }
            )

            job_dir = store.job_dir("smoke-no-voice")
            self.assertTrue(result.success)
            self.assertTrue((job_dir / "final.mp4").exists())
            self.assertIsNone(result.output_audio_path)
            self.assertEqual(result.output_final_path, str(job_dir / "final.mp4"))
            self.assertIn("without voice", result.message.lower())
            self.assertTrue(any(artifact.key == "final_output_mp4" for artifact in result.artifacts))

    def test_core_multi_scene_run_creates_scene_plan_and_scene_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-multi-scene",
                    "idea": "A modular agent stages a multi-scene teaser.",
                    "script": "Scene one shows the system booting. Scene two shows validation and planning. Scene three shows rendering and export.",
                    "duration_sec": 10,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                }
            )

            job_dir = store.job_dir("smoke-multi-scene")
            self.assertTrue(result.success)
            self.assertTrue((job_dir / "scene_plan.json").exists())
            self.assertTrue(any(artifact.key == "scene_plan_file" for artifact in result.artifacts))
            self.assertTrue(any(artifact.key.startswith("scene_") and artifact.key.endswith("_video") for artifact in result.artifacts))
            self.assertTrue(any(artifact.key == "assembled_video" for artifact in result.artifacts))
            self.assertEqual(result.metadata["scene_count"], 3)

    def test_multi_scene_multi_take_prefers_technically_valid_take_and_persists_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-multi-take",
                    "idea": "A modular agent stages multiple takes.",
                    "script": "Scene one shows planning. Scene two shows rendering.",
                    "duration_sec": 8,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {"scene_count": 2, "takes_per_scene": 2, "reject_take_indices": [1]},
                }
            )

            job_dir = store.job_dir("smoke-multi-take")
            state_payload = json.loads((job_dir / "state.json").read_text())
            take_payload = json.loads((job_dir / "takes.json").read_text())
            self.assertTrue(result.success)
            self.assertTrue((job_dir / "takes.json").exists())
            self.assertEqual(result.metadata["selection_mode"], "quality_guarded_best_valid_take")
            self.assertEqual(take_payload["takes_per_scene"], 2)
            self.assertTrue(any(artifact.key == "take_report_file" for artifact in result.artifacts))
            self.assertEqual(len(take_payload["scene_outputs"]), 2)
            self.assertTrue(all(scene["selected_take_id"].endswith("take_02") for scene in take_payload["scene_outputs"]))
            self.assertTrue(
                all(scene["takes"][0]["review_status"] == "rejected" for scene in take_payload["scene_outputs"])
            )
            self.assertEqual(state_payload["steps"]["video"]["details"]["selection_mode"], "quality_guarded_best_valid_take")
            self.assertTrue(state_payload["steps"]["video"]["details"]["quality_guard_enabled"])

    def test_multi_scene_quality_retry_recovers_from_invalid_take(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-quality-retry",
                    "idea": "A modular agent recovers from a technically invalid take.",
                    "script": "Scene one boots. Scene two exports.",
                    "duration_sec": 8,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {"scene_count": 2, "takes_per_scene": 1, "reject_take_indices": [1]},
                }
            )

            job_dir = store.job_dir("smoke-quality-retry")
            state_payload = json.loads((job_dir / "state.json").read_text())
            take_payload = json.loads((job_dir / "takes.json").read_text())
            self.assertTrue(result.success)
            scene_outputs = take_payload["scene_outputs"]
            self.assertEqual(take_payload["total_retry_count"], 2)
            self.assertTrue(all(scene["selected_take_id"].endswith("retry_01") for scene in scene_outputs))
            self.assertTrue(all(len(scene["retry_history"]) == 1 for scene in scene_outputs))
            self.assertTrue(
                all(any(take["review_status"] == "rejected" for take in scene["takes"]) for scene in scene_outputs)
            )
            self.assertTrue(
                all(scene["selected_take"]["review_status"] == "selected" for scene in scene_outputs)
            )
            self.assertEqual(state_payload["steps"]["video"]["details"]["total_retry_count"], 2)

    def test_multi_scene_take_fallback_survives_single_take_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-take-fallback",
                    "idea": "A modular agent recovers from an early take failure.",
                    "script": "Scene one boots. Scene two exports.",
                    "duration_sec": 8,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {"scene_count": 2, "takes_per_scene": 3, "fail_take_indices": [1]},
                }
            )

            self.assertTrue(result.success)
            scene_outputs = result.backend_runs["video"]["metadata"]["scene_outputs"]
            self.assertTrue(all(scene["selected_take_id"].endswith("take_02") for scene in scene_outputs))
            self.assertTrue(
                all(any(take["status"] == "failed" for take in scene["takes"]) for scene in scene_outputs)
            )

    def test_variation_multi_take_flow_persists_variant_data_and_can_select_later_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-variation-flow",
                    "idea": "A modular agent renders controlled creative variants.",
                    "script": "One scene shows the system booting in multiple visual treatments.",
                    "duration_sec": 6,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {
                        "force_single_scene": True,
                        "variations_per_scene": 2,
                        "takes_per_scene": 2,
                        "reject_take_indices": [1, 2],
                    },
                }
            )

            job_dir = store.job_dir("smoke-variation-flow")
            scene_plan_payload = json.loads((job_dir / "scene_plan.json").read_text())
            state_payload = json.loads((job_dir / "state.json").read_text())
            take_payload = json.loads((job_dir / "takes.json").read_text())
            self.assertTrue(result.success)
            self.assertEqual(scene_plan_payload["scene_count"], 1)
            self.assertEqual(len(scene_plan_payload["scenes"][0]["variations"]), 2)
            self.assertEqual(take_payload["variations_per_scene"], 2)
            self.assertEqual(take_payload["takes_per_variation"], 2)
            self.assertEqual(take_payload["takes_per_scene"], 4)
            scene_output = take_payload["scene_outputs"][0]
            self.assertEqual(scene_output["variation_count"], 2)
            self.assertEqual(len(scene_output["variations"]), 2)
            self.assertEqual(scene_output["selected_variation_id"], "scene_01_var_02")
            self.assertEqual(scene_output["selected_variation"]["variation_id"], "scene_01_var_02")
            self.assertTrue(all(take["variation_id"] for take in scene_output["takes"]))
            self.assertTrue(all(take["shot_type"] for take in scene_output["takes"]))
            self.assertEqual(state_payload["steps"]["video"]["details"]["variations_per_scene"], 2)
            self.assertEqual(state_payload["steps"]["video"]["details"]["takes_per_variation"], 2)
            self.assertEqual(
                result.metadata["selected_scene_outputs"][0]["selected_variation_id"],
                "scene_01_var_02",
            )
            self.assertIn("technical_score", scene_output)
            self.assertIn("creative_score", scene_output)
            self.assertIn("selected_by_rule", scene_output)
            self.assertIn("selection_reason", scene_output)
            self.assertIn("technical_selection_status", scene_output)
            self.assertIn("creative_selection_status", scene_output)
            self.assertIn("selection_scores", scene_output["selected_take"]["metadata"])
            self.assertEqual(
                state_payload["steps"]["video"]["details"]["selected_scene_outputs"][0]["selected_by_rule"],
                scene_output["selected_by_rule"],
            )

    def test_creative_selection_prefers_establishing_for_opening_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-creative-opening",
                    "idea": "A modular agent opens on a new scene.",
                    "script": "The system boots and reveals the workspace.",
                    "duration_sec": 6,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {
                        "force_single_scene": True,
                        "variations_per_scene": 3,
                        "takes_per_scene": 1,
                    },
                }
            )

            scene_output = result.backend_runs["video"]["metadata"]["scene_outputs"][0]
            self.assertTrue(result.success)
            self.assertEqual(scene_output["selected_variation_id"], "scene_01_var_01")
            self.assertEqual(scene_output["selected_take"]["shot_type"], "establishing")
            self.assertEqual(scene_output["selected_by_rule"], "opening_prefers_establishing")
            self.assertEqual(scene_output["creative_selection_status"], "single_best_creative_candidate")

    def test_creative_selection_avoids_repeating_adjacent_shot_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-creative-adjacent",
                    "idea": "A modular agent varies adjacent scene coverage.",
                    "script": "Scene one shows the system booting. Scene two shows render progress moving across the interface.",
                    "duration_sec": 8,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {
                        "scene_count": 2,
                        "variations_per_scene": 2,
                        "takes_per_scene": 1,
                    },
                }
            )

            scene_outputs = result.backend_runs["video"]["metadata"]["scene_outputs"]
            self.assertTrue(result.success)
            self.assertEqual(scene_outputs[0]["selected_take"]["shot_type"], "establishing")
            self.assertEqual(scene_outputs[1]["selected_take"]["shot_type"], "medium_action")
            self.assertEqual(scene_outputs[1]["selected_by_rule"], "scene_goal_motion_match")
            self.assertTrue(
                any(
                    rule["rule"] == "adjacent_diversity_bonus"
                    for rule in scene_outputs[1]["selection"]["rule_hits"]
                )
            )

    def test_creative_selection_tie_break_falls_back_to_first_valid_take(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = BackendRegistry([FakeVoiceAdapter(), FakeVideoAdapter()])
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=registry,
                state_store=store,
                planner=ProductionPlanner(registry),
                assembler=ResultAssembler(),
            )

            result = agent.run_job(
                {
                    "job_id": "smoke-creative-tie-break",
                    "idea": "A modular agent resolves equal creative candidates deterministically.",
                    "script": "One clean scene only.",
                    "duration_sec": 6,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {
                        "force_single_scene": True,
                        "variations_per_scene": 1,
                        "takes_per_scene": 2,
                    },
                }
            )

            scene_output = result.backend_runs["video"]["metadata"]["scene_outputs"][0]
            self.assertTrue(result.success)
            self.assertEqual(scene_output["selected_take_id"], "scene_01_take_01")
            self.assertEqual(scene_output["creative_selection_status"], "creative_tie_first_valid_fallback")
            self.assertEqual(scene_output["selected_by_rule"], "first_successful_take")
            self.assertTrue(scene_output["selection"]["fallback_used"])
            self.assertIn("fell back to first successful valid take", scene_output["selection_reason"])


if __name__ == "__main__":
    unittest.main()
