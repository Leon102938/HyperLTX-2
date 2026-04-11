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
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x240:r=24",
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
                    "resolution": "draft",
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
                    "resolution": "draft",
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
                    "resolution": "draft",
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

    def test_multi_scene_multi_take_selects_first_successful_take_and_persists_report(self) -> None:
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
                    "resolution": "draft",
                    "orientation": "landscape",
                    "metadata": {"scene_count": 2, "takes_per_scene": 2, "fail_take_indices": [1]},
                }
            )

            job_dir = store.job_dir("smoke-multi-take")
            state_payload = json.loads((job_dir / "state.json").read_text())
            take_payload = json.loads((job_dir / "takes.json").read_text())
            self.assertTrue(result.success)
            self.assertTrue((job_dir / "takes.json").exists())
            self.assertEqual(result.metadata["selection_mode"], "first_successful_take")
            self.assertEqual(take_payload["takes_per_scene"], 2)
            self.assertTrue(any(artifact.key == "take_report_file" for artifact in result.artifacts))
            self.assertEqual(len(take_payload["scene_outputs"]), 2)
            self.assertTrue(all(scene["selected_take_id"].endswith("take_02") for scene in take_payload["scene_outputs"]))
            self.assertEqual(state_payload["steps"]["video"]["details"]["selection_mode"], "first_successful_take")

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
                    "resolution": "draft",
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


if __name__ == "__main__":
    unittest.main()
