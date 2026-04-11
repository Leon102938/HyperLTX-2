import json
import tempfile
import unittest
import wave
from pathlib import Path

from PIL import Image

from agent_core.agent import VideoAgent
from agent_core.adapters.base import StoryboardAdapter, VideoAdapter, VoiceAdapter
from agent_core.assembler import ResultAssembler
from agent_core.backend_registry import BackendRegistry
from agent_core.planner import ProductionPlanner
from agent_core.schemas import ArtifactRef, BackendCapabilities, ExecutionResult, JobInput, ProductionPlan
from agent_core.state_store import StateStore


class StoryboardFakeVoiceAdapter(VoiceAdapter):
    name = "storyboard_fake_voice"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="voice", available=True, phase1_enabled=True)

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
            duration_sec=1.0,
            artifacts=[ArtifactRef(key="voice_audio", kind="audio", path=str(audio_path), origin=self.name, exists=True)],
        )


class StoryboardFakeVideoAdapter(VideoAdapter):
    name = "storyboard_fake_video"

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
        import subprocess

        workspace.mkdir(parents=True, exist_ok=True)
        video_path = workspace / "fake_video.mp4"
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
            artifacts=[ArtifactRef(key="final_video", kind="video", path=str(video_path), origin=self.name, exists=True)],
        )


class StoryboardFakeAdapter(StoryboardAdapter):
    name = "storyboard_fake"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="storyboard", available=True, phase1_enabled=True)

    def generate_storyboard(self, job: JobInput, plan: ProductionPlan, workspace: Path) -> ExecutionResult:
        workspace.mkdir(parents=True, exist_ok=True)
        output_path = workspace / "storyboard.png"
        candidate_id = str(job.metadata.get("candidate_id") or "")
        reject_ids = {str(value) for value in job.metadata.get("reject_storyboard_candidate_ids", [])}
        if candidate_id in reject_ids:
            output_path.write_bytes(b"bad")
        else:
            Image.new("RGB", (plan.width, plan.height), color=(32, 32, 32)).save(output_path, compress_level=0)
        return ExecutionResult(
            step_name="storyboard",
            success=True,
            status="succeeded",
            backend_name=self.name,
            backend_job_id=f"{job.job_id}_storyboard",
            output_path=str(output_path),
            artifacts=[ArtifactRef(key="storyboard_image", kind="image", path=str(output_path), origin=self.name, exists=True)],
        )


class StoryboardPipelineTest(unittest.TestCase):
    def _agent(self, tmpdir: str) -> tuple[VideoAgent, StateStore]:
        registry = BackendRegistry(
            [StoryboardFakeVoiceAdapter(), StoryboardFakeAdapter(), StoryboardFakeVideoAdapter()]
        )
        store = StateStore(Path(tmpdir) / "runs")
        agent = VideoAgent(
            registry=registry,
            state_store=store,
            planner=ProductionPlanner(registry),
            assembler=ResultAssembler(),
        )
        return agent, store

    def test_storyboard_pipeline_persists_selected_keyframe_and_relates_to_takes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, store = self._agent(tmpdir)
            result = agent.run_job(
                {
                    "job_id": "storyboard-smoke",
                    "idea": "A modular agent previsualizes a scene.",
                    "script": "Scene one opens on the pod waking up.",
                    "duration_sec": 6,
                    "use_voice": False,
                    "use_storyboard": True,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {
                        "force_single_scene": True,
                        "variations_per_scene": 2,
                        "takes_per_scene": 1,
                        "storyboard_candidates_per_scene": 2,
                    },
                }
            )

            job_dir = store.job_dir("storyboard-smoke")
            storyboard_payload = json.loads((job_dir / "storyboard_plan.json").read_text())
            takes_payload = json.loads((job_dir / "takes.json").read_text())
            state_payload = json.loads((job_dir / "state.json").read_text())

            self.assertTrue(result.success)
            self.assertTrue((job_dir / "storyboard_plan.json").exists())
            self.assertTrue(any(artifact.key == "storyboard_plan_file" for artifact in result.artifacts))
            self.assertEqual(storyboard_payload["candidate_count"], 2)
            self.assertEqual(len(storyboard_payload["selected_scene_storyboards"]), 1)
            self.assertIsNotNone(takes_payload["scene_outputs"][0]["selected_keyframe"])
            self.assertIsNotNone(takes_payload["scene_outputs"][0]["selected_take"]["metadata"]["selected_keyframe"])
            self.assertEqual(
                state_payload["steps"]["storyboard"]["details"]["selected_scene_storyboards"][0]["scene_id"],
                "scene_01",
            )

    def test_storyboard_step_is_skipped_cleanly_when_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, store = self._agent(tmpdir)
            result = agent.run_job(
                {
                    "job_id": "storyboard-off",
                    "idea": "A modular agent renders without storyboard.",
                    "duration_sec": 4,
                    "use_voice": False,
                    "use_storyboard": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                }
            )

            job_dir = store.job_dir("storyboard-off")
            state_payload = json.loads((job_dir / "state.json").read_text())
            self.assertTrue(result.success)
            self.assertFalse((job_dir / "storyboard_plan.json").exists())
            self.assertEqual(state_payload["steps"]["storyboard"]["status"], "skipped")
            self.assertFalse(result.metadata["storyboard_enabled"])

    def test_storyboard_selection_falls_back_when_preferred_candidate_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, store = self._agent(tmpdir)
            result = agent.run_job(
                {
                    "job_id": "storyboard-fallback",
                    "idea": "A modular agent falls back to a second keyframe candidate.",
                    "script": "Scene one opens on the pod waking up.",
                    "duration_sec": 6,
                    "use_voice": False,
                    "use_storyboard": True,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {
                        "force_single_scene": True,
                        "variations_per_scene": 2,
                        "takes_per_scene": 1,
                        "storyboard_candidates_per_scene": 2,
                        "reject_storyboard_candidate_ids": ["scene_01_var_01_keyframe_01"],
                    },
                }
            )

            storyboard_payload = json.loads((store.job_dir("storyboard-fallback") / "storyboard_plan.json").read_text())
            scene_storyboard = storyboard_payload["scene_storyboards"][0]
            self.assertTrue(result.success)
            self.assertEqual(scene_storyboard["selected_keyframe"]["candidate_id"], "scene_01_var_02_keyframe_02")
            self.assertEqual(scene_storyboard["selection"]["selected_by_rule"], "priority_rank_first_valid")


if __name__ == "__main__":
    unittest.main()
