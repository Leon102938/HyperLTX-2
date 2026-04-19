import json
import os
import threading
import subprocess
import tempfile
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from agent_core.agent import VideoAgent
from agent_core.backend_registry import BackendRegistry
from agent_core.director import DirectorEngine
from agent_core.llm_adapter import LocalOpenAICompatibleLLMAdapter
from agent_core.planner import ProductionPlanner
from agent_core.schemas import ArtifactRef, BackendCapabilities, ExecutionResult, JobInput, ProductionPlan
from agent_core.state_store import StateStore
from agent_core.adapters.base import VideoAdapter, VoiceAdapter


class DirectorVoiceAdapter(VoiceAdapter):
    name = "director_voice"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, kind="voice", available=True, phase1_enabled=True)

    def generate_voice(self, job: JobInput, plan: ProductionPlan, workspace: Path) -> ExecutionResult:
        workspace.mkdir(parents=True, exist_ok=True)
        audio_path = workspace / "director_voice.wav"
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


class DirectorVideoAdapter(VideoAdapter):
    name = "director_video"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            kind="video",
            available=True,
            phase1_enabled=True,
            supported_pipelines=["ti2vid"],
            supports_image_conditioning=True,
        )

    def generate_video(
        self,
        job: JobInput,
        plan: ProductionPlan,
        workspace: Path,
        voice_result: ExecutionResult | None = None,
    ) -> ExecutionResult:
        workspace.mkdir(parents=True, exist_ok=True)
        output_path = workspace / "director_video.mp4"
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
                str(output_path),
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
            output_path=str(output_path),
            duration_sec=plan.target_duration_sec,
            artifacts=[ArtifactRef(key="final_video", kind="video", path=str(output_path), origin=self.name, exists=True)],
        )


class FakeDirectorLLM:
    def plan_director(self, *, job: JobInput, scene_beats, fallback_output):  # noqa: ANN001
        payload = fallback_output.model_dump(mode="json")
        payload["mode"] = "llm_augmented"
        payload["creative_brief"]["hook"] = "Open on a stronger director-crafted reveal."
        payload["scene_intents"][0]["hook_focus"] = "director llm strengthened the opening hook"
        return {
            "ok": True,
            "provider": "local_openai_compatible",
            "model": "fake-qwen-like",
            "payload": payload,
        }


class DirectorLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        sanitized_env = {key: value for key, value in os.environ.items() if not key.startswith("DIRECTOR_LLM_")}
        self._env_patch = patch.dict(os.environ, sanitized_env, clear=True)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        self.registry = BackendRegistry([DirectorVoiceAdapter(), DirectorVideoAdapter()])

    def _start_fake_openai_server(
        self,
        *,
        include_reasoning_prefix: bool = False,
        wrap_scene_map: str | None = None,
    ) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                request_payload = json.loads(raw.decode("utf-8"))
                prompt_payload = json.loads(request_payload["messages"][1]["content"])
                response_payload = prompt_payload["required_json_shape"]
                if "creative_brief" in response_payload:
                    response_payload["mode"] = "llm_augmented"
                    response_payload["creative_brief"]["hook"] = "Local Qwen-style director server sharpened the opening reveal."
                    response_payload["scene_intents"][0]["hook_focus"] = "server-backed director hook"
                    response_payload["metadata"]["fake_server"] = "ok"
                else:
                    first_scene_id = next(iter(response_payload))
                    response_payload[first_scene_id]["visual_concept"] = "Server-backed director hook and a stronger opening reveal."
                    if "prompt_seed" in response_payload[first_scene_id]:
                        response_payload[first_scene_id]["prompt_seed"] = "clean hero seed for the opening shot"
                        response_payload[first_scene_id]["camera_movement"] = "slow push-in"
                    else:
                        response_payload[first_scene_id]["lighting_design"] = "moody practical reflections with controlled contrast"
                        response_payload[first_scene_id]["color_grading"] = "cool neon blues with warm metallic highlights"
                if wrap_scene_map:
                    response_payload = {wrap_scene_map: response_payload}
                content = json.dumps(response_payload)
                if include_reasoning_prefix:
                    content = f"<think>hidden scratchpad</think>\n{content}"
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": content,
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):  # noqa: A003
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, f"http://127.0.0.1:{server.server_port}"

    def test_director_output_structure_falls_back_without_llm(self) -> None:
        planner = ProductionPlanner(self.registry)
        job = JobInput(
            idea="A director layer strengthens the hook.",
            script="Scene one reveals the system. Scene two shows the render.",
            duration_sec=8,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
        )

        plan = planner.build_plan(job)

        self.assertIsNotNone(plan.director_output)
        self.assertEqual(plan.director_output.mode, "rule_based_fallback")
        self.assertEqual(plan.director_output.fallback_reason, "director_llm_not_configured")
        self.assertTrue(plan.director_output.creative_brief.hook)
        self.assertTrue(plan.director_output.style_lock.visual_identity)
        self.assertEqual(len(plan.director_output.scene_intents), len(plan.scenes))
        self.assertFalse(plan.director_output.llm_active)

    def test_prompt_building_adds_opening_metadata_and_creative_intent(self) -> None:
        planner = ProductionPlanner(self.registry)
        job = JobInput(
            idea="A director layer sharpens prompts.",
            script="The system boots and reveals the workspace.",
            duration_sec=6,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
            metadata={"force_single_scene": True, "variations_per_scene": 2},
        )

        plan = planner.build_plan(job)
        scene = plan.scenes[0]

        self.assertIn("Opening shot:", scene.prompt_text)
        self.assertEqual(scene.prompt_build_metadata["builder_version"], "phase5a_director_v1")
        self.assertTrue(scene.variations[0].creative_intent)
        self.assertEqual(scene.variations[0].prompt_build_metadata["prompt_kind"], "variation")

    def test_fake_llm_director_path_can_override_fallback(self) -> None:
        director = DirectorEngine(llm_adapter=FakeDirectorLLM())
        planner = ProductionPlanner(self.registry, director=director)
        job = JobInput(
            idea="A configured local director model refines the hook.",
            script="The system boots into frame.",
            duration_sec=6,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
            metadata={"force_single_scene": True},
        )

        plan = planner.build_plan(job)

        self.assertEqual(plan.director_output.mode, "llm_augmented")
        self.assertEqual(plan.director_output.llm_model, "fake-qwen-like")
        self.assertTrue(plan.director_output.llm_active)
        self.assertIn("director-crafted reveal", plan.director_output.creative_brief.hook)

    def test_local_llama_cpp_profile_can_use_reachable_openai_compatible_server(self) -> None:
        server, thread, base_url = self._start_fake_openai_server()
        try:
            planner = ProductionPlanner(self.registry)
            job = JobInput(
                idea="A real local OpenAI-compatible server refines the director output.",
                script="The node boots and the opening shot locks in.",
                duration_sec=6,
                use_voice=False,
                resolution="draft",
                orientation="landscape",
                metadata={
                    "force_single_scene": True,
                    "director_llm": {
                        "profile": "qwen36_llama_cpp_local",
                        "base_url": base_url,
                        "enabled": True,
                    }
                },
            )

            plan = planner.build_plan(job)

            self.assertEqual(plan.director_output.mode, "llm_augmented")
            self.assertTrue(plan.director_output.llm_active)
            self.assertEqual(plan.director_output.llm_provider, "llama_cpp_local")
            self.assertEqual(plan.director_output.llm_model, "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf")
            self.assertEqual(plan.director_output.llm_endpoint, f"{base_url}/v1/chat/completions")
            self.assertTrue(plan.director_output.creative_brief.hook)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_local_llama_cpp_profile_can_parse_qwen_reasoning_preface(self) -> None:
        server, thread, base_url = self._start_fake_openai_server(include_reasoning_prefix=True)
        try:
            planner = ProductionPlanner(self.registry)
            job = JobInput(
                idea="The local Qwen server may emit hidden reasoning tags before the JSON object.",
                script="The opening still needs a structured director response.",
                duration_sec=6,
                use_voice=False,
                resolution="draft",
                orientation="landscape",
                metadata={
                    "force_single_scene": True,
                    "director_llm": {
                        "profile": "qwen36_llama_cpp_local",
                        "base_url": base_url,
                        "enabled": True,
                    }
                },
            )

            plan = planner.build_plan(job)

            self.assertEqual(plan.director_output.mode, "llm_augmented")
            self.assertTrue(plan.director_output.llm_active)
            self.assertEqual(plan.director_output.llm_model, "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf")
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_local_llama_cpp_profile_can_parse_wrapped_scene_map_payload(self) -> None:
        server, thread, base_url = self._start_fake_openai_server(wrap_scene_map="scene_map")
        try:
            planner = ProductionPlanner(self.registry)
            job = JobInput(
                idea="A wrapped scene_map payload should still normalize into llm_augmented output.",
                script="The opening reveals the car. The middle beat circles the subject. The ending lands on a hero frame.",
                duration_sec=10,
                use_voice=False,
                resolution="draft",
                orientation="landscape",
                metadata={
                    "director_llm": {
                        "profile": "qwen36_llama_cpp_local",
                        "base_url": base_url,
                        "enabled": True,
                    }
                },
            )

            plan = planner.build_plan(job)

            self.assertEqual(plan.director_output.mode, "llm_augmented")
            self.assertTrue(plan.director_output.llm_active)
            self.assertEqual(plan.director_output.llm_provider, "llama_cpp_local")
            self.assertTrue(plan.director_output.scene_intents[0].hook_focus)
            self.assertGreaterEqual(len(plan.director_output.scene_intents[0].variation_directives), 1)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_local_llama_cpp_compact_scene_map_keeps_rule_based_variation_structure(self) -> None:
        server, thread, base_url = self._start_fake_openai_server()
        try:
            planner = ProductionPlanner(self.registry)
            job = JobInput(
                idea="A compact local scene map should still normalize into the existing planner contract.",
                script="The opening reveals the car. The middle beat circles the subject. The ending lands on a hero frame.",
                duration_sec=10,
                use_voice=False,
                resolution="draft",
                orientation="landscape",
                metadata={
                    "director_llm": {
                        "profile": "qwen36_llama_cpp_local",
                        "base_url": base_url,
                        "enabled": True,
                    }
                },
            )

            plan = planner.build_plan(job)

            self.assertEqual(plan.director_output.mode, "llm_augmented")
            self.assertTrue(plan.director_output.llm_active)
            self.assertIn("clean hero seed", plan.director_output.scene_intents[0].hook_focus.lower())
            self.assertGreaterEqual(len(plan.director_output.scene_intents[0].variation_directives), 1)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_local_llama_cpp_profile_falls_back_when_server_is_unreachable(self) -> None:
        planner = ProductionPlanner(self.registry)
        job = JobInput(
            idea="The local director server is offline, so fallback must stay clean.",
            script="The opening still needs to stay coherent.",
            duration_sec=6,
            use_voice=False,
            resolution="draft",
            orientation="landscape",
            metadata={
                "force_single_scene": True,
                "director_llm": {
                    "profile": "qwen36_llama_cpp_local",
                    "base_url": "http://127.0.0.1:9",
                    "enabled": True,
                    "timeout_sec": 0.2,
                }
            },
        )

        plan = planner.build_plan(job)

        self.assertEqual(plan.director_output.mode, "rule_based_fallback")
        self.assertFalse(plan.director_output.llm_active)
        self.assertEqual(plan.director_output.llm_provider, "llama_cpp_local")
        self.assertEqual(plan.director_output.llm_model, "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf")
        self.assertTrue(plan.director_output.fallback_reason.startswith("director_llm_request_failed:"))

    def test_local_llama_cpp_budget_is_capped_for_small_context(self) -> None:
        single_scene = LocalOpenAICompatibleLLMAdapter._effective_max_tokens(
            provider="llama_cpp_local",
            requested_max_tokens=640,
            ctx_tokens=2048,
            scene_count=1,
            system_prompt="system prompt",
            user_prompt="x" * 400,
        )
        self.assertEqual(single_scene, 256)

        multi_scene = LocalOpenAICompatibleLLMAdapter._effective_max_tokens(
            provider="llama_cpp_local",
            requested_max_tokens=640,
            ctx_tokens=2048,
            scene_count=3,
            system_prompt="system prompt",
            user_prompt="x" * 400,
        )
        self.assertEqual(multi_scene, 576)

        constrained = LocalOpenAICompatibleLLMAdapter._effective_max_tokens(
            provider="llama_cpp_local",
            requested_max_tokens=640,
            ctx_tokens=512,
            scene_count=3,
            system_prompt="system prompt",
            user_prompt="x" * 1200,
        )
        self.assertEqual(constrained, 32)

    def test_agent_persists_director_output_and_prompt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(Path(tmpdir) / "runs")
            agent = VideoAgent(
                registry=self.registry,
                state_store=store,
                planner=ProductionPlanner(self.registry),
            )

            result = agent.run_job(
                {
                    "job_id": "director-persist",
                    "idea": "A director layer persists its planning output.",
                    "script": "The system boots. The render resolves.",
                    "duration_sec": 8,
                    "use_voice": False,
                    "resolution": "320x256",
                    "orientation": "landscape",
                    "metadata": {"scene_count": 2, "variations_per_scene": 2, "takes_per_scene": 1, "use_storyboard": False},
                }
            )

            job_dir = store.job_dir("director-persist")
            director_payload = json.loads((job_dir / "director_output.json").read_text())
            scene_plan_payload = json.loads((job_dir / "scene_plan.json").read_text())
            takes_payload = json.loads((job_dir / "takes.json").read_text())

            self.assertTrue(result.success)
            self.assertEqual(director_payload["director_mode"], "rule_based_fallback")
            self.assertFalse(director_payload["director_llm_active"])
            self.assertEqual(director_payload["director_fallback_reason"], "director_llm_not_configured")
            self.assertEqual(director_payload["director_output"]["mode"], "rule_based_fallback")
            self.assertFalse(director_payload["llm"]["active"])
            self.assertTrue(director_payload["style_lock"]["camera_language"])
            self.assertIsNotNone(scene_plan_payload["director_output"])
            self.assertIsNotNone(scene_plan_payload["style_lock"])
            self.assertEqual(takes_payload["director_mode"], "rule_based_fallback")
            self.assertFalse(takes_payload["director_llm_active"])
            self.assertIn("prompt_build_metadata", takes_payload["scene_outputs"][0])
            self.assertIsNotNone(result.metadata["director_output"])
            self.assertEqual(result.metadata["director_mode"], "rule_based_fallback")
            self.assertFalse(result.metadata["director_llm_active"])
            self.assertEqual(result.metadata["director_fallback_reason"], "director_llm_not_configured")


if __name__ == "__main__":
    unittest.main()
