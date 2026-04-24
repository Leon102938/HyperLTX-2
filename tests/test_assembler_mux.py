import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from agent_core.assembler import ResultAssembler
from agent_core.schemas import ExecutionResult, JobInput, JobState, ProductionPlan, ScenePlan
from agent_core.utils import probe_media_duration, utc_now_iso


def _build_video(path: Path, duration_sec: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=24",
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


def _build_wav(path: Path, duration_sec: float, sample_rate: int = 24000) -> None:
    frame_count = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


def _build_plan(job_id: str, duration_sec: float) -> ProductionPlan:
    return ProductionPlan(
        job_id=job_id,
        orientation="landscape",
        resolution_label="draft",
        width=320,
        height=240,
        render_profile="balanced",
        selected_pipeline="ti2vid",
        requested_duration_sec=duration_sec,
        target_duration_sec=duration_sec,
        prompt_text="test prompt",
        metadata={"frame_rate": 24},
    )


def _build_state(job_id: str) -> JobState:
    now = utc_now_iso()
    return JobState(
        job_id=job_id,
        status="video_generated",
        current_phase="video_generated",
        created_at=now,
        updated_at=now,
    )


class AssemblerMuxTest(unittest.TestCase):
    def test_mux_pads_short_voice_to_video_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            audio_path = workspace / "short.wav"
            _build_video(video_path, 2.0)
            _build_wav(audio_path, 1.0)

            result = ResultAssembler().assemble(
                JobInput(job_id="pad-case", idea="pad"),
                _build_plan("pad-case", 2.0),
                _build_state("pad-case"),
                workspace,
                ExecutionResult(
                    step_name="voice",
                    success=True,
                    status="succeeded",
                    backend_name="fake_voice",
                    output_path=str(audio_path),
                    duration_sec=probe_media_duration(str(audio_path)),
                ),
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    output_path=str(video_path),
                    duration_sec=probe_media_duration(str(video_path)),
                ),
            )

            self.assertEqual(result.metadata["assembly"]["timing_mode"], "voice_padded_to_video")
            self.assertAlmostEqual(result.actual_final_duration_sec, 2.0, places=1)

    def test_mux_trims_long_voice_to_video_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            audio_path = workspace / "long.wav"
            _build_video(video_path, 2.0)
            _build_wav(audio_path, 4.0)

            result = ResultAssembler().assemble(
                JobInput(job_id="trim-case", idea="trim"),
                _build_plan("trim-case", 2.0),
                _build_state("trim-case"),
                workspace,
                ExecutionResult(
                    step_name="voice",
                    success=True,
                    status="succeeded",
                    backend_name="fake_voice",
                    output_path=str(audio_path),
                    duration_sec=probe_media_duration(str(audio_path)),
                ),
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    output_path=str(video_path),
                    duration_sec=probe_media_duration(str(video_path)),
                ),
            )

            self.assertEqual(result.metadata["assembly"]["timing_mode"], "voice_trimmed_to_video")
            self.assertAlmostEqual(result.actual_final_duration_sec, 2.0, places=1)

    def test_no_voice_artifact_copies_rendered_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            _build_video(video_path, 2.0)

            result = ResultAssembler().assemble(
                JobInput(job_id="no-voice-case", idea="no voice"),
                _build_plan("no-voice-case", 2.0),
                _build_state("no-voice-case"),
                workspace,
                None,
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    output_path=str(video_path),
                    duration_sec=probe_media_duration(str(video_path)),
                ),
            )

            self.assertEqual(result.metadata["assembly"]["timing_mode"], "no_voice_artifact")
            self.assertAlmostEqual(result.actual_final_duration_sec, 2.0, places=1)

    def test_multi_scene_assembly_uses_selected_takes_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            scene_a = workspace / "scene_a.mp4"
            scene_b = workspace / "scene_b.mp4"
            discarded = workspace / "discarded.mp4"
            _build_video(scene_a, 1.0)
            _build_video(scene_b, 1.5)
            _build_video(discarded, 0.5)

            result = ResultAssembler().assemble(
                JobInput(job_id="selected-takes-case", idea="selected takes"),
                _build_plan("selected-takes-case", 2.5),
                _build_state("selected-takes-case"),
                workspace,
                None,
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    duration_sec=2.5,
                    metadata={
                        "scene_outputs": [
                            {
                                "scene_id": "scene_01",
                                "output_path": str(discarded),
                                "selected_take_id": "scene_01_take_01",
                            },
                            {
                                "scene_id": "scene_02",
                                "output_path": str(discarded),
                                "selected_take_id": "scene_02_take_01",
                            },
                        ],
                        "selected_scene_outputs": [
                            {
                                "scene_id": "scene_01",
                                "output_path": str(scene_a),
                                "selected_take_id": "scene_01_take_02",
                                "review_status": "selected",
                                "validation": {"passed": True},
                            },
                            {
                                "scene_id": "scene_02",
                                "output_path": str(scene_b),
                                "selected_take_id": "scene_02_take_03",
                                "review_status": "selected",
                                "validation": {"passed": True},
                            },
                        ],
                        "selection_mode": "quality_guarded_best_valid_take",
                    },
                ),
            )

            self.assertEqual(result.metadata["selection_mode"], "quality_guarded_best_valid_take")
            self.assertEqual(
                [scene["selected_take_id"] for scene in result.metadata["selected_scene_outputs"]],
                ["scene_01_take_02", "scene_02_take_03"],
            )
            self.assertTrue((workspace / "assembled_video.mp4").exists())

    def test_multi_scene_assembly_rejects_non_valid_selected_take(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            scene_a = workspace / "scene_a.mp4"
            _build_video(scene_a, 1.0)

            with self.assertRaises(RuntimeError):
                ResultAssembler().assemble(
                    JobInput(job_id="invalid-selected-take", idea="invalid"),
                    _build_plan("invalid-selected-take", 1.0),
                    _build_state("invalid-selected-take"),
                    workspace,
                    None,
                    ExecutionResult(
                        step_name="video",
                        success=True,
                        status="succeeded",
                        backend_name="fake_video",
                        duration_sec=1.0,
                        metadata={
                            "selected_scene_outputs": [
                                {
                                    "scene_id": "scene_01",
                                    "output_path": str(scene_a),
                                    "selected_take_id": "scene_01_take_01",
                                    "review_status": "rejected",
                                    "validation": {"passed": False},
                                }
                            ]
                        },
                    ),
                )

    def test_enhanced_assembly_writes_subtitles_and_mixes_music(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            voice_path = workspace / "voice.wav"
            music_path = workspace / "music.wav"
            _build_video(video_path, 2.0)
            _build_wav(voice_path, 2.0)
            _build_wav(music_path, 2.0)

            plan = _build_plan("enhanced-case", 2.0).model_copy(
                update={
                    "scenes": [
                        ScenePlan(
                    scene_id="scene_01",
                    index=1,
                    title="Scene 1",
                    description="A useful beat.",
                    target_duration_sec=2.0,
                    num_frames=49,
                    prompt_text="test prompt",
                    narration_text="One sharp tip that stays readable.",
                    narration_start_sec=0.0,
                    narration_end_sec=2.0,
                )
                    ]
                }
            )

            result = ResultAssembler().assemble(
                JobInput(
                    job_id="enhanced-case",
                    idea="enhanced",
                    use_music=True,
                    metadata={"subtitle_mode": "sidecar", "overlay_text": "Quick Tip"},
                ),
                plan,
                _build_state("enhanced-case"),
                workspace,
                ExecutionResult(
                    step_name="voice",
                    success=True,
                    status="succeeded",
                    backend_name="fake_voice",
                    output_path=str(voice_path),
                    duration_sec=probe_media_duration(str(voice_path)),
                ),
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    output_path=str(video_path),
                    duration_sec=probe_media_duration(str(video_path)),
                ),
                music_result=ExecutionResult(
                    step_name="music",
                    success=True,
                    status="succeeded",
                    backend_name="fake_music",
                    output_path=str(music_path),
                    duration_sec=probe_media_duration(str(music_path)),
                ),
            )

            self.assertEqual(result.metadata["assembly"]["mode"], "enhanced_mix")
            self.assertEqual(result.metadata["assembly"]["subtitle_mode"], "sidecar")
            self.assertTrue((workspace / "captions.srt").exists())
            self.assertTrue((workspace / "final.mp4").exists())
            self.assertAlmostEqual(result.actual_final_duration_sec, 2.0, places=1)

    def test_enhanced_assembly_prefers_plan_subtitle_defaults_for_social_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            video_path = workspace / "video.mp4"
            voice_path = workspace / "voice.wav"
            _build_video(video_path, 4.5)
            _build_wav(voice_path, 4.5)

            plan = _build_plan("social-subtitle-case", 4.5).model_copy(
                update={
                    "metadata": {
                        "frame_rate": 24,
                        "subtitle_min_words": 3,
                        "subtitle_min_duration_sec": 1.1,
                    },
                    "scenes": [
                        ScenePlan(
                            scene_id="scene_01",
                            index=1,
                            title="Scene 1",
                            description="A social beat.",
                            target_duration_sec=4.5,
                            num_frames=109,
                            prompt_text="test prompt",
                            narration_text="Drei kleine Schritte, weniger Reibung, mehr Klarheit fuer den Rest des Tages.",
                            narration_start_sec=0.0,
                            narration_end_sec=4.5,
                        )
                    ],
                }
            )

            ResultAssembler().assemble(
                JobInput(
                    job_id="social-subtitle-case",
                    idea="social",
                    use_voice=True,
                    metadata={"subtitle_mode": "sidecar"},
                ),
                plan,
                _build_state("social-subtitle-case"),
                workspace,
                ExecutionResult(
                    step_name="voice",
                    success=True,
                    status="succeeded",
                    backend_name="fake_voice",
                    output_path=str(voice_path),
                    duration_sec=probe_media_duration(str(voice_path)),
                ),
                ExecutionResult(
                    step_name="video",
                    success=True,
                    status="succeeded",
                    backend_name="fake_video",
                    output_path=str(video_path),
                    duration_sec=probe_media_duration(str(video_path)),
                ),
            )

            captions = (workspace / "captions.srt").read_text(encoding="utf-8")
            self.assertIn("Drei kleine Schritte weniger Reibung", captions)
            self.assertNotIn("\nweniger Reibung\n", captions)


if __name__ == "__main__":
    unittest.main()
