from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_core.creative_os.phase1_runtime import Phase1RunConfig, run_phase1
from agent_core.creative_os.skill_tree_v1 import load_skill_tree


class SkillTreeV1Tests(unittest.TestCase):
    def test_manifest_loads_real_skill_files(self) -> None:
        config = SimpleNamespace(mode="calm_evergreen", style="clean_warm_lifestyle", topic="calm evergreen morning routine tip")
        match, tree = load_skill_tree(config)

        self.assertEqual("skill_tree_v1", match["version"])
        self.assertEqual("ok", match["status"])
        self.assertIn("modes/calm_evergreen", match["loaded_skill_ids"])
        self.assertIn("styles/clean_warm_lifestyle", match["loaded_skill_ids"])
        self.assertIn("hooks/soft_observation_hook", match["loaded_skill_ids"])
        self.assertIn("models/hidream_o1_no_unwanted_text_rules", match["loaded_skill_ids"])
        self.assertIn("models/hidream_o1_storyboard_rules", match["loaded_skill_ids"])
        self.assertIn("Open with a quiet observation.", tree["loaded_skills"]["modes"][0]["rules"])

    def test_missing_skill_file_is_marked_missing_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_root = Path(tmp) / "skills"
            shutil.copytree("/workspace/skills", skill_root)
            (skill_root / "styles" / "clean_warm_lifestyle.md").unlink()

            config = SimpleNamespace(mode="calm_evergreen", style="clean_warm_lifestyle", topic="calm evergreen morning routine tip")
            match, tree = load_skill_tree(config, skill_root=skill_root)

            self.assertEqual("missing", match["status"])
            self.assertIn("styles/clean_warm_lifestyle", match["missing_skill_ids"])
            self.assertEqual("missing", tree["missing_skills"][0]["status"])

    def test_stage03_writes_skill_match_and_tree_from_real_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_phase1(
                Phase1RunConfig(
                    job_id="skill-tree-stage03",
                    topic="calm evergreen morning routine tip",
                    mode="calm_evergreen",
                    style="clean_warm_lifestyle",
                    runs_root=Path(tmp),
                    attempt_images=False,
                )
            )
            run_dir = Path(tmp) / "skill-tree-stage03" / "creative_os"
            match = json.loads((run_dir / "skill_match.json").read_text(encoding="utf-8"))
            tree = json.loads((run_dir / "skill_tree.json").read_text(encoding="utf-8"))

            self.assertEqual("skill_tree_v1", tree["version"])
            self.assertEqual("skills/skill_manifest.json", tree["source"])
            self.assertIn("models/hidream_o1_no_unwanted_text_rules", match["loaded_skill_ids"])
            self.assertIn("models/hidream_o1_storyboard_rules", match["loaded_skill_ids"])
            self.assertIn("Do not generate readable text", " ".join(rule for skill in tree["loaded_skills"]["models"] for rule in skill["rules"]))

    def test_stage04_to_08_read_skill_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_phase1(
                Phase1RunConfig(
                    job_id="skill-tree-wiring",
                    topic="calm evergreen morning routine tip",
                    mode="calm_evergreen",
                    style="clean_warm_lifestyle",
                    runs_root=Path(tmp),
                    attempt_images=False,
                )
            )
            run_dir = Path(tmp) / "skill-tree-wiring" / "creative_os"
            strategy = json.loads((run_dir / "creative_strategy.json").read_text(encoding="utf-8"))
            beat = json.loads((run_dir / "beat_hook_plan.json").read_text(encoding="utf-8"))
            judge = json.loads((run_dir / "creative_judge.json").read_text(encoding="utf-8"))
            contracts = json.loads((run_dir / "scene_contracts.json").read_text(encoding="utf-8"))
            prompts = json.loads((run_dir / "prompt_payload_compiled.json").read_text(encoding="utf-8"))

            self.assertEqual("skills loaded", strategy["source"])
            self.assertIn("Open with a quiet observation.", strategy["camera_visual_rules"])
            self.assertEqual("skills loaded", beat["source"])
            self.assertIn("First beat notices a small everyday moment.", beat["active_hook_rules"])
            self.assertEqual("soft observation", beat["selected_beat_plan"]["hook_type"])
            self.assertIn("small everyday observation", beat["selected_beat_plan"]["hook"])
            self.assertIn("Do not generate readable text", " ".join(judge["active_skill_rules"]))
            self.assertIn("Minimal props, no labels", " ".join(contracts[0]["skill_rules_applied"]["style"]))
            self.assertIn("visual_motif", contracts[0])
            self.assertIn("physical_objects", contracts[0]["visual_motif"])
            self.assertIn("ceramic mug", contracts[0]["visual_motif"]["physical_objects"])
            self.assertIn("Do not generate readable text", " ".join(prompts["model_skill_rules"]))
            self.assertIn("Do not create poster", " ".join(prompts["model_skill_rules"]))
            self.assertNotIn("Motion prompts should", " ".join(prompts["model_skill_rules"]))
            self.assertNotIn("Voice direction", " ".join(prompts["model_skill_rules"]))

    def test_hidream_text_rules_land_in_prompt_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_phase1(
                Phase1RunConfig(
                    job_id="skill-tree-hidream",
                    topic="calm evergreen morning routine tip",
                    mode="calm_evergreen",
                    style="clean_warm_lifestyle",
                    runs_root=Path(tmp),
                    attempt_images=False,
                )
            )
            run_dir = Path(tmp) / "skill-tree-hidream" / "creative_os"
            hidream_prompts = json.loads((run_dir / "hidream_prompts.json").read_text(encoding="utf-8"))
            compiled = json.loads((run_dir / "prompt_payload_compiled.json").read_text(encoding="utf-8"))
            positive_prompt = hidream_prompts[0]["positive_prompt"]
            model_prompt = hidream_prompts[0]["model_prompt"]

            self.assertNotIn("calm evergreen morning routine tip", positive_prompt)
            self.assertNotIn("calm evergreen morning routine tip", model_prompt)
            for forbidden in (
                "Use text-sparse",
                "Do not generate readable text",
                "Add negative prompt terms",
                "Voice direction",
                "Motion prompts should",
            ):
                self.assertNotIn(forbidden, positive_prompt)
                self.assertNotIn(forbidden, model_prompt)
            self.assertIn("subject:", positive_prompt)
            self.assertIn("environment:", positive_prompt)
            self.assertIn("physical objects:", positive_prompt)
            self.assertIn("action:", positive_prompt)
            self.assertIn("camera:", positive_prompt)
            self.assertIn("lighting:", positive_prompt)
            self.assertIn("style:", positive_prompt)
            self.assertIn("mood:", positive_prompt)
            self.assertIn("composition:", positive_prompt)
            self.assertTrue(any(term in positive_prompt for term in ("mug", "kitchen", "window", "plant")))
            for term in (
                "readable text",
                "fake text",
                "typography",
                "letters",
                "numbers",
                "logos",
                "labels",
                "UI",
                "screens",
                "captions",
                "subtitles",
                "watermarks",
                "documents",
                "charts",
                "paper labels",
                "distorted letters",
                "poster",
                "title card",
                "typography layout",
                "headline",
                "social media graphic",
            ):
                self.assertIn(term, hidream_prompts[0]["negative_prompt"])
            self.assertIn("hidream_text_artifact_rules", hidream_prompts[0])
            self.assertIn("Use text-sparse", " ".join(hidream_prompts[0]["hidream_text_artifact_rules"]))
            self.assertIn("poster", " ".join(hidream_prompts[0]["hidream_text_artifact_rules"]))
            self.assertIn("hidream_o1_dev_prompt_rules", " ".join(compiled["skill_source"]["loaded"]["models"]))

    def test_real_run_has_no_stale_stage04_to_08_demo_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_phase1(
                Phase1RunConfig(
                    job_id="skill-tree-no-stale-demo",
                    topic="calm evergreen morning routine tip",
                    mode="calm_evergreen",
                    style="clean_warm_lifestyle",
                    runs_root=Path(tmp),
                    attempt_images=False,
                )
            )
            run_dir = Path(tmp) / "skill-tree-no-stale-demo" / "creative_os"
            payload = "\n".join(
                (run_dir / name).read_text(encoding="utf-8")
                for name in (
                    "creative_strategy.json",
                    "beat_hook_plan.json",
                    "creative_judge.json",
                    "scene_contracts.json",
                    "prompt_payload_compiled.json",
                    "hidream_prompts.json",
                )
            )
            self.assertNotIn("jungle sunrise progression", payload)
            self.assertNotIn("misty jungle canopy", payload)
            self.assertNotIn("curiosity reveal", payload)

    def test_real_run_has_no_fake_skill_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_phase1(
                Phase1RunConfig(
                    job_id="skill-tree-real-run",
                    topic="calm evergreen morning routine tip",
                    mode="calm_evergreen",
                    style="clean_warm_lifestyle",
                    runs_root=Path(tmp),
                    attempt_images=False,
                )
            )
            run_dir = Path(tmp) / "skill-tree-real-run" / "creative_os"
            tree_text = (run_dir / "skill_tree.json").read_text(encoding="utf-8")
            match = json.loads((run_dir / "skill_match.json").read_text(encoding="utf-8"))

            self.assertNotIn("fake_skill", tree_text.lower())
            self.assertNotIn("fake skill", tree_text.lower())
            self.assertNotIn("demo", tree_text.lower())
            self.assertIn("modes/calm_evergreen", match["loaded_skill_ids"])


if __name__ == "__main__":
    unittest.main()
