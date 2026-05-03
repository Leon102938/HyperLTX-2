# HyperLTX / Content Maschine System Architecture Map

This map is a read-only operator guide for debugging quality problems in the current workspace. It describes the pipeline as it exists in code, not as a proposed rewrite.

Primary run directory: `/workspace/agent_runs/<job_id>`.

Backend job directories are usually under `/workspace/jobs/...` and expose their own `job.log`, status JSON, request JSON, and result/output files.

## 1. High-level human schaltplan

Think of the Content Maschine as a production line:

1. **Input desk**: `scripts/agent_core_cli.py` turns human CLI flags into an API payload.
2. **Job counter**: FastAPI accepts the job and hands it to `agent_core.agent.VideoAgent`.
3. **Factory manager**: `VideoAgent.run_job()` creates the run folder, saves `input_job.json`, and moves the job through planning, generation, review, selection, assembly, and final verdict.
4. **Planning office**: `agent_core/planner.py` decides duration, resolution, scenes, variations, takes, storyboard candidates, render modes, retries, and step contracts.
5. **Creative director**: `agent_core/director.py` creates the creative brief, style lock, prompt guidance, scene intents, and variation directives. It may use the local Director LLM, otherwise rule-based fallback.
6. **Creative operating system**: `agent_core/creative_system/*` provides modes, styles, shot recipes, anti-patterns, and prompt rules.
7. **Prompt compiler**: `agent_core/prompt_builder.py` converts intent into debug prompts, scene world contracts, positive model prompts, negative prompt terms, and backend-specific prompt strings.
8. **Storyboard bay**: `agent_core/adapters/zimage_storyboard_adapter.py` sends positive-only keyframe prompts to Z-Image via FastAPI.
9. **Video bay**: `agent_core/adapters/ltx2_adapter.py` sends LTX prompt payloads and overrides to LTX via FastAPI.
10. **Inspection station**: `agent_core/utils.py` validates images/videos, extracts review frames, evaluates keyframe risk, runs heuristic or Qwen3-VL review, and computes final quality verdicts.
11. **Selection station**: `VideoAgent._select_keyframe_candidate()` and `_select_take_record()` choose the safest keyframe/take using technical validation, visual review, postability, and creative rules.
12. **Assembly station**: `agent_core/assembler.py` concatenates scene takes when needed, muxes voice/music/subtitles/overlay, writes `final.mp4`, and attaches the final verdict.
13. **Shipping shelf**: `result.json`, `state.json`, `takes.json`, `storyboard_plan.json`, `model_prompts.json`, `prompt_audit.json`, `logs/agent.log`, and `final.mp4` are the main evidence for debugging.

## Full Flow

### User CLI Input -> API Job

Responsible files/classes/functions:

- `scripts/agent_core_cli.py`
- `_build_parser()`, `_http_json()`, `main()`, inspect helpers such as `load_json_safe()`
- FastAPI routing is in `app/agent_core_api.py`

Input data:

- CLI flags: `--idea`, `--script`, `--duration-sec`, `--orientation`, `--resolution`, `--use-storyboard`, `--subtitle-mode`, `--vision-review-provider`, etc.
- Optional `--inspect-run` reads an existing run without submitting.

Output data:

- JSON payload submitted to local FastAPI.
- Polling output from `/agent-core/jobs/<job_id>`.

JSON artifacts written:

- The CLI itself does not write run artifacts. `VideoAgent` writes them after the API starts the job.

Where quality can go wrong:

- Wrong `orientation`, `scene_count`, `variations_per_scene`, `takes_per_scene`, `subtitle_mode`, or review provider can make later behavior look wrong even when code is working.
- Short social jobs need metadata and format that trigger social-tip safeguards.

Diagnosis:

- Check `/workspace/agent_runs/<job_id>/input_job.json`.
- Compare CLI intent with `result.json.metadata` and `plan.json.metadata`.

### API Job -> `agent_core/agent.py`

Responsible:

- `agent_core/agent.py`
- `VideoAgent.load_job()`
- `VideoAgent.run_job()`
- `StateStore.initialize()`, `save_plan()`, `save_result()`, `transition()`

Input:

- `JobInput` from `agent_core/schemas.py`.

Output:

- A `ResultSummary`.
- Step transitions in `state.json`.
- Logs in `logs/agent.log`.

JSON artifacts:

- `input_job.json`
- `state.json`
- `plan.json`
- `director_output.json`
- `scene_plan.json`
- `prompt_audit.json`
- `model_prompts.json`
- later `storyboard_plan.json`, `takes.json`, `result.json`

Quality risks:

- A backend can fail softly or hard.
- Plan may be rebuilt after voice duration, changing scenes/prompts.
- If `vision_review_provider` or model dir is wrong, review becomes heuristic or `needs_review`.

Diagnosis:

- Read `state.json.steps`.
- Read `logs/agent.log` in timestamp order.
- Confirm whether planning was updated after voice generation.

### `agent_core/agent.py` -> planner/director

Responsible:

- `agent_core/planner.py`
- `ProductionPlanner.build_plan()`
- `agent_core/director.py`
- `DirectorEngine.build_direction()`

Input:

- `JobInput`
- backend capabilities from `BackendRegistry`
- optional actual voice duration

Output:

- `ProductionPlan`
- `DirectorOutput`
- scene plans, variations, storyboard candidates, takes

Artifacts:

- `plan.json`
- `director_output.json`
- `scene_plan.json`

Quality risks:

- Scene segmentation can be wrong.
- Director may be off-topic or fall back.
- Social-tip visual guard may not trigger if input does not match its conditions.
- Render duration is snapped to LTX frame contract, which can change timing.

Diagnosis:

- `director_output.json.llm` tells whether Director LLM was used or fallback.
- `plan.json.metadata.social_tip_visual_guard` tells whether social protection is active.
- `scene_plan.json.scenes[].scene_intent` shows planned visual role and goal.

### planner/director -> creative_system

Responsible:

- `agent_core/creative_system/loader.py`
- `load_creative_system()`
- `detect_mode_id()`
- `modes/*.yaml`, `styles/*.yaml`, `libraries/*.yaml`, `prompts/*.md`

Input:

- `idea`, `script`, and `job.metadata.mode_id`

Output:

- active mode and style metadata injected into `director_output.metadata`
- playbook values used by `PromptBuilder`

Artifacts:

- `plan.json.metadata.mode_id`
- `plan.json.metadata.style_id`
- `scene_plan.json.scenes[].prompt_build_metadata.scene_world_contract`

Quality risks:

- Wrong mode detection causes wrong motifs.
- Missing or weak shot recipes produce boring scenes.
- Anti-patterns may be documented but not sufficiently enforced by prompts/review.

Diagnosis:

- Check `mode_id`, `style_id`, `motif_id`, `shot_recipe_id`, and `anti_patterns_checked`.

### creative_system -> prompt_builder

Responsible:

- `agent_core/prompt_builder.py`
- `PromptBuilder.build_global_prompt()`
- `build_scene_prompt()`
- `build_scene_world_contract()`
- `build_variation_prompt()`
- `build_storyboard_effective_prompt()`
- `compile_visual_prompt_parts()`

Input:

- `JobInput`
- `DirectorOutput`
- `SceneIntent`
- style lock and mode playbook

Output:

- debug prompt for audit
- scene world contract
- positive/negative/model/backend prompts
- prompt audit metadata

Artifacts:

- `scene_plan.json.scenes[].prompt_text`
- `scene_plan.json.scenes[].prompt_build_metadata`
- `prompt_audit.json`
- `model_prompts.json`

Quality risks:

- Debug labels or script text can leak into backend prompts.
- Positive prompts can accidentally include risk terms: `phone`, `screen`, `ui`, `website`, `social`, `content`, `letters`, `logo`, etc.
- Negative prompt spam can steer models toward forbidden concepts.
- Variation prompts can weaken the scene contract.

Diagnosis:

- Start with `model_prompts.json.checks`.
- Inspect `positive_model_prompt`, `negative_model_prompt`, `zimage_prompt_sent`, `ltx_prompt_sent`.
- Debug prompts may contain labels. Backend prompts must not.

### prompt_builder -> Z-Image storyboard/keyframes

Responsible:

- `agent_core/planner.py`
- `ProductionPlanner.build_storyboard_render_plan()`
- `agent_core/adapters/zimage_storyboard_adapter.py`
- `ZImageStoryboardAdapter.generate_storyboard()`

Input:

- keyframe candidate plan
- `effective_model_prompt`
- `positive_model_prompt`
- width/height/seed/steps/guidance

Output:

- storyboard image
- candidate validation and risk review
- selected keyframe per scene

Artifacts:

- `storyboard_plan.json`
- per-candidate images under scene storyboard folders
- Z-Image backend request/status/result files under `/workspace/jobs/...`

Quality risks:

- Z-Image receiving negative/debug text can create visible text or UI-like artifacts.
- Candidate may be technically valid but visually risky.
- No selected keyframe means video may fall back to text-only/reference behavior.

Diagnosis:

- `storyboard_plan.json.scene_storyboards[].generated_candidates[]`
- Check `visual_risk_review`, `validation`, `effective_prompt`, `effective_model_prompt`, `prompt_source`.

### Z-Image storyboard/keyframes -> LTX video/takes

Responsible:

- `agent_core/agent.py`
- `_run_video_step()`, `_build_take_job()`, `_validate_take_output()`
- `agent_core/planner.py`
- `build_take_render_plan()`
- `agent_core/adapters/ltx2_adapter.py`

Input:

- selected keyframe if any
- take-specific prompt metadata
- LTX prompt payload and overrides
- width/height/frame count/frame rate/seed

Output:

- one or more video takes per scene
- technical validation
- review frames
- take visual review

Artifacts:

- `takes.json`
- scene take videos under `/workspace/agent_runs/<job_id>/scenes/<scene_id>/takes/`
- review frames under each take workspace
- backend LTX job logs under `/workspace/jobs/.../job.log`

Quality risks:

- LTX may ignore clean prompts and hallucinate text/devices.
- Bad keyframe can poison the video.
- Duration/frame contract mismatch causes validation rejection.
- If no negative prompt channel exists, the current code embeds short avoid terms into `ltx_prompt_sent`.

Diagnosis:

- `takes.json.scene_outputs[].takes[].metadata.backend_metadata`
- `validation.issues`
- `take_visual_review`
- backend `job.log`

### LTX video/takes -> Qwen3-VL review

Responsible:

- `agent_core/utils.py`
- `extract_review_frames()`
- `evaluate_take_visual_review()`
- `_evaluate_take_visual_review_heuristic()`
- `_evaluate_take_visual_review_qwen3_vl()`
- `scripts/qwen3_vl_review_subprocess.py`
- `agent_core/creative_system/prompts/qwen3_vl_reviewer_system.md`

Input:

- extracted frames
- scene world contract
- prompt text and variation prompt
- selected keyframe visual risk
- `VISION_REVIEW_*` env or job metadata

Output:

- `take_visual_review_status`
- `postability_score`
- issues, warnings, problem frames

Artifacts:

- Review results embedded in `takes.json`.
- Final frame review embedded in `result.json.metadata.final_quality_verdict`.

Quality risks:

- Missing model dir causes Qwen3-VL fallback to `needs_review`.
- Non-JSON VLM output becomes parser warnings.
- Heuristic review does not actually see image content.
- Review prompt may be too strict or too lenient.

Diagnosis:

- `takes.json...take_visual_review.provider`
- `real_vlm_inference_used`
- warnings containing `qwen3_vl`
- frame paths in `review_frames`

### Qwen3-VL review -> take selection

Responsible:

- `agent_core/agent.py`
- `_select_keyframe_candidate()`
- `_select_take_record()`
- `_compute_technical_score()`
- `_compute_creative_score()`

Input:

- valid candidates/takes
- visual review status and score
- scene position and shot type
- duration delta and retry state

Output:

- `selected_keyframe`
- `selected_take`
- `selected_scene_outputs`

Artifacts:

- `storyboard_plan.json.selected_scene_storyboards`
- `takes.json.selected_scene_outputs`
- `takes.json.scene_outputs[].selection`

Quality risks:

- A visually better take can lose if review status/score is worse.
- Creative scoring may favor a wrong shot type.
- If all valid takes are rejected visually, a rejected take can be selected as last resort.

Diagnosis:

- Inspect `selection.visual_candidates`, `scored_candidates`, `technical_candidates`, `selected_by_rule`, `selection_reason`.

### take selection -> assembler

Responsible:

- `agent_core/assembler.py`
- `ResultAssembler.assemble()`
- `concat_video_segments()`
- `mux_voice_into_video()`
- `assemble_final_video()`
- `write_srt_subtitles()`

Input:

- selected scene outputs
- voice/music/storyboard/video results
- subtitle and overlay metadata

Output:

- `assembled_video.mp4` for multi-scene jobs
- `captions.srt` if sidecar/burn subtitles enabled
- `overlay_title.txt` if overlay enabled
- `final.mp4`

Artifacts:

- media files in run dir
- `result.json`
- `state.json.artifacts`

Quality risks:

- Burned subtitles and overlay text intentionally add visible text.
- Audio/video duration mismatch can create trims/padding.
- Assembly can fail if selected takes are invalid.

Diagnosis:

- `result.json.metadata.assembly`
- `state.json.artifacts`
- final file path and duration fields.

### assembler -> final quality verdict -> output files

Responsible:

- `agent_core/utils.py`
- `evaluate_final_quality_verdict()`
- `validate_video_take()`
- `extract_review_frames()`
- optional final Qwen3-VL review

Input:

- `final.mp4`
- selected scene outputs
- selected storyboards
- assembly metadata
- voice/music metadata

Output:

- `final_quality_verdict`
- final review frames

Artifacts:

- `result.json.metadata.final_quality_verdict`
- final review frames under `/workspace/agent_runs/<job_id>/final_review_frames`

Quality risks:

- Verdict can be `needs_review` because review is heuristic-only.
- Burned subtitles lower score even when intentional.
- Final review may catch problems not caught by take review.

Diagnosis:

- `result.json.metadata.final_quality_verdict.final_quality_status`
- `main_issues`, `warnings`, `problem_scenes`, `quality_sources`

## 2. Technical Module Map

### `scripts/agent_core_cli.py`

Purpose: Human CLI for submitting jobs, polling jobs, inspecting existing runs, and summarizing errors.

Important functions/classes:

- `_build_parser()`
- `_http_json()`
- `load_json_safe()`
- run inspection and log-tail helpers

Reads:

- CLI args
- API responses
- existing run files when `--inspect-run` is used

Writes:

- No core run artifacts directly.

Quality problems it can cause:

- Bad metadata or flags.
- Misleading diagnosis if inspecting stale run IDs.

Change when:

- You need better operator visibility, safer payload defaults, or diagnosis summaries.

Do not change when:

- The model output is bad but payload and artifacts show planning/render causes.

### `agent_core/agent.py`

Purpose: Main orchestrator for lifecycle, planning, storyboard, video, validation, review, selection, and result saving.

Important:

- `VideoAgent.run_job()`
- `_run_storyboard_step()`
- `_run_video_step()`
- `_save_prompt_audit()`
- `_save_model_prompts_trace()`
- `_select_keyframe_candidate()`
- `_select_take_record()`

Reads:

- `JobInput`
- `ProductionPlan`
- backend results
- review frames

Writes:

- Most run artifacts through `StateStore`
- `prompt_audit.json`
- `model_prompts.json`
- embedded take/storyboard metadata

Quality problems:

- Wrong selection logic.
- Review wiring wrong.
- Prompt trace incomplete.
- Retry budget or fallback behavior wrong.

Change when:

- Quality evidence exists but selection/review/orchestration chooses badly.

Do not change when:

- The prompt itself, creative strategy, backend model, or assembly is the real cause.

### `agent_core/planner.py`

Purpose: Converts a job into a complete production plan.

Important:

- `ProductionPlanner.build_plan()`
- `_build_scene_plans()`
- `_build_variation_plans()`
- `build_storyboard_render_plan()`
- `build_take_render_plan()`
- social-tip guard helpers

Reads:

- job fields and metadata
- backend capabilities
- creative system playbooks
- director output

Writes:

- `ProductionPlan`
- scene/take/storyboard candidate structures later serialized into `plan.json` and `scene_plan.json`

Quality problems:

- Wrong scene count, motifs, render mode, storyboard enablement, duration, variation count, or social guard activation.

Change when:

- The plan shape is wrong before rendering starts.

Do not change when:

- Plan is good but backend rendering or review fails.

### `agent_core/director.py`

Purpose: Builds creative direction, either with local Director LLM or rule-based fallback.

Important:

- `DirectorEngine.build_direction()`
- `_build_rule_based_output()`
- `_normalize_llm_payload()`
- `_coerce_scene_map_payload()`
- `_build_scene_intent()`

Reads:

- job idea/script
- scene beats
- style memory
- optional LLM response

Writes:

- `DirectorOutput`, saved in `director_output.json`

Quality problems:

- Off-topic scenes, weak intent, wrong hook, wrong style lock, bad variation directives.

Change when:

- `director_output.json` is already wrong before PromptBuilder compiles prompts.

Do not change when:

- Director output is good but prompts/backend outputs are bad.

### `agent_core/prompt_builder.py`

Purpose: Converts direction into backend-safe visual prompts and auditable prompt metadata.

Important:

- `PromptBuilder.BUILDER_VERSION`
- `build_scene_world_contract()`
- `build_scene_prompt()`
- `build_variation_prompt()`
- `build_storyboard_effective_prompt()`
- `compile_visual_prompt_parts()`
- `_build_positive_model_prompt()`
- `_build_negative_model_terms()`

Reads:

- job text
- scene intent
- style lock
- creative mode/style metadata

Writes:

- prompt strings and metadata inside scene/take/storyboard plans

Quality problems:

- Debug label leakage.
- Positive risky terms.
- Overlong prompts.
- Weak negative terms for LTX.
- Too much negative text sent to Z-Image.

Change when:

- `model_prompts.json` or `prompt_audit.json` shows bad prompt compilation.

Do not change when:

- Clean prompts still render badly due to backend/model limits.

### `agent_core/creative_system/*`

Purpose: Local creative playbooks and prompt instructions.

Important:

- `loader.py`
- `modes/morning_reset.yaml`
- `styles/clean_lifestyle_morning.yaml`
- `libraries/hook_patterns.yaml`
- `libraries/shot_recipes.yaml`
- `libraries/anti_patterns.yaml`
- `prompts/director_system.md`
- `prompts/qwen3_vl_reviewer_system.md`

Reads:

- static JSON-like YAML and Markdown.

Writes:

- Nothing directly. Values flow into planner/director/prompt metadata.

Quality problems:

- Weak motif library.
- Missing anti-patterns.
- No mode for a job type.
- Reviewer instructions too vague.

Change when:

- The strategy is wrong, boring, or too generic.

Do not change when:

- Runtime plumbing or backend failures are the issue.

### `agent_core/adapters/zimage_storyboard_adapter.py`

Purpose: HTTP adapter for Z-Image storyboard generation.

Important:

- `ZImageStoryboardAdapter.capabilities()`
- `generate_storyboard()`
- `_resolve_effective_prompt()`

Reads:

- storyboard step params
- FastAPI Z-Image readiness/status/result endpoints

Writes:

- `ExecutionResult`
- backend Z-Image jobs write their own files

Quality problems:

- Wrong prompt source.
- Positive-only policy not applied.
- Candidate image generated at wrong size/seed.

Change when:

- The adapter sends the wrong prompt or cannot interpret backend status/result.

Do not change when:

- The prompt strategy itself is bad.

### `agent_core/adapters/ltx2_adapter.py`

Purpose: HTTP adapter for LTX video generation.

Important:

- `LTX2Adapter.capabilities()`
- `generate_video()`
- `_has_explicit_image_override()`

Reads:

- video step params
- selected keyframe usage
- backend overrides
- FastAPI LTX endpoints

Writes:

- `ExecutionResult`
- backend LTX jobs write video and `job.log`

Quality problems:

- Wrong `prompt` sent to LTX.
- Wrong `images` conditioning payload.
- Duration/frame overrides wrong.
- Backend result metadata not captured.

Change when:

- LTX request construction or response handling is wrong.

Do not change when:

- Prompt/creative system is wrong or model simply hallucinates.

### `agent_core/utils.py`

Purpose: Shared utilities for JSON, media probing, validation, review, prompt compression, Qwen3-VL subprocess, subtitles, and final verdict.

Important:

- `write_json()`, `read_json()`
- `validate_image_candidate()`
- `validate_video_take()`
- `extract_review_frames()`
- `evaluate_keyframe_visual_risk()`
- `evaluate_take_visual_review()`
- `_evaluate_take_visual_review_qwen3_vl()`
- `evaluate_final_quality_verdict()`
- `assemble_final_video()`

Reads:

- media files
- review frames
- scene contracts
- env vars for Qwen3-VL

Writes:

- JSON via callers
- review frames
- subtitles and media outputs via assembler helpers

Quality problems:

- Bad validation thresholds.
- Heuristic review blind spots.
- Qwen parser failures.
- Final verdict too strict or too loose.

Change when:

- Review/validation evidence is wrong compared with actual frames.

Do not change when:

- Planning or prompting created the wrong content.

### `agent_core/assembler.py`

Purpose: Builds final deliverables from selected takes and audio/music/subtitle/overlay choices.

Important:

- `ResultAssembler.assemble()`
- `failure()`
- `_assert_selected_scene_outputs_are_valid()`

Reads:

- selected scene outputs
- voice/music/storyboard/video results
- plan metadata

Writes:

- `assembled_video.mp4`
- `captions.srt`
- `overlay_title.txt`
- `final.mp4`
- final quality metadata in `result.json`

Quality problems:

- Burned subtitles or overlays introduce visible text.
- Bad audio/video timing.
- Multi-scene concat issues.

Change when:

- Selected takes are good but final assembled output is wrong.

Do not change when:

- Bad takes were selected or generated.

### `agent_core/schemas.py`

Purpose: Pydantic contracts for jobs, plans, steps, artifacts, reviews, takes, scene intents, and results.

Important:

- `JobInput`
- `ProductionPlan`, `ScenePlan`, `VariationPlan`, `TakePlan`
- `DirectorOutput`, `SceneIntent`, `StyleLock`, `PromptGuidance`
- `TakeResultRecord`, `KeyframeCandidateResult`, `ResultSummary`, `JobState`

Reads:

- Raw dicts from API/files/internal builders.

Writes:

- Validated model instances serialized by `StateStore`.

Quality problems:

- Missing fields prevent evidence from being persisted.
- Too-loose fields allow unclear contracts.

Change when:

- You need durable new data in artifacts.

Do not change when:

- You only need better creative defaults or prompts.

## 3. Run Artifact Map

### `input_job.json`

Created: at job initialization.

Writer: `StateStore.initialize()`.

Contains: validated `JobInput`: idea, script, duration, format/orientation, style, metadata, backend overrides.

How to read: confirm the system received what you meant to submit.

Reveals: wrong flags, missing storyboard/review metadata, wrong orientation, wrong subtitle mode.

### `director_output.json`

Created: after planning.

Writer: `StateStore.save_director_output()`.

Contains: director mode, LLM status, creative brief, style lock, prompt guidance, scene intents.

How to read: inspect whether the plan was creative and on-topic before prompt compilation.

Reveals: LLM fallback, off-topic director plan, weak style lock, bad scene intent.

### `plan.json`

Created: after `ProductionPlanner.build_plan()`. May be rewritten after real voice duration.

Writer: `StateStore.save_plan()`.

Contains: full `ProductionPlan`: steps, scenes, takes, storyboard candidates, metadata.

How to read: this is the full contract sent into execution.

Reveals: wrong scene count, disabled storyboard, wrong render mode, wrong duration/frame contract.

### `scene_plan.json`

Created: after planning. May be rewritten after voice duration.

Writer: `StateStore.save_scene_plan()`.

Contains: compact scene-focused view: director output, style lock, scenes.

How to read: inspect `scenes[].prompt_build_metadata.scene_world_contract` and `scenes[].variations`.

Reveals: wrong motif, missing scene world contract, prompt metadata problems.

### `storyboard_plan.json`

Created: after storyboard step finishes, even when candidates fail.

Writer: `StateStore.save_storyboard_report()` via `VideoAgent._build_storyboard_report_payload()`.

Contains: scene storyboard configs, keyframe candidates, generated candidates, selected keyframes.

How to read: compare candidate prompts, output paths, validation, visual risk, and selection.

Reveals: bad keyframe prompt, failed Z-Image job, risky selected keyframe, missing storyboard.

### `prompt_audit.json`

Created: immediately after planning.

Writer: `VideoAgent._save_prompt_audit()`.

Contains: scene prompt audit, leaked terms checked, positive/negative checks, shot recipe and anti-pattern checks.

How to read: start with `checks`, then inspect per-scene prompts.

Reveals: debug label leakage, script leakage, risky positive words, missing shot recipes.

### `model_prompts.json`

Created: immediately after planning.

Writer: `VideoAgent._save_model_prompts_trace()`.

Contains: backend-facing prompt trace per scene: positive, negative, combined, `zimage_prompt_sent`, `ltx_prompt_sent`, sources, checks.

How to read: this is the best file for prompt debugging.

Reveals: Z-Image not positive-only, LTX prompt too long, debug/script leakage, risky positive terms.

### `takes.json`

Created: after video step produces scene outputs.

Writer: `StateStore.save_take_report()` via `VideoAgent._build_take_report_payload()`.

Contains: all take records, validation, review frames, Qwen/heuristic review, selection details, selected outputs.

How to read: inspect selected scene first, then rejected alternatives.

Reveals: good take not selected, technical rejection, visual review rejection, backend metadata, frame extraction problems.

### `state.json`

Created: at initialization and rewritten on each transition/step.

Writer: `StateStore.save_state()`.

Contains: current phase, steps, artifacts, errors, notes.

How to read: quick status and artifact index.

Reveals: where the job stopped, backend job IDs, output paths, failure messages.

### `result.json`

Created: after assembly or failure.

Writer: `StateStore.save_result()`.

Contains: `ResultSummary`: success, paths, backend runs, selected scene outputs, assembly metadata, final quality verdict.

How to read: first file for terminal outcome.

Reveals: final status, final quality score, assembly mode, selected outputs, high-level errors.

### `logs/agent.log`

Created: at initialization and appended throughout.

Writer: `StateStore.append_log()`.

Contains: timestamped orchestration events and tracebacks.

How to read: follow chronological execution when JSON is incomplete.

Reveals: failed step, retry scheduling, Director fallback, story/video start/finish events.

### `final.mp4`

Created: during assembly.

Writer: `ResultAssembler.assemble()`.

Contains: final deliverable video.

How to read: inspect visually and compare against selected take paths.

Reveals: assembly artifacts, subtitle/overlay text, timing/audio problems, final model artifacts.

### `/workspace/jobs/.../job.log`

Created: by backend services such as LTX, Qwen TTS, ACE-Step, upscaler jobs.

Writer: backend app modules such as `app/LTX2.py`, `app/qwen_tts.py`, `app/ace_step_1_5.py`.

Contains: backend runtime logs.

How to read: use when `takes.json` says backend failed or output differs from request.

Reveals: model load errors, request/override issues, CUDA/runtime exceptions, missing files.

## 4. Prompt Flow

### Terms

- **user idea**: high-level creative request. It may contain platform words like "social clip" or "TikTok"; those are metadata, not visual objects.
- **user script**: narration/timing intent. It must not be copied into visual model prompts as literal text.
- **director plan**: creative brief, style lock, prompt guidance, scene intents, variation directives.
- **scene world contract**: structured per-scene visual contract: subject, environment, action, allowed props, forbidden props, text policy, motif, shot recipe, backend policy.
- **debug_prompt**: human-readable prompt with labels such as `WORLD / SETTING`. It is for audit and diagnosis.
- **positive_model_prompt**: clean model-facing visual prose with only desired visual content.
- **negative_model_prompt**: separate avoid terms, usually comma-separated.
- **model_prompt / combined_model_prompt**: currently the LTX-style combined prompt, positive plus short `Avoid:` clause.
- **zimage_prompt_sent**: what should go to Z-Image. Current policy: `positive_only`.
- **ltx_prompt_sent**: what should go to LTX. Current policy: positive plus short avoid terms.
- **ltx_negative_prompt_sent**: no separate durable field is currently emitted in the inspected code. Negative terms are stored as `negative_model_prompt`; LTX receives short avoid terms inside `ltx_prompt_sent`.

### What may go where

- `idea` and `script` may influence Director intent, scene beats, narration, and timing.
- Physical visual translations may go into `positive_model_prompt`: curtains, water glass, window, hand action, soft light.
- Avoid/risk concepts should go into `negative_model_prompt`, `forbidden_props`, `text_risk_policy`, and short LTX avoid clauses.
- Debug labels may exist in `debug_prompt`, `scene.prompt_text`, and audit files.

### What must never go where

- `WORLD / SETTING`, `SUBJECT / ACTION`, `FORBIDDEN VISUALS`, `TEXT RISK POLICY`, `MOTIF SAFETY` must not go to image/video model prompts.
- German narration snippets such as `Vorhang auf`, `Stell ein Glas Wasser ab`, and `Atme ruhig am Fenster` must not go to model prompts.
- Platform terms such as `social clip`, `content`, `TikTok`, `website`, `app`, `UI`, `screen`, `phone` should not be positive visual objects unless the user explicitly asks for visible devices.

### Why `WORLD / SETTING` caused problems

Image/video models can render prompt text as visual text when section headers are sent to the backend. Labels like `WORLD / SETTING` are useful for humans but dangerous for generated frames. The system now keeps them in debug/audit prompts while compiling clean model prompts separately.

### Why debug_prompt vs model_prompt exists

`debug_prompt` is for humans to understand planning. `model_prompt` is for backends. Mixing them makes diagnosis easy but output worse. The split lets the run preserve rich planning evidence without handing model-visible labels to Z-Image or LTX.

### Why Z-Image should be `positive_only`

Z-Image keyframes are especially vulnerable to visible text and UI artifacts when given negative clauses full of risky words. The current policy sends only `positive_model_prompt` to Z-Image and keeps negative concepts in audit metadata.

### Why LTX should use positive prompt + short negative prompt if supported

LTX benefits from concise avoid guidance, but long negative prompt spam can still introduce forbidden concepts. Current code uses a short `Avoid:` clause embedded in `ltx_prompt_sent`. A cleaner future architecture would pass `positive_prompt` and `negative_prompt` separately if the backend exposes a stable field.

## 5. Creative Operating System

### `modes/morning_reset.yaml`

Purpose: Mode playbook for a clean morning reset short.

Used by: `detect_mode_id()`, `ProductionPlanner.build_plan()`, `PromptBuilder.build_scene_world_contract()`.

Effect: selects scene arc, motifs, shot recipes, anti-patterns, global forbidden visuals, backend prompt policy.

Improve by: adding stronger scene arcs, more model-robust safe motifs, clearer hard rules, and better quality targets.

Add new modes by: creating a new mode file with `mode_id`, `visual_style`, scene arc, backend prompt policy, anti-patterns, and global forbidden terms; then update detection or pass `metadata.mode_id`.

### `styles/clean_lifestyle_morning.yaml`

Purpose: Defines visual style principles, camera options, texture targets, object count, and positive prompt rules.

Used by: creative system metadata and Director/PromptBuilder style lock.

Effect: controls lighting, object density, camera language, and what visual language is safe.

Improve by: adding style-specific examples and stricter object-count rules.

### `libraries/hook_patterns.yaml`

Purpose: Describes reusable hook mechanics: light reveal, tactile detail, motion-first, quiet payoff.

Used by: mode playbooks and Director instructions.

Effect: keeps scenes from becoming static or generic.

Improve by: adding hook patterns that map to safe physical actions.

### `libraries/shot_recipes.yaml`

Purpose: Concrete shot recipes with visual action, safe props, camera, lighting, prompt seeds, avoid terms, and why they work.

Used by: Morning Reset mode and PromptBuilder motif-specific prompt generation.

Effect: anchors scenes in model-robust visual motifs.

Improve by: adding more tested recipes and avoiding risky props like paper/devices.

### `libraries/anti_patterns.yaml`

Purpose: Names recurring failure modes and fix strategies.

Used by: Director prompt, mode metadata, prompt audit.

Effect: helps planning and diagnosis speak the same language.

Improve by: adding failures observed in real runs and concrete prompt/selection fixes.

### `prompts/director_system.md`

Purpose: Instructions for Director LLM behavior.

Used by: local Director LLM adapter.

Effect: tells Director to use playbooks, avoid script/platform leakage, and output structured scene plans.

Improve by: adding stricter JSON expectations or examples for new modes.

### `prompts/qwen3_vl_reviewer_system.md`

Purpose: System prompt for frame review.

Used by: Qwen3-VL subprocess review.

Effect: tells reviewer to return JSON only and reject visible text/devices/UI.

Improve by: adding calibrated examples and clearer scoring rules.

## 6. Diagnosis Workflow

Inspect in this exact order when a video is bad:

### 1. `result.json`

Question: Did the job finish, and what did the final quality verdict say?

Good: `success=true`, `output_final_path` exists, final verdict `passed` or explainable `needs_review`.

Bad: `success=false`, missing final path, final verdict `failed`, serious `main_issues`.

Improve next: if final failure is assembly or final review, inspect `assembler.py`/`utils.py`; otherwise continue.

### 2. `final_quality_verdict`

Question: Is the final problem technical, visual, review-related, or assembly-related?

Good: high `final_postability_score`, no main issues, quality sources include relevant reviews.

Bad: missing final, rejected selected take, visible text warnings, heuristic-only warnings.

Improve next: review rules in `utils.py`, prompt/selection if problem scenes are named.

### 3. `takes.json`

Question: Were good takes generated, and was the right one selected?

Good: selected take has passed validation, passed/acceptable visual review, good postability, clear selection reason.

Bad: selected take is last resort, better-looking rejected take exists, all takes failed, Qwen parser warnings.

Improve next: `agent.py` selection/review, `prompt_builder.py`, or backend prompt policy.

### 4. `model_prompts.json`

Question: Did clean prompts reach the backends?

Good: Z-Image source is `positive_model_prompt`; no debug/script leaks; LTX prompt <= 140 words.

Bad: debug labels, script snippets, risky positive terms, Z-Image `Avoid:`.

Improve next: `prompt_builder.py` or creative playbooks.

### 5. `storyboard_plan.json`

Question: Did the keyframe help or harm?

Good: selected keyframe passed validation and visual risk review; prompt source is clean.

Bad: selected risky keyframe, missing keyframe, failed candidates, wrong motif.

Improve next: `prompt_builder.py`, Z-Image adapter, storyboard selection, or mode recipes.

### 6. `scene_plan.json`

Question: Was the planned scene itself correct?

Good: motif, shot recipe, scene intent, allowed/forbidden props match the desired output.

Bad: wrong motif, off-topic scene intent, missing social guard, unsafe allowed props.

Improve next: `planner.py`, `director.py`, or creative system files.

### 7. backend `job.log`

Question: Did the backend receive and execute the expected request?

Good: backend logs show normal load/render/save.

Bad: missing model files, CUDA/runtime error, wrong overrides, timeout, request mismatch.

Improve next: backend/runtime only in a separate implementation goal, not during creative prompt work.

### 8. `final.mp4` and frames

Question: What is actually visible?

Good: matches selected take intent, no forbidden visuals, clean timing/audio.

Bad: text, UI, phone, split screen, boring motion, wrong subject, bad transition.

Improve next: map the visible problem to the table below.

## 7. Problem-to-module Table

| Problem | Likely cause | Inspect | Module to change | Likely fix type |
|---|---|---|---|---|
| Clip is boring | Weak hook/action or too-safe static motif | `scene_plan.json`, `director_output.json`, `takes.json.selection.rule_hits` | `creative_system/*`, `director.py`, `planner.py` | Better hook patterns, shot recipes, scene intent scoring |
| Wrong motif | Mode detection or Director scene intent wrong | `plan.json.metadata.mode_id`, `scene_plan.json.scenes[].scene_intent`, `motif_id` | `creative_system/loader.py`, mode YAML, `director.py` | Add mode detection, stronger scene arc |
| Visible text | Prompt leak or model hallucination | `model_prompts.json`, `prompt_audit.json`, `takes.json.take_visual_review`, frames | `prompt_builder.py`, `creative_system/*`, `utils.py` | Remove risky positives, stronger review/negative policy |
| Phone/UI in image | Social/platform words became visual or model drift | `positive_model_prompt`, `allowed_props`, `forbidden_props`, frames | `prompt_builder.py`, mode YAML, shot recipes | Treat platform/device terms as metadata/forbidden |
| Split-screen/collage | Model interprets social format graphically | `scene_world_contract`, `ltx_prompt_sent`, frames | `prompt_builder.py`, mode anti-patterns | Force single full-frame physical scene |
| Good image but bad video | LTX motion/conditioning/prompt issue | `storyboard_plan.json`, `takes.json.backend_metadata`, backend `job.log` | `ltx2_adapter.py`, `prompt_builder.py`, future backend policy | Better LTX prompt/negative handling or conditioning |
| Good take not selected | Selection scoring/review status issue | `takes.json.scene_outputs[].selection`, `scored_candidates` | `agent.py` | Adjust selection ranking or review normalization |
| Qwen review seems wrong | Parser/model prompt/frame issue | `takes.json...take_visual_review`, review frames, warnings | `utils.py`, `qwen3_vl_reviewer_system.md`, subprocess script | Better JSON extraction, reviewer prompt, frame sampling |
| Director plans off-topic | LLM/fallback prompt or mode guidance weak | `director_output.json`, `scene_plan.json` | `director.py`, `director_system.md`, mode YAML | Stronger schema/examples/playbook constraints |
| Voice fits but visuals do not | Script used for timing but visual translation weak | `director_output.json.scene_intents`, `scene_world_contract` | `director.py`, `prompt_builder.py`, creative modes | Better script-to-visual motif translation |
| Prompts look clean but output is bad | Backend/model limitation or insufficient review gate | `model_prompts.json`, backend `job.log`, frames, `takes.json` | creative playbooks, `utils.py`, future provider selector | Add tested motifs, stronger checkpoints, more takes/review |

## 8. OpenMontage-inspired Next Architecture

Do not implement this yet. This is the target design vocabulary.

### Skills

Map each reusable creative/technical capability into a versioned skill: Morning Reset planning, text-risk avoidance, tactile detail shots, Qwen frame review, LTX prompt policy. Skills should declare inputs, outputs, known failure modes, and tests.

### Model skills

Represent model-specific behavior as skills: Z-Image positive-only keyframe generation, LTX video prompt constraints, Qwen3-VL strict JSON review. This avoids pretending all models need the same prompt.

### Platform skills

Represent output contexts: portrait social tip, landscape demo, silent B-roll, voice-led explainer. Platform skills decide subtitle policy, text risk tolerance, object density, and review strictness.

### Stage director skills

Stage director skills choose scene arcs, shot recipes, variation strategy, and approval gates for a mode. Morning Reset would use light reveal -> tactile detail -> breathing payoff.

### `pipeline_defs`

Define pipelines as declarative files: steps, required artifacts, optional steps, provider choices, retry policy, and artifact contracts. Example: `voice -> storyboard -> keyframe_review -> takes -> take_review -> selection -> assembly -> final_verdict`.

### Checkpoints

Make checkpoints explicit stop/evaluate points: after Director, after prompt compile, after storyboard, after take generation, before assembly, after final verdict.

### Approval gates

Allow human or automated approval before expensive/render-risky steps. Example: approve `model_prompts.json` before rendering, approve selected keyframes before LTX, approve selected takes before assembly.

### `decision_log`

Persist why each major decision happened: mode selected, Director fallback, prompt policy, keyframe selected, take selected, final verdict. This should be a first-class artifact, not scattered across JSON files.

### Provider/tool selector

Separate "what the pipeline needs" from "which backend provides it." Selector can choose Z-Image vs another keyframe provider, LTX settings, Qwen vs heuristic review, based on availability, quality history, and job type.

## 9. `/goal` Preparation

### Future goal 1: Skill Layer design

`/goal Design a read-only Skill Layer architecture for the HyperLTX / Content Maschine system. Do not implement code. Create codex/SKILL_LAYER_DESIGN.md explaining skill types, skill file schema, model skills, platform skills, stage director skills, versioning, tests, artifact evidence, and how Morning Reset would be represented. Include migration steps from current creative_system YAML and prompt_builder logic.`

### Future goal 2: pipeline_defs/checkpoints/approval gates

`/goal Implement a small, testable pipeline_defs/checkpoints/approval-gates foundation for Content Maschine without changing backend runtimes or model downloads. Add declarative pipeline definitions, checkpoint artifacts, approval gate states, and tests. Preserve current default behavior unless gates are explicitly enabled.`

### Future goal 3: Morning Reset quality through skills and creative strategy

`/goal Improve Morning Reset output quality through the new skill/creative strategy layer. Focus on stronger motifs, safer prompt contracts, text/device artifact avoidance, keyframe/take review criteria, and tests. Do not change runtime, model downloads, GUI, or backend APIs.`

Single next implementation goal: **implement pipeline_defs/checkpoints/approval gates** after the architecture and skill design are stable, because it gives the owner explicit inspect/approve points before expensive or risky generation steps.
