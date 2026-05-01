# SESSION_SUMMARY.md

## Datum UTC
2026-04-30T21:29:43Z

## Letzter Sicherer Stand
- CLI E2/E2.1/E2.2 Dashboard gesichert.
- Qwen3-VL Review isoliert ueber /workspace/venvs/qwen3-vl-review; Venv nicht archiviert, Ensure-Script archiviert.
- LTX/Gemma Main Runtime bleibt global auf transformers 4.52.4; Qwen3-VL-Venv nutzt transformers 5.7.0.
- Erster Morning-Reset Quality-Fix umgesetzt: Visual Prompt Sanitizer, Safe Motifs, allowed_props Cleanup, Storyboard Prompt Schutz, strengere Device/UI-Risiken.

## Heute Gebaut / Geaendert
- CLI Dashboard: Run Header, Pipeline Labels, Vision Review Status, gruppierte Issues, Success/Failure/Inspect Summaries, smartere Next Actions.
- Qwen3-VL Isolation dokumentiert und per Venv-Import geprueft.
- Morning Reset Prompt-/Motiv-/Review-Qualitaetsfix umgesetzt.

## CLI Dashboard Status
- E2/E2.1/E2.2 gesichert: Dashboard, Inspect, Success/Failure, Pipeline Label Fix, Vision Status, Issue-Gruppierung, Next Actions.

## Qwen3-VL Isolation Status
- Venv: /workspace/venvs/qwen3-vl-review (nicht archiviert).
- Subprocess: /workspace/scripts/qwen3_vl_review_subprocess.py.
- Ensure: /workspace/scripts/ensure_qwen3_vl_review_runtime.sh.

## LTX/Gemma Status
- FastAPI /health ok; LTX backend ltx-2.3.
- Global transformers 4.52.4 fuer LTX/Gemma-Kompatibilitaet.

## Morning Reset Quality-Fix Status
- Visual Prompt Sanitizer, Safe Motif Contract, allowed_props Cleanup, Storyboard Prompt Schutz und strengere Device/UI-Risiken umgesetzt.

## quality-morning-reset-005
- Technisch success=True, final_phase=assembled, Real VLM True.
- Quality needs_review, Score 0.543.
- Probleme: final frame rejected, Scene 3 needs_review, Qwen3-VL non-json/parser warning, burned subtitles visible text.

## quality-morning-reset-006
- Technisch success=True, final_phase=assembled, final.mp4 vorhanden, Director llm_augmented, LTX/Storyboard/Voice/Qwen3-VL aktiv.
- Quality failed, Score 0.277.
- Bekannte offene Probleme: Scene 1 Fake-Text, Scene 2 Phone neben Glas/Device-Risiko, Scene 3 Split-Screen/Collage/Text/UI-Drift, rejected selected Take Bug, Qwen3-VL non-json/parser warning.

## Status Commands
```text
$ date -u
2026-04-30T21:29:43Z
$ git status --short
 M agent_core/planner.py
 M agent_core/prompt_builder.py
 M agent_core/utils.py
 M codex/ACTIVE_PLAN.md
 M codex/CHANGELOG.md
 M codex/HANDOFF.md
 M codex/MEMORY.md
 M codex/PROJECT_STATE.md
 M init.sh
 M scripts/agent_core_cli.py
 M scripts/check_director_llm.py
 M scripts/download_director_model.py
 M scripts/download_qwen3_vl_model.py
 M scripts/ensure_llama_cpp.sh
 M scripts/ensure_qwen3_vl_review_runtime.sh
 M scripts/qwen3_vl_review_subprocess.py
 M scripts/serve_director_llm.sh
 M tests/test_planner_rules.py
 M tests/test_scene_planner.py
 M tests/test_take_visual_review.py
 M tools/llama.cpp/build/bin/llama-cli
 M tools/llama.cpp/build/bin/llama-server
?? .ipynb_checkpoints/
?? ACE-Step-1.5/acestep/third_parts/nano-vllm/nano_vllm.egg-info/
?? ACE-Step-1.5/checkpoints/
?? LTX-2/checkpoints/
?? fastapi.log
?? hyperltx_project_day_archive_20260430_212853/
?? hyperltx_project_day_archive_20260430_212943/
?? jupyter.log
?? venvs/
$ git diff --name-only
agent_core/planner.py
agent_core/prompt_builder.py
agent_core/utils.py
codex/ACTIVE_PLAN.md
codex/CHANGELOG.md
codex/HANDOFF.md
codex/MEMORY.md
codex/PROJECT_STATE.md
init.sh
scripts/agent_core_cli.py
scripts/check_director_llm.py
scripts/download_director_model.py
scripts/download_qwen3_vl_model.py
scripts/ensure_llama_cpp.sh
scripts/ensure_qwen3_vl_review_runtime.sh
scripts/qwen3_vl_review_subprocess.py
scripts/serve_director_llm.sh
tests/test_planner_rules.py
tests/test_scene_planner.py
tests/test_take_visual_review.py
tools/llama.cpp/build/bin/llama-cli
tools/llama.cpp/build/bin/llama-server
$ git diff --stat
 agent_core/planner.py                     |  34 ++++-----
 agent_core/prompt_builder.py              | 134 +++++++++++++++++++++++++++++----
 agent_core/utils.py                       |  39 +++++++++-
 codex/ACTIVE_PLAN.md                      |   9 +++
 codex/CHANGELOG.md                        |  36 +++++++++
 codex/HANDOFF.md                          |  12 ++-
 codex/MEMORY.md                           |  17 +++++
 codex/PROJECT_STATE.md                    |  22 ++++++
 init.sh                                   |   0
 scripts/agent_core_cli.py                 | 699 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++---------------------
 scripts/check_director_llm.py             |   0
 scripts/download_director_model.py        |   0
 scripts/download_qwen3_vl_model.py        |   0
 scripts/ensure_llama_cpp.sh               |   0
 scripts/ensure_qwen3_vl_review_runtime.sh |   0
 scripts/qwen3_vl_review_subprocess.py     |   0
 scripts/serve_director_llm.sh             |   0
 tests/test_planner_rules.py               |  38 ++++++++++
 tests/test_scene_planner.py               |  27 +++++++
 tests/test_take_visual_review.py          |  20 +++++
 tools/llama.cpp/build/bin/llama-cli       | Bin 1546904 -> 1546904 bytes
 tools/llama.cpp/build/bin/llama-server    | Bin 9172552 -> 9172552 bytes
 22 files changed, 965 insertions(+), 122 deletions(-)
$ python3 --version
Python 3.12.13
$ global transformers
global transformers 4.52.4 /usr/local/lib/python3.12/dist-packages/transformers/__init__.py
$ qwen venv transformers/import
qwen venv transformers 5.7.0 /workspace/venvs/qwen3-vl-review/lib/python3.12/site-packages/transformers/__init__.py
qwen3vl import ok
$ FastAPI health
{"status":"ok","init_ready":true,"ltx_backend":"ltx-2.3"}
$ Director models
{"models":[{"name":"Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf","model":"Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf","aliases":[],"tags":[],"object":"model","created":1777584589,"owned_by":"llamacpp","meta":{"vocab_type":2,"n_vocab":248320,"n_ctx_train":262144,"n_embd":2048,"n_params":34660610688,"size":21380459008}}]}
```

## Quick Tests
```text
bash -n /workspace/init.sh: OK
python3 -m py_compile scripts/agent_core_cli.py: OK
python3 -m py_compile scripts/qwen3_vl_review_subprocess.py: OK
bash -n scripts/ensure_qwen3_vl_review_runtime.sh: OK
python3 -m unittest tests/test_planner_rules.py tests/test_scene_planner.py tests/test_storyboard_pipeline.py tests/test_take_visual_review.py tests/test_output_quality_utils.py tests/test_final_quality_verdict.py: 50 tests OK
```

## Inspect Runs
- quality-morning-reset-006: success=True, final_phase=assembled, Quality failed, qwen3_vl parser warning.
- quality-morning-reset-005: success=True, needs_review, qwen3_vl parser warning, burned subtitles warning.
- quality-morning-reset-004: success=True, needs_review, qwen3_vl runtime missing in that older run.

## Real Geaenderte Dateien
```text
agent_core/planner.py
agent_core/prompt_builder.py
agent_core/utils.py
codex/ACTIVE_PLAN.md
codex/CHANGELOG.md
codex/HANDOFF.md
codex/MEMORY.md
codex/PROJECT_STATE.md
init.sh
scripts/agent_core_cli.py
scripts/check_director_llm.py
scripts/download_director_model.py
scripts/download_qwen3_vl_model.py
scripts/ensure_llama_cpp.sh
scripts/ensure_qwen3_vl_review_runtime.sh
scripts/qwen3_vl_review_subprocess.py
scripts/serve_director_llm.sh
tests/test_planner_rules.py
tests/test_scene_planner.py
tests/test_take_visual_review.py
tools/llama.cpp/build/bin/llama-cli
tools/llama.cpp/build/bin/llama-server
```

## Nicht Im Archiv Enthalten
- /workspace/models
- /workspace/venvs inklusive Qwen3-VL-Venv
- /workspace/.cache und HF cache
- *.safetensors, *.gguf, *.incomplete
- LTX-2/checkpoints und grosse Modell-/Runtime-/Job-Output-Ordner
- llama.cpp Build-Binaries

## Restore Anleitung
1. Archiv nach /workspace entpacken.
2. bash /workspace/init.sh
3. bash /workspace/scripts/ensure_qwen3_vl_review_runtime.sh
4. FastAPI/Director pruefen: curl -sS http://127.0.0.1:8000/health und curl -sS http://127.0.0.1:8011/v1/models
5. Dann erst neuer Qualitaetsrun.

## Naechster Schritt Morgen
Rejected-Take-Selection-Bug und Hard Keyframe Visual Gate gegen Text/Phone/Split-Screen fixen.
