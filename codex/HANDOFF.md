# HANDOFF.md

## Stand 2026-04-30
- `init.sh` ist klein und stabilisiert: normaler HF-Downloader als Default, Xet aus, minimaler Init-Lock, Qwen3-VL optional.
- LTX/Gemma ist wieder lauffaehig: globale Main-Runtime nutzt `transformers 4.52.4`.
- Qwen3-VL Review ist isoliert: `/workspace/venvs/qwen3-vl-review` mit `transformers 5.7.0` und `kernels 0.13.0`, aufgerufen ueber `/workspace/scripts/qwen3_vl_review_subprocess.py`.
- Die Qwen3-VL-Venv wird nicht archiviert. Sie wird nach Restore mit `/workspace/scripts/ensure_qwen3_vl_review_runtime.sh` neu erstellt.
- Phase E2/E2.1/E2.2 CLI Produktions-Cockpit ist umgesetzt; `scripts/agent_core_cli.py --inspect-run <job_id>` ist der schnelle Diagnosepfad mit Pipeline Labels, Vision-Status, gruppierten Issues und Next Actions.
- Erster Morning-Reset-Quality-Fix ist umgesetzt: Visual Prompt Sanitizer, Safe Morning Reset Motifs, allowed_props Cleanup, Storyboard Prompt Schutz und strengere Device-/UI-Risiken.
- Aktueller echter Kontrollrun: `quality-morning-reset-006`, technisch `success=True`, `final_phase=assembled`, `final.mp4` vorhanden, aber Final Quality `failed`.
- Diagnose `quality-morning-reset-006`: Scene 1 Fake-Text, Scene 2 Smartphone/Phone neben Glas in einem Take-Kontext, Scene 3 Split-Screen/Collage/Text/UI-Drift, Qwen3-VL non-json/parser warning.
- Offener Bug fuer morgen: rejected Take darf nicht selected werden, wenn passed/needs_review existiert; zusaetzlich hartes Keyframe Gate gegen Fake-Text/Phone/Split-Screen und robustere Qwen3-VL JSON-Auswertung.

## Restore Nach Frischem Pod
1. Archiv nach `/workspace` entpacken.
2. `bash /workspace/init.sh`
3. `bash /workspace/scripts/ensure_qwen3_vl_review_runtime.sh`
4. FastAPI/Director pruefen:
   - `curl -sS http://127.0.0.1:8000/health`
   - `curl -sS http://127.0.0.1:8011/v1/models`
5. Schneller Run-Check:
   - `python3 /workspace/scripts/agent_core_cli.py --inspect-run quality-morning-reset-006`
   - `python3 /workspace/scripts/agent_core_cli.py --inspect-run quality-morning-reset-005`

## Naechster Schritt
Rejected-Take-Selection-Bug und Hard Keyframe Visual Gate gegen Text/Phone/Split-Screen fixen. Danach erst neuen Clean-Visual-Run starten.
