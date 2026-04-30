# HANDOFF.md

## Stand 2026-04-30
- `init.sh` ist klein und stabilisiert: normaler HF-Downloader als Default, Xet aus, minimaler Init-Lock, Qwen3-VL optional.
- LTX/Gemma ist wieder lauffaehig: globale Main-Runtime nutzt `transformers 4.52.4`.
- Qwen3-VL Review ist isoliert: `/workspace/venvs/qwen3-vl-review` mit `transformers 5.7.0` und `kernels 0.13.0`, aufgerufen ueber `/workspace/scripts/qwen3_vl_review_subprocess.py`.
- Die Qwen3-VL-Venv wird nicht archiviert. Sie wird nach Restore mit `/workspace/scripts/ensure_qwen3_vl_review_runtime.sh` neu erstellt.
- Phase E CLI Produktions-Cockpit ist umgesetzt; `scripts/agent_core_cli.py --inspect-run <job_id>` ist der schnelle Diagnosepfad.
- Letzter echter Kontrollrun: `quality-morning-reset-003`, `success=True`, `final_phase=assembled`, `final.mp4` vorhanden. Final Quality ist `needs_review`, weil Qwen3-VL echte sichtbare Text-/Papier-/Subtitle-Risiken meldete.

## Restore Nach Frischem Pod
1. Archiv nach `/workspace` entpacken.
2. `bash /workspace/init.sh`
3. `bash /workspace/scripts/ensure_qwen3_vl_review_runtime.sh`
4. FastAPI/Director pruefen:
   - `curl -sS http://127.0.0.1:8000/health`
   - `curl -sS http://127.0.0.1:8011/v1/models`
5. Schneller Run-Check:
   - `python3 /workspace/scripts/agent_core_cli.py --inspect-run quality-morning-reset-003`

## Naechster Schritt
Echte Qualitaetsanalyse von `quality-morning-reset-003` und gezielter Motiv-/Prompt-Feinschliff. Keine weitere Setup-Arbeit als naechsten Schritt.
