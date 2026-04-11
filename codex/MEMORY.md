# MEMORY.md

## Dauerhafte Erkenntnisse
- Kanonischer Projekt-Memory-Pfad ist `/workspace/codex`.
- Legacy-Notizen existieren in `/workspace/Codex`; nicht automatisch als Wahrheit bevorzugen.
- Git-Root des eigentlichen Projekts ist `/workspace`, nicht `/workspace/codex`.
- Das Root-Repo ist bereits ein RunPod-Medien-Template mit FastAPI-Wrappern und lokalen Modell-/Tool-Bereichen.
- Basisdienste laufen standardmaessig ueber `start.sh`: Jupyter auf `8888`, FastAPI auf `8000`, Init im Hintergrund.
- Das Root-Python ist nicht die einzige relevante Runtime; wichtige Audio-/ACE-Step-Abhaengigkeiten leben im separaten Venv `/workspace/venvs/qwen3-tts`.
- Vorhandene lokale Medien-Backends sind nutzbar, aber noch nicht als eigener Agent-Core abstrahiert.
- Der Nutzer will zuerst einen starken modularen Agent-Core, nicht sofort API, n8n oder GUI.
- Fremdrepos duerfen spaeter als Referenz dienen, aber nicht blind uebernommen werden.
- Der neue Agent-Core lebt als eigenes Paket `agent_core/` im Git-Root.
- Phase 1 nutzt lokale HTTP-Adapter ueber die bestehende FastAPI, nicht tiefe Backend-Umbauten.
- Der Planner soll Voice-Laenge explizit in die Video-Planung einbeziehen; erst geschaetzt, spaeter mit echter Dauer nachgezogen.
- Jeder Core-Job speichert mindestens `input_job.json`, `plan.json`, `state.json`, `result.json` und `logs/agent.log`.
- Jeder erfolgreiche Phase-1-Job soll jetzt auch `final.mp4` im Job-Workspace haben.
- Phase 2A fuehrt `scene_plan.json` als dauerhaftes Plan-Artefakt pro Job ein.
- Phase 2B fuehrt `takes.json` als dauerhaftes Take-Artefakt pro Job ein.
- LTX Two-Stage braucht in Phase 1 Aufloesungen als Vielfache von `64`.
- LTX Phase-1-Framezahl sollte auf das Schema `8k+1` geschnappt werden.
- Die kanonische Dauerquelle fuer den Video-Render ist jetzt der quantisierte Planner-Vertrag: `plan.target_duration_sec` plus `video.params.num_frames`.
- `a2vid` mit generierter Qwen-TTS-Audio ist im aktuellen Pod-Setup nicht als stabiler Phase-1-Vertrag verifiziert.
- Der reale stabile Phase-1-Renderpfad ist aktuell `ti2vid`; Voice beeinflusst die geplante Videolaenge und wird danach in `final.mp4` gemuxt.
- Phase 2A verbessert Produktionsqualitaet zuerst ueber strukturierte Planung und mehrere Segmente, nicht ueber neue Backends.
- Multi-Segment-Jobs rendern pro Szene eigene Rohclips und fuehren diese vor dem finalen Mux zu `assembled_video.mp4` zusammen.
- Phase 2B fuehrt mehrere Takes pro Szene ein; die aktuelle stabile Auswahlregel ist `first_successful_take`.
- Erfolgreiche Take-Videos werden in den Job-Workspace unter `scenes/<scene_id>/takes/` gespiegelt, auch wenn das Backend seine Originaldateien extern unter `/workspace/jobs` schreibt.
- Die finale Assembly arbeitet ab Phase 2B nur noch mit den selektierten Takes.
- Commit-wuerdig sind primaer `agent_core/`, `tests/`, `examples/`, `.gitignore` und der kanonische Projekt-Memory unter `/workspace/codex`.
- Laufzeit- und Artefaktordner wie `agent_runs/`, `exports/`, `jobs/`, `status/`, `venvs/`, Checkpoints und lokale Pod-Logs sollen nicht Teil eines normalen Code-Commits sein.
- Der Single-Flow bleibt als `single_scene`-Fallback explizit erhalten.
- Rohe LTX2-MP4s koennen bereits einen Audio-Stream enthalten; der Assembler soll fuer Phase 1 trotzdem immer die eigene Voice-Spur bevorzugen.
- Wenn kein nutzbares Voice-Artefakt vorliegt, soll `final.mp4` trotzdem als Kopie des Render-Videos entstehen statt den Job unnoetig scheitern zu lassen.
- Der LTX2-Adapter darf `num_frames` nicht noch einmal aus einer gerundeten Plan-Dauer neu berechnen; sonst driftet der Dauervertrag.
- Fuer verifizierte Phase-1-Runs liegt das reale Delta zwischen Plan und Video nach dem Fix bei etwa `0.001s`.

## Wiederkehrende Stolperfallen
- Pfadkonflikt zwischen `/workspace/Codex` und `/workspace/codex`.
- Paketunterschiede zwischen Root-Python und `qwen3-tts`-Venv.
- Readiness-Flags bedeuten nicht automatisch, dass ein neuer Agent-Core existiert.
- Vorhandene HTTP-Endpunkte koennen fuer Recon hilfreich sein, sollen aber den neuen Core nicht definieren.
- Tests sollten Fake-Adapter nutzen, damit der Core verifiziert werden kann, ohne echte Modelljobs auszufuehren.
- Ein erfolgreicher TTS-Lauf bedeutet nicht automatisch, dass `a2vid` mit derselben Audio-Datei stabil funktioniert.
- Der Randfall `Voice laenger als Video` ist im regulaeren Agent-Pfad absichtlich schwer erreichbar, weil der Planner das inzwischen verhindert; fuer reale Validierung muss er daher gezielt auf Assembler-Ebene mit echten Artefakten provoziert werden.
- Bei Multi-Segment-Jobs kann durch MP4-Concat noch ein kleines Timing-Delta gegenueber der geplanten Szenensumme entstehen; im realen Lauf `real-phase2a-multiscene-1` lag es bei etwa `0.023s`.
- Ein Mehrfach-Take-Run kann alle Takes erfolgreich beenden; die aktuelle Selektion ist dann trotzdem bewusst konservativ und nimmt den ersten erfolgreichen Take statt spaeterer Bewertung.
