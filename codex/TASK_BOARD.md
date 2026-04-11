# TASK_BOARD.md

## TODO
- zweiten produktiven Backend-Pfad waehlen
- Artefakt-Vertrag fuer Folge-Backends schaerfen
- optionalen direkten Python-Zugriff pro Adapter evaluieren
- Integration von ACE-Step oder ZImage als naechsten Vertical Slice entscheiden
- `a2vid` spaeter separat und gezielt wieder evaluieren
- leichte Quality-/Selection-Regeln ueber `first_successful_take` hinaus definieren
- Concat-/Assembly-Timing fuer Multi-Segment-Jobs weiter beobachten
- vor dem naechsten grossen Ausbau einen sauberen Commit nur fuer Code, Tests und kanonische Doku schneiden

## IN PROGRESS
- naechsten grossen Schritt nach Phase 2B festlegen

## BLOCKED
- noch leer

## DONE
- Bootstrap- und Recon-Analyse durchgefuehrt
- RunPod-Umgebung verifiziert
- vorhandene lokale Modelle, Services und Wrapper dokumentiert
- Projektgedaechtnis auf `/workspace/codex` kanonisiert
- erste technische Bestandsaufnahme erstellt
- `agent_core/` als neuer Phase-1-Kern gebaut
- produktiver Minimal-Vertical-Slice `text/script -> optional qwen_tts -> ltx2 -> Resultat` implementiert
- filesystem-basierter Job-State-Store implementiert
- Planner-Regeln fuer Voice-Laenge implementiert
- Smoke- und Planner-Tests implementiert und erfolgreich ausgefuehrt
- echter End-to-End-Core-Lauf gegen reale Qwen-TTS- und LTX2-Backends erfolgreich verifiziert
- reale Backend-Fixes fuer Aufloesung, Framezahl und Phase-1-Pipeline-Vertrag umgesetzt
- Assembler auf finales MP4 mit gemuxter Voice erweitert
- sauberer Fallback implementiert: ohne nutzbares Voice-Artefakt wird trotzdem `final.mp4` erzeugt
- echter End-to-End-Mux-Lauf gegen reale Qwen-TTS- und LTX2-Backends erfolgreich verifiziert
- Dauervertrag zwischen Planner, LTX2-Frame-Quantisierung, realem Video und finalem Muxing geschaerft
- reale Randfaelle fuer Voice kuerzer, Voice laenger und kein Voice-Artefakt verifiziert
- Phase 2A Scene-/Shot-Planung implementiert
- `scene_plan.json` als neues Plan-Artefakt eingefuehrt
- produktiver Multi-Segment-Flow mit Single-Flow-Fallback umgesetzt
- echter Multi-Segment-Lauf gegen reales LTX2 erfolgreich verifiziert
- Phase 2B Mehrfach-Takes pro Szene implementiert
- `takes.json` als neues Take-Artefakt eingefuehrt
- Auswahlregel `first_successful_take` implementiert
- finale Assembly auf selektierte Takes umgestellt
- echter Multi-Take-Lauf gegen reales LTX2 erfolgreich verifiziert
- `.gitignore` fuer Laufzeit-/Artefaktordner geschaerft
- `HANDOFF.md` fuer die naechste Session angelegt
