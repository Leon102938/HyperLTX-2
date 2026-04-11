# TASK_BOARD.md

## TODO
- zweiten produktiven Backend-Pfad waehlen
- Artefakt-Vertrag fuer Folge-Backends schaerfen
- optionalen direkten Python-Zugriff pro Adapter evaluieren
- Integration von ACE-Step oder ZImage als naechsten Vertical Slice entscheiden
- `a2vid` spaeter separat und gezielt wieder evaluieren
- keyframe-gestuetzten Video-Pfad auf Basis des jetzigen Storyboard-Vertrags vorsichtig planen
- spaetere hoehere Inhaltsbewertung erst nach klarer Definition von Hook-/Narrativ- oder Bildqualitaetszielen planen
- Concat-/Assembly-Timing fuer Multi-Segment-Jobs weiter beobachten
- vor dem naechsten grossen Ausbau einen sauberen Commit nur fuer Code, Tests und kanonische Doku schneiden

## IN PROGRESS
- naechsten grossen Schritt nach Phase 3A festlegen

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
- Phase 2C technischer Quality-Guard pro Take implementiert
- `takes.json` und `state.json` um Guard-Status, Validation und Retry-Historie erweitert
- Auswahlregel auf `quality_guarded_best_valid_take` geschaerft
- begrenzte Retry-Regeln pro Szene implementiert
- Assembler auf validierte selektierte Takes geschaerft
- neue Tests fuer Quality-Guard, Auswahl, Retry und Persistenz implementiert und erfolgreich ausgefuehrt
- echter Phase-2C-Lauf gegen reales LTX2 erfolgreich verifiziert
- Phase 2D Shot-/Prompt-Variation-Engine pro Szene implementiert
- `scene_plan.json`, `takes.json` und `state.json` um Varianten und Variantenzuordnung erweitert
- Variationen und Multi-Take-Flow kompatibel gemacht
- neue Tests fuer Variations-Erzeugung, Struktur, Flow-Kompatibilitaet und Persistenz implementiert und erfolgreich ausgefuehrt
- echter Phase-2D-Lauf gegen reales LTX2 erfolgreich verifiziert
- Phase 2E leichte kreative Varianten-/Take-Auswahl ueber dem technischen Vertrag implementiert
- `takes.json`, `state.json` und Result-Metadaten um `technical_score`, `creative_score`, `selection_reason` und `selected_by_rule` erweitert
- neue Tests fuer kreative Auswahlregeln, benachbarte Shot-Diversitaet, Tie-Break und Persistenz implementiert und erfolgreich ausgefuehrt
- echter Phase-2E-Lauf gegen reales LTX2 erfolgreich verifiziert
- Phase 3A optionale Storyboard-/Keyframe-Pipeline implementiert
- produktiver Z-Image-Storyboard-Adapter ueber vorhandene FastAPI-Endpunkte integriert
- `storyboard_plan.json` sowie Keyframe-Kandidaten und selektierte Keyframes im Job-Workspace eingefuehrt
- neue Tests fuer Storyboard-Planung, Persistenz, Fallback und Keyframe-Auswahl implementiert und erfolgreich ausgefuehrt
- echter Phase-3A-Lauf gegen reales Z-Image erfolgreich verifiziert
- `.gitignore` fuer Laufzeit-/Artefaktordner geschaerft
- `HANDOFF.md` fuer die naechste Session angelegt
