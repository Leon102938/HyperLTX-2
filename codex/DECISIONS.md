# DECISIONS.md

## D-001
Datum: 2026-04-11

Entscheidung:
Zuerst wird ein headless Agent-Core gebaut.

Begruendung:
Der interne Kern muss funktionieren, bevor eine neue API-Schicht oder n8n-Anbindung sinnvoll ist.

Auswirkung:
Phase 1 fokussiert auf Job-Schema, Planner, State, Runner und Adapter.

## D-002
Datum: 2026-04-11

Entscheidung:
n8n bleibt spaeter die aeussere Zentrale.

Begruendung:
Das ist explizite Nutzerpraeferenz und Architekturgrenze.

Auswirkung:
Der Core bleibt intern stark, aber nicht die gesamte Orchestrierungsplattform.

## D-003
Datum: 2026-04-11

Entscheidung:
Kein ComfyUI-Zwang.

Begruendung:
Der Pod bringt bereits mehrere lokale Backends mit, die direkt oder ueber Adapter nutzbar sind.

Auswirkung:
Backend-Zugriffe werden abstrahiert, nicht auf ComfyUI fest verdrahtet.

## D-004
Datum: 2026-04-11

Entscheidung:
Markdown-Dateien dienen als dauerhaftes Projektgedaechtnis.

Begruendung:
Der Projektstand soll ueber Sessions hinweg kontrolliert nachvollziehbar bleiben.

Auswirkung:
Projektstatus, Plan, Aufgaben, Entscheidungen und Audit muessen aktiv mitgefuehrt werden.

## D-005
Datum: 2026-04-11

Entscheidung:
`/workspace/codex` ist ab jetzt der kanonische Pfad fuer das Projektgedaechtnis.

Begruendung:
Der Nutzer hat diesen Pfad explizit vorgegeben; vorhanden war nur ein Jupyter-erstellter Legacy-Stand unter `/workspace/Codex`.

Auswirkung:
Neue Memory-Dateien und Aktualisierungen passieren unter `/workspace/codex`; `/workspace/Codex` bleibt vorerst als Legacy-Snapshot bestehen.

## D-006
Datum: 2026-04-11

Entscheidung:
Phase 1 kapselt vorhandene lokale Backends ueber duenne Adapter, statt vendorte Subrepos umzubauen.

Begruendung:
Im Pod existieren bereits lauffaehige Modellpfade, Services und Wrapper. Ein leichter Core ueber diesen Backends bringt schneller einen belastbaren Kern als tiefe Eingriffe in `LTX-2`, `ACE-Step-1.5` oder `Qwen3-TTS`.

Auswirkung:
Der erste Core wird auf Adapter, Planner, State und Runner fokussiert. Vorhandene FastAPI- und Python-Wrapper werden als Integrationsflaechen behandelt.

## D-007
Datum: 2026-04-11

Entscheidung:
Der erste produktive Core-Zugriff auf Qwen TTS und LTX2 erfolgt ueber lokale HTTP-Endpunkte der vorhandenen FastAPI.

Begruendung:
Diese Endpunkte sind bereits verifiziert, entkoppeln den Core von Backend-Details und vermeiden tiefe Eingriffe in bestehende Wrapper oder vendorte Repos.

Auswirkung:
`agent_core` bleibt eine duenne Orchestrierungsschicht. Spaeter kann pro Adapter direkter Python-Zugriff ergaenzt werden, ohne den Agent selbst umzubauen.

## D-008
Datum: 2026-04-11

Entscheidung:
Jeder Core-Job erhaelt einen filesystem-basierten Job-Workspace mit festen Artefaktdateien.

Begruendung:
Phase 1 braucht nachvollziehbaren State, robuste Debugbarkeit und einfache n8n-/API-Vorbereitung ohne sofortige Datenbank oder Queue.

Auswirkung:
Pro Job werden mindestens `input_job.json`, `plan.json`, `state.json`, `result.json` und `logs/agent.log` geschrieben.

## D-009
Datum: 2026-04-11

Entscheidung:
Wenn Voice aktiviert ist, koppelt der Planner die geplante finale Videolaenge an geschaetzte oder echte Voice-Dauer plus 1s Guard-Padding.

Begruendung:
Der erste Vertical Slice soll keine offensichtlich falsche Audio-/Video-Laengenrelation erzeugen. Diese Regel ist klein, real und sofort nuetzlich.

Auswirkung:
Der Plan wird zuerst mit geschaetzter Voice-Laenge gebaut und nach echter TTS-Dauer erneut verfeinert, bevor das Video startet.

## D-010
Datum: 2026-04-11

Entscheidung:
Der stabil verifizierte LTX2-Phase-1-Pfad ist `ti2vid`, nicht `a2vid`.

Begruendung:
Der reale End-to-End-Test zeigte zwei konkrete Probleme bei `a2vid` mit generierter TTS-Audio:
- Aufloesung muss fuer Two-Stage-LTX Vielfache von 64 sein
- selbst mit gueltiger Aufloesung trat ein Audio-/Latent-Shape-Mismatch auf

Auswirkung:
Phase 1 nutzt Qwen-TTS fuer Dauerplanung und Voice-Artefakte, aber rendert das Video stabil ueber `ti2vid`. `a2vid` bleibt spaeter moegliche Erweiterung nach gesonderter Validierung.

## D-011
Datum: 2026-04-11

Entscheidung:
Der quantisierte Planner-Vertrag ist die einzige Quelle fuer die Phase-1-Video-Dauer.

Begruendung:
Ein realer Drift entstand, weil der Planner bereits auf den LTX-Framevertrag quantisierte, der LTX2-Adapter `num_frames` danach aber erneut aus einer gerundeten Dauer ableitete. Dadurch konnte das reale Video laenger werden als geplant.

Auswirkung:
- `plan.target_duration_sec` repraesentiert die quantisierte Zieldauer
- `video.params.num_frames` ist die kanonische Framezahl fuer den Render
- der LTX2-Adapter uebernimmt diese Framezahl direkt
- State und Result dokumentieren geplante sowie reale Dauer explizit

## D-012
Datum: 2026-04-11

Entscheidung:
Phase 2A verbessert den Core zuerst ueber regelbasierte Scene-/Shot-Planung statt ueber neue Backend-Familien.

Begruendung:
Die vorhandenen Qwen- und LTX2-Backends sind bereits produktiv verifiziert. Der groesste kurzfristige Hebel fuer bessere Videoqualitaet liegt daher in strukturierter Segmentierung und saubereren Produktionsprompts, nicht in sofortigem Backend-Ausbau.

Auswirkung:
- `ProductionPlan` enthaelt mehrere Szenen mit jeweils einem ersten renderbaren Shot
- Multi-Segment-Jobs rendern mehrere LTX2-Clips und concateniert diese vor der Finalisierung
- der Single-Flow bleibt explizit als Fallback erhalten

## D-013
Datum: 2026-04-11

Entscheidung:
Phase 2B fuehrt Mehrfach-Takes pro Szene ein und waehlt vorerst den ersten erfolgreichen Take als stabilen Default.

Begruendung:
Der naechste Qualitaetshebel nach der Segmentierung ist nicht sofort komplexe Bewertung, sondern robuster Mehrfach-Render mit klarer Persistenz. `first_successful_take` ist klein, nachvollziehbar und haelt den Vertrag fuer State, Result und Assembly stabil.

Auswirkung:
- jede Szene enthaelt mehrere geplante Takes mit deterministischem Seed
- erfolgreiche Take-Videos werden im Job-Workspace gespiegelt
- `takes.json` dokumentiert alle Take-Kandidaten und den `selected_take` pro Szene
- die finale Assembly arbeitet nur mit den selektierten Takes
- spaetere Score-/Quality-Auswahl kann auf demselben Take-Vertrag aufsetzen
