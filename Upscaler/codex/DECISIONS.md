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

## D-014
Datum: 2026-04-11

Entscheidung:
Phase 2C validiert jeden erfolgreichen Take technisch vor der Selektion und erlaubt nur ein kleines begrenztes Retry-Budget pro Szene.

Begruendung:
Der naechste robuste Schritt nach Mehrfach-Takes ist keine AI-Inhaltsbewertung, sondern ein klarer technischer Guard gegen leere, triviale, falsch formatierte oder korrupt wirkende Medienartefakte. So bleibt der Produktionsvertrag klein, nachvollziehbar und backendnah.

Auswirkung:
- jeder Take dokumentiert `review_status` und `validation`
- die Standardauswahl ist jetzt `quality_guarded_best_valid_take`
- `first_successful_take` bleibt nur noch als Tie-Break/Fallback fuer technisch gleichwertige valide Kandidaten
- technisch abgelehnte Takes koennen pro Szene begrenzt neu gerendert werden
- der Assembler akzeptiert nur noch validierte selektierte Takes

## D-015
Datum: 2026-04-11

Entscheidung:
Phase 2D fuehrt eine kleine regelbasierte Shot-/Prompt-Variation-Engine pro Szene ein, statt sofort in AI-Planung oder Inhaltsbewertung zu springen.

Begruendung:
Der naechste sinnvolle Qualitaetshebel nach technischem Guard und Mehrfach-Takes ist kontrollierte kreative Kandidatenvielfalt. Ein regelbasierter Variantenvertrag ist klein, testbar und kompatibel mit dem bestehenden Guard-/Retry-/Selection-Flow.

Auswirkung:
- jede Szene kann mehrere `variations` mit eigenem Shot- und Prompt-Profil enthalten
- pro Variation koennen danach mehrere Takes geplant und gerendert werden
- `scene_plan.json`, `takes.json` und `state.json` dokumentieren Varianten und die ausgewaehlte Variation
- technische Selektion bleibt unveraendert; inhaltliche Variantenbewertung bleibt fuer spaetere Phasen offen

## D-016
Datum: 2026-04-11

Entscheidung:
Phase 2E fuehrt nur eine kleine regelbasierte kreative Auswahl ueber technisch validen Kandidaten ein, keine AI-Bewertungsmaschine.

Begruendung:
Nach Phase 2D existieren mehrere kreative Varianten pro Szene, aber ein grosser Bewertungsapparat waere fuer den aktuellen Core zu schwer, schwerer testbar und wuerde den stabilen Produktionsvertrag unnoetig aufweichen. Ein kleiner heuristischer Layer bringt bereits echten Mehrwert bei geringem Risiko.

Auswirkung:
- technische Validitaet bleibt harte Voraussetzung
- kreative Regeln arbeiten nur innerhalb der technisch besten validen Kandidaten
- persistiert werden mindestens `technical_score`, `creative_score`, `selection_reason` und `selected_by_rule`
- direkte Nachbarschaftsabwechslung und grobe Szenenziel-Passung werden jetzt explizit in der Auswahl beruecksichtigt

## D-017
Datum: 2026-04-11

Entscheidung:
Phase 3A nutzt Z-Image als vorhandenen lokalen Storyboard-/Keyframe-Pfad.

Begruendung:
Z-Image existiert bereits im Pod-Stack, haengt an der bestehenden FastAPI, erzeugt echte Bildartefakte und erfordert keinen neuen Backend-Zweig ausserhalb des vorhandenen Systems. Fuer Phase 3A ist das der kleinste produktive Schritt zu visueller Vorsteuerung.

Auswirkung:
- Storyboard bleibt optional und ergaenzend
- der Core erzeugt echte PNG-Keyframes pro Szene bzw. Variation
- Keyframes werden selektiert und sauber persistiert
- der Video-Flow bekommt Storyboard-Kontext, aber noch keinen grossen i2v-Umbau

## D-018
Datum: 2026-04-16

Entscheidung:
Phase 3B nutzt selektierte Storyboard-Keyframes produktiv nur ueber den bereits vorhandenen First-Frame-Image-Conditioning-Pfad des stabilen LTX2-`ti2vid`-Stacks.

Begruendung:
Der vorhandene Pod-Stack und FastAPI-Wrapper koennen reales Image-Conditioning im bestehenden `ti2vid`-Vertrag bereits sauber durchreichen. Ein neuer separater Keyframe-Interpolations- oder Retake-Pfad waere fuer diesen Schritt unnoetig gross und lokal noch nicht als stabiler Produktionsvertrag verifiziert.

Auswirkung:
- `video_mode=keyframe_conditioned` ist jetzt produktiv moeglich, aber nur im bestehenden `ti2vid`-Pfad
- bei fehlendem selektierten Keyframe oder nicht verfuegbarer Storyboard-/Image-Conditioning-Lage faellt der Core ehrlich auf `storyboard_reference` oder `text_only` zurueck
- keine Fake-Freigabe fuer Multi-Keyframe-Interpolation, neuen Backend-Zweig oder grossen Refactor

## D-019
Datum: 2026-04-16

Entscheidung:
Phase 4A nutzt die bereits laufende lokale FastAPI als duenne Worker-/n8n-Bridge fuer den bestehenden `agent_core`.

Begruendung:
Im aktuellen Pod existiert bereits eine produktiv genutzte lokale FastAPI mit bestehenden Medienendpunkten. Ein kleiner zusaetzlicher Router ist daher die kleinste saubere Aussenintegration fuer externe Caller wie n8n. Ein neuer separater API-Stack oder eine neue CLI-Familie wuerde fuer diesen Schritt keinen Mehrwert bringen und den Systemrand unnoetig verbreitern.

Auswirkung:
- externer Einstieg erfolgt jetzt ueber `POST /agent-core/run` und `GET /agent-core/jobs/{job_id}`
- der bestehende `VideoAgent` bleibt die eigentliche Produktionslogik; die Bridge fuegt nur Request-/Response-Vertrag und Referenzen hinzu
- Queue, Auth, Multi-User-Management und grosse API-Plattform bleiben explizit spaetere Schritte
