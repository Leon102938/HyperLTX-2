# DECISIONS.md

## D-G3-001
Datum: 2026-05-02

Entscheidung:
Stage Role Contracts werden als tracebares Artefakt (`stage_contracts.json`) eingefuehrt, nicht als neuer Executor.

Begruendung:
Der bestehende Agent-Core soll stabil bleiben, waehrend kreative Rollen sauber voneinander getrennt und morgen aktiv in Director/Planner/PromptBuilder eingespeist werden koennen.

Auswirkung:
Runs enthalten `CreativeStrategy`, `BeatPlan`, `VisualDirection`, `ModelPromptPlan` und `ReviewPlan` als JSON-Vertraege in Plan-Metadata und Audit-Artefakten.

## D-G4-001
Datum: 2026-05-02

Entscheidung:
Stop-after wird als kontrollierter Metadata-/Checkpoint-Vertrag umgesetzt; Resume bleibt vorerst ein dokumentierter Contract, kein Executor.

Begruendung:
Ein halber Resume-Executor waere riskant und koennte alte Prompts/Takes mischen. Der sichere Nutzen liegt heute in klaren Stopps vor Backends und nachvollziehbaren Approval-/Reject-Dateien.

Auswirkung:
CLI/Core unterstuetzen `--stop-after scene_plan|model_prompts|storyboard`; `agent_core/resume_contract.py` prueft wiederverwendbare Artefakte und Rejections.

## D-G5-001
Datum: 2026-05-02

Entscheidung:
Creative-Quality-Review startet als metadata-only Heuristik und behauptet keine echte VLM-Sichtpruefung.

Begruendung:
Der Auftrag verbietet Modellladung/Render. Kreative Warnungen sind nuetzlich, duerfen aber nicht als echte Bildanalyse verkauft werden.

Auswirkung:
Take Reviews und Final Quality Verdict koennen boring/static/generic/platform warnings tragen; echte Qwen3-VL-Inferenz bleibt ein vorhandener optionaler Runtime-Pfad und wird in diesem Task nicht gestartet.

## D-G2-001
Datum: 2026-05-02

Entscheidung:
G2 baut Skills als Markdown-basierte, ladbare Wissensvertraege unter `agent_core/creative_system/skills/`.

Begruendung:
Die Content-Maschine braucht nachvollziehbare kreative Produktionsregeln, ohne den bestehenden Executor oder die Backends riskant umzubauen.

Auswirkung:
Pipeline Definitions, Prompt Audit und spaetere Director-/Planner-/PromptBuilder-Phasen koennen Skills referenzieren und tracen.

## D-G2-002
Datum: 2026-05-02

Entscheidung:
`clean_shortform_v1` wird als neue skill-aware Pipeline eingefuehrt, waehrend `simple_video_v1` unveraendert kompatibel bleibt.

Begruendung:
G2 soll Pipeline Modes vorbereiten, aber den stabilen G1/G1.1-Vertrag nicht brechen.

Auswirkung:
Neue Shortform-Flows koennen Skills und Stage Roles deklarieren; bestehende Runs und Tests bleiben auf `simple_video_v1` lauffaehig.

## D-G2-003
Datum: 2026-05-02

Entscheidung:
Morning Reset wird von festen Pflichtszenen zu flexiblen Motivfamilien erweitert.

Begruendung:
Die starre Vorhang/Wasserglas/Fenster-Sequenz half kurzfristig gegen Drift, fuehrte aber zu Wiederholung und Mini-Fix-Denken.

Auswirkung:
`motif_families` und `motif_family_guidance` sind die neuen kreativen Leitplanken; alte Shot Recipes bleiben als kompatible Bausteine erhalten.

## D-G2-004
Datum: 2026-05-02

Entscheidung:
G2 dokumentiert LTX-Negative-Prompt-Trennung tracebar, baut aber keinen Adapter-/Backend-Umbau fuer ein separates `negative_prompt` Feld.

Begruendung:
Der Auftrag verbietet Runtime-/Backend-Umbauten. Ein unvalidierter Adapter-Eingriff waere riskanter als ein sauberer Trace mit TODO.

Auswirkung:
`model_prompts.json` enthaelt `ltx_positive_prompt_sent`, `ltx_negative_prompt_sent` und `ltx_negative_prompt_supported=false`; echte Backend-Trennung ist Folgearbeit.

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

## D-020
Datum: 2026-04-17

Entscheidung:
Phase 5A wird als kleine Director-/Brain-Schicht direkt im bestehenden Planner verankert.

Begruendung:
Die kleinste saubere Eingriffsstelle fuer bessere Regie, Style-Locks und staerkere Prompts liegt vor der vorhandenen Scene-/Variation-/Storyboard-/Take-Planung. Ein grosser Refactor des `agent_core` waere fuer diesen Schritt unnoetig riskant und widerspricht dem Auftrag.

Auswirkung:
- `director.py`, `llm_adapter.py`, `prompt_builder.py` und `style_memory.py` erweitern den Planner
- Agent, Storyboard, Video-Flow und Assembler bleiben kompatibel
- die Director-Ausgabe wird als strukturierter Plan-Kontext statt als separater Subsystem-Umbau persistiert

## D-021
Datum: 2026-04-17

Entscheidung:
Es wird nur ein echter optionaler Adapter fuer lokale OpenAI-kompatible Director-LLMs gebaut; ohne produktiven Dienst greift ein ehrlicher regelbasierter Fallback.

Begruendung:
Im aktuellen Pod ist kein produktiv laufender Director-Textdienst verifiziert. Das vorhandene `gemma-3` im LTX2-Stack ist kein Beweis fuer einen nutzbaren lokalen Director-Endpunkt. Eine Fake-Gemma-4-Integration wuerde die Architektur und den Projektstand falsch darstellen.

Auswirkung:
- Phase 5A liefert sofort bessere Planung und Prompts ueber `rule_based_fallback`
- spaetere echte lokale Director-Modelle koennen ueber denselben Adapter andocken
- kein Modell-Download- oder Bootstrap-Umbau war fuer diesen Schritt noetig

## D-022
Datum: 2026-04-17

Entscheidung:
Der echte lokale Director-LLM-Pfad fuer Phase 5B wird ueber `llama.cpp` + GGUF gebaut, nicht ueber einen schweren neuen Serving-Stack oder einen vollen Safetensor-Download.

Begruendung:
Im Pod waren nur noch rund `33G` frei. Ein voller Gemma-4-Safetensor-/Transformers-Pfad waere damit unnoetig riskant gewesen. `llama.cpp` liefert bereits einen lokalen OpenAI-kompatiblen Endpoint und `Gemma 4 26B-A4B-it` ist als `Q4_K_M`-GGUF gross genug fuer echten Nutzen, aber klein genug fuer den aktuellen Pod.

Auswirkung:
- produktives Binary: `/workspace/tools/llama.cpp/build/bin/llama-server`
- Modellpfad: `/workspace/models/director/gemma-4-26b-a4b-it/gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf`
- der Director bleibt ueber den bestehenden `llm_adapter.py` integriert
- Fallback auf `rule_based_fallback` bleibt erhalten

## D-023
Datum: 2026-04-17

Entscheidung:
Fuer den lokalen `llama.cpp`-Pfad wird kein volles `DirectorOutput`-Schema direkt vom Modell erzwungen; stattdessen liefert Gemma 4 einen kleineren `scene_map`-Vertrag, der danach sauber in den bestehenden Director-Vertrag normalisiert wird.

Begruendung:
Im realen Pod-Lauf war ein direktes Vollschema-Prompting gegen Gemma 4 unzuverlaessig und fuehrte zu ungueltigem oder abgeschnittenem JSON. Die kleinere `scene_map`-Form ist fuer das Modell stabiler und enthaelt trotzdem die kreativen Signale, die der Director-Layer braucht.

Auswirkung:
- `llm_adapter.py` nutzt fuer `gemma4_llama_cpp_local` einen kleineren JSON-Vertrag
- `director.py` normalisiert reale Gemma-4-Szenenausgaben in `creative_brief`, `style_lock`, `scene_intents`, `prompt_guidance` und Variationshinweise
- `assembler.py` spiegelt den aktiven Director-LLM-Pfad zusaetzlich in `result.json`, damit die echte Nutzung nicht nur im Director-Artefakt sichtbar ist
- der bestehende Planner-/Variation-/Storyboard-/Assembler-Vertrag bleibt intakt

## D-024
Datum: 2026-04-18

Entscheidung:
Der produktive Phase-5B-Director wird von der gestern vorbereiteten Gemma-4-Richtung auf `Qwen3.6-35B-A3B` als GGUF `Q4_K_M` umgestellt.

Begruendung:
Der Nutzer hat diese Umstellung explizit priorisiert. Fuer den vorhandenen Pod ist ein lokaler `llama.cpp`-Pfad mit einer praktikablen Qwen-Quantisierung weiterhin der kleinste echte Produktionsweg. Im realen Lauf war ein community-GGUF fuer Qwen3.6-35B-A3B verfuegbar und erfolgreich nutzbar; damit ist ein sauberer lokaler Director-Pfad ohne neue Plattform moeglich.

Auswirkung:
- produktives Profil: `qwen36_llama_cpp_local`
- produktiver Modellpfad: `/workspace/models/director/qwen3.6-35b-a3b/gguf/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`
- `init.sh` bereitet das Modell jetzt idempotent vor und kann den lokalen Director-Serve optional automatisch starten
- der bestehende `scene_map`-Vertrag und der `rule_based_fallback` bleiben erhalten; nur das echte Director-Modell wurde ausgetauscht

## D-025
Datum: 2026-05-02

Entscheidung:
Phase G1 fuehrt deklarative Pipeline-Definitionen, generische Checkpoints und lokale Approval Gates im bestehenden `agent_core` ein, ohne n8n/API/GUI oder Medien-Backends umzubauen.

Begruendung:
Der Core soll als kontrollierbare Produktionsmaschine nachvollziehbar werden. Grosse Schritte brauchen Status, Artefakte und Freigabepunkte, bevor teure oder riskante Backend-Schritte starten. Die kleinste stabile Form ist ein datei-/state-basierter Vertrag im Run-Workspace.

Auswirkung:
- `agent_core/pipeline_defs/simple_video_v1.json` beschreibt den ersten Pipeline-Vertrag
- `CheckpointRecord` und Pipeline-Schemas erweitern den bestehenden Datenvertrag
- `state.json`, `checkpoints.json` und `result.json.metadata.pipeline` zeigen Pipeline- und Gate-Status
- `approval_gates_enabled=true` macht Plan-/Prompt-Gates blocking; lokale Approval-Dateien koennen spaeter von CLI, n8n oder Human Review geschrieben werden
- `pipeline_dry_run=true` erlaubt G1-Tests ohne Render- oder Modellstart
- der bestehende Render-/Assembler-/Backend-Flow bleibt kompatibel und wird nicht durch einen neuen Executor ersetzt

## D-026
Datum: 2026-05-02

Entscheidung:
Phase G1.1 behandelt Approval/Reject in der CLI als kontrollierte lokale Dateientscheidung und baut noch keinen Resume-Executor.

Begruendung:
Die Checkpoints muessen fuer den Operator sofort sichtbar und bedienbar sein, ohne den stabilen Run-/Render-Pfad, die FastAPI, n8n oder Backend-Runtimes umzubauen. Ein echter Resume-Executor braucht einen separaten Vertrag, weil er bestehende Run-Artefakte wiederaufnehmen und Idempotenz sauber definieren muss.

Auswirkung:
- `--inspect-run` und `--inspect-checkpoints` lesen Checkpoints aus `checkpoints.json` oder `state.json`
- `--approve-checkpoint` und `--reject-checkpoint` schreiben lokale `approvals/<checkpoint_id>.json` im Run-Ordner
- vorhandene Approval-Dateien werden nur mit explizitem `--force-approval` ueberschrieben
- die CLI zeigt nach Approval klar, dass Resume vorbereitet ist, aber die eigentliche Executor-Fortsetzung Future Work bleibt
