# Panel 04 Visual Analysis

## 1. Zweck des Panels

- Das Panel zeigt Creative Strategy als von Director/AI erzeugten Plan.
- Die Pipeline-Stufe ist `04 Creative Strategy`.
- Im Active Workspace verbindet es Input Context, Skill Stack und JSON-Pattern-Preview zu einer freigegebenen Strategie fuer Stage 05.

## 2. Gesamtstruktur

- Workspace-Titel: `ACTIVE WORKSPACE / STAGE 04: CREATIVE STRATEGY`.
- Oben eine Current-Position-Leiste.
- Darunter ein dreispaltiger Hauptbereich: links `A) STRATEGY ENGINE / AI` und darunter `B) INPUT CONTEXT`, mittig `C) SKILL STACK / LOADED COMPONENTS`, rechts `D) STRATEGY BUILD / JSON PATTERN PREVIEW`.
- Unten im Workspace eine breite `E) OUTPUT SUMMARY`.

## 3. Farben

- Cyan fuer Box-Borders und Section-Titel.
- Gruen fuer enabled, geladene Skills und approved Output.
- Gelb/orange fuer `building strategy` und neue Artefakte.
- JSON-Preview nutzt gruenen Text auf dunklem Panel.
- Fallback-Eintrag ist cyan/blau.

## 4. Typografie / Text-Hierarchie

- Stage-Titel cyan uppercase.
- Blockbuchstaben `A)`, `B)`, `C)`, `D)`, `E)` strukturieren die Ansicht.
- Feldlabels sind hellgrau.
- Wichtige Werte sind gruen oder cyan.
- JSON wirkt monospaced und kleiner.
- Output Summary hat grosse Icons und kurze Wertbloecke.

## 5. Panel- und Card-Aufbau

- Links zwei gestapelte Karten.
- Mitte eine Skill-Stack-Karte mit sechs Reihen und einer kleinen Summary-Card innen.
- Rechts eine grosse JSON-Preview-Karte.
- Unten eine breite Output-Summary mit vier Segmenten: Strategy, Narrative direction, Output, Next handoff.

## 6. Inhaltliche Bloecke

- Strategy Engine / AI: Director AI, Strategy model, Image model target, Active reasoning, Status.
- Input Context: Mode, Style package, Story goal, Duration target, Scene count, Audience intent.
- Skill Stack: core/tiktok_shortform, positive_image_prompting, anti_boring, artifact_avoidance, cinematic_nature, fallback.
- JSON Preview: story_arc, hook_style, scene_plan, camera_language, visual_rules, output_ready.
- Output Summary: Strategy approved, narrative direction, creative_strategy.json ready, next handoff Beat/Hook Planner.

## 7. Interaktion / Zustaende

- Strategy wird als aktiv im Bau und gleichzeitig output-ready dargestellt.
- Gelber Spinner/Status fuer laufende Strategie.
- Gelb markiertes Artefakt `creative_strategy.json` im Bottom Artifact Panel.

## 8. Was NICHT passieren darf

- Keine rohe Strategie ohne Input Context.
- Keine JSON-Wand ohne danebenliegende menschliche Zusammenfassung.
- Skill Stack nicht ausblenden.
- Kein Wechsel auf Stage 05, solange Stage 04 nicht sichtbar approved/ready ist.

## 9. Umsetzungshinweise fuer Textual

- Blocklabels A-E beibehalten.
- JSON-Preview kurz und gekuerzt rendern.
- Skill Stack als Listenkarte mit Status rechts.
- Output Summary als breite Box mit vier kurzen Segmenten.
- Keine echten AI-Aufrufe starten; nur vorhandene Artefakte lesen.

## 10. Abweichungen / Unsicherheiten

- Dateiname hat Tippfehler `Creativ_Startegy`.
- Ob `building strategy` Live-Status oder Mockup-Status ist, muss bei Umsetzung als Demo/Fixture markiert werden, falls nicht live.
