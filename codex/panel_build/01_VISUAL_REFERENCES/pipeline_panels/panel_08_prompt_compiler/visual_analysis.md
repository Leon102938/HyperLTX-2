# Panel 08 Visual Analysis

## 1. Zweck des Panels

- Das Panel zeigt den Prompt Compiler mit aktivem Image Compiler.
- Die Pipeline-Stufe ist `08.1 Prompt Compiler / Image Compiler`.
- Im Active Workspace wird sichtbar, wie Scene Contracts, Assets, Style Signals und Skill Sources zu finalen Prompt Payloads werden.

## 2. Gesamtstruktur

- Workspace-Titel: `ACTIVE WORKSPACE - PROMPT COMPILER`.
- Oben Current-Position-Leiste mit Current Step `PromptCompiler`, Operator focus `compile final prompts`, Last passed `07 Scene Contracts`, Next technical step `09 Image / Keyframe Generation`.
- Darunter `COMPILER SCOPE / OVERVIEW` als breite Metrik-Leiste.
- Hauptbereich darunter ist zweigeteilt: links eine grosse gruen gerahmte `IMAGE COMPILER (ACTIVE)` Zone, rechts drei gestapelte Kontextkarten fuer Video, Audio und Music Compiler.
- In der Image-Compiler-Zone links `SCENE CONTRACT INPUTS`, rechts `SCENE PROMPT SUMMARIES`, unten `FINAL PROMPT PAYLOAD (JSON PREVIEW)` mit Copy-Button.
- Unten Prozesskette mit `Prompt Compiler` und `Image Compiler` hervorgehoben.

## 3. Farben

- Dunkler Hintergrund.
- Standard-Borders cyan.
- Aktiver Image-Compiler-Bereich hat deutlich gruenen Border, nicht nur cyan.
- Aktive Pipeline-Map-Unterstage `08.1 Image Compiler` ist gruen/gelb markiert.
- Kompilierte Scene-Prompts sind gruen.
- Validating ist gelb.
- Confidence-Werte gruen.
- Queued Compiler rechts haben cyan Badges.
- Warnung unten `scene_03 contrast balance` ist gelb.

## 4. Typografie / Text-Hierarchie

- `IMAGE COMPILER (ACTIVE)` ist gruen uppercase und sehr prominent.
- `COMPILER SCOPE / OVERVIEW` ist cyan.
- Kleine Metrikwerte `Scenes 3`, `Assets 12`, `Style Signals 7`, `Skill Sources 8`, `Output Targets image video audio music`.
- Scene IDs sind klein, Statusbadges `compiled`/`validating` farbig.
- Prompt Summary ist kleiner, mehrzeilig, aber gekuerzt.
- JSON Preview ist klein und monospaced.

## 5. Panel- und Card-Aufbau

- Der wichtigste Unterschied: Stage 08 ist kein allgemeiner Readiness-Screen, sondern ein Compiler-Workspace mit aktivem gruenem Image-Compiler-Panel.
- `SCENE CONTRACT INPUTS` ist eine linke kompakte Parameterliste.
- `SCENE PROMPT SUMMARIES` zeigt drei Scene Rows, jede mit Statusbadge, Thumbnail/Preview-Bild und Prompt-Text plus Confidence.
- `FINAL PROMPT PAYLOAD` ist eine breite JSON-Karte unten im Image Compiler.
- Rechts stehen drei separate Karten: Video Compiler, Audio Compiler, Music Compiler; alle queued.

## 6. Inhaltliche Bloecke

- Compiler Scope: Scenes, Assets, Style Signals, Skill Sources, Output Targets.
- Scene Contract Inputs: Style, Mode, Tone, Lighting, Camera, Composition, Color Grade, Duration Target, Resolution, Aspect.
- Scene Prompt Summaries: scene_01 compiled, scene_02 compiled, scene_03 validating; pro Scene Prompttext und Confidence.
- Final Prompt Payload: project, mode, format, scenes mit id/prompt/style/cam/mood.
- Video Compiler: target, motion style, transitions, duration mapping, output.
- Audio Compiler: ambience, SFX, VO, output.
- Music Compiler: mood, tempo, instrumentation, key, output.
- Artifacts: scene_contracts.json, style_signals.json, asset_manifest.json, prompt_payload_compiled.json in progress, video/audio/music prompts queued.

## 7. Interaktion / Zustaende

- Image Compiler active: gruen.
- Scene 01/02 compiled: gruen.
- Scene 03 validating: gelb.
- Video/Audio/Music queued: cyan.
- Copy-Button sichtbar am JSON Payload.
- Pipeline Map hat Unterpunkte 08.1 bis 08.4.

## 8. Was NICHT passieren darf

- Stage 08 darf nicht nur `COMPILER READINESS` und generische Prompt Cards zeigen; das Mockup verlangt eine aktive Image-Compiler-Zone.
- Video/Audio/Music duerfen nicht implementiert wirken; sie sind queued/context.
- Keine echten neuen Bilder generieren; vorhandene kleine Scene-Previews im Mockup sind visuelle Referenz, in Textual ggf. nur Text/ASCII oder vorhandene Assets.
- Keine riesigen Prompttexte; Summaries muessen kurz bleiben.
- Keine fehlende Compiler-Scope-Leiste.

## 9. Umsetzungshinweise fuer Textual

- Zuerst `COMPILER SCOPE / OVERVIEW` bauen.
- Dann eine dominante `IMAGE COMPILER (ACTIVE)` Box mit gruenem/active Styling, falls Theme das zulaesst.
- Innerhalb: Contract Inputs, Prompt Summaries, Final Payload.
- Rechts oder darunter: Video/Audio/Music Compiler als queued Kontextkarten.
- Prompt Summaries als drei feste Cards/Rows mit Status und Confidence.
- JSON Preview stark kuerzen und keinen Copy-Button funktional machen, wenn keine Interaktion existiert.
- Daten spaeter aus scene_contracts, style_signals, asset_manifest, prompt_payload_compiled oder zimage/model prompts lesen.

## 10. Abweichungen / Unsicherheiten

- Das Bild zeigt kleine Scene-Thumbnails; wenn keine echten Assets vorhanden sind, duerfen keine Fake-Bilder ausgegeben werden.
- Stage-Nummer im Header ist `08.1`, waehrend aktuelle Stage Registry `08` nutzt. Keine Stage-ID-Neuordnung ohne separate Entscheidung.
- Current implementation weicht stark ab: sie zeigt Readiness und lineare Prompt Cards, aber nicht die gruen gerahmte Image-Compiler-Workspace-Struktur mit Scope-Leiste, Prompt Summaries, JSON Payload und queued Compiler-Familie rechts.
