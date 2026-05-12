# Panel 09 Visual Analysis

## 1. Zweck des Panels

- Das Panel zeigt Text-to-Image beziehungsweise Keyframe Generation.
- Die Pipeline-Stufe ist im Bild `06/12 TextToImage`, fachlich aber Referenz fuer `09 Image / Keyframe Generation`.
- Im Active Workspace werden Prompt/Image Jobs mit Bildvorschau, Status, Fortschritt und Queue dargestellt.

## 2. Gesamtstruktur

- Header und linke Sidebar bleiben erhalten.
- Hauptbereich hat oben `CURRENT POSITION AND PIPELINE PATH`.
- Darunter dominiert eine grosse `PROMPTS / IMAGE JOBS` Box.
- In dieser Box liegen drei Image-Jobs als getrennte horizontale Cards.
- Image 2 ist expanded und zeigt Prompt, Statusbar, Prozent, elapsed und Backend.
- Unter den Cards liegt eine horizontale Prozesskette mit aktivem `Keyframes`.

## 3. Farben

- Cyan Standard-Borders.
- Gruen fuer finished und positive Checks.
- Gelb fuer in work, Fortschrittsbalken und aktive Stage.
- Cyan fuer in queue.
- Grau fuer nicht aktive Texte und Trenner.
- Bildvorschauen sind echte/bitmapartige Miniaturen im Mockup.

## 4. Typografie / Text-Hierarchie

- `TextToImage` als Current Step ist gross und cyan.
- `PROMPTS / IMAGE JOBS` ist cyan.
- Image-Titel sind weiss/fett.
- Statuswerte rechts sind farbig und klar: finished, in work, in queue.
- Expanded-Details nutzen kleine Labels: Prompt, status, elapsed, backend.

## 5. Panel- und Card-Aufbau

- Drei getrennte Image Cards.
- Jede Card hat links Nummer und Thumbnail, mittig Titel/Prompt-Auszug, rechts Status und Icon/Caret.
- Expanded Card 2 ist hoeher und zeigt Detailbereich.
- Cards sind durch horizontale Linien getrennt.
- Prozesskette unten ist eigene Box.

## 6. Inhaltliche Bloecke

- Current Position: Current Step, Operator focus, Render paused, Last passed, Next technical step.
- Image 1: finished, Check, Prompt-Auszug.
- Image 2: in work, expanded, Generating Image number 2, Prompt, Statusbar 62%, elapsed 00:20, backend zimage_http.
- Image 3: in queue, Uhrsymbol.
- Artifacts: zimage_prompts.json, keyframe finished/generating/queued, ltx_video_takes_manifest.
- Next: continue keyframe generation, monitor image 2 and queue image 3.

## 7. Interaktion / Zustaende

- Finished: gruen mit Check.
- In work/generating: gelb mit expanded caret.
- In queue: cyan mit Uhr.
- Expanded/collapsed sichtbar ueber Pfeil/Caret.
- Aktive Stage gelb in Pipeline Map.

## 8. Was NICHT passieren darf

- Image Cards duerfen nicht zusammenlaufen.
- Expanded Image 2 muss nutzbar bleiben.
- Keine Fake-Bilder, wenn echte Keyframes nicht vorhanden sind; dann Platzhalter klar markieren.
- Keine ueberlangen Prompttexte.
- Keine Readiness-Zone, die die Cards visuell verdraengt.

## 9. Umsetzungshinweise fuer Textual

- Bestehende Card-Helper mit festen Breiten nutzen.
- Status rechts stabil halten.
- Expanded-Details nur fuer ausgewaehlten Job.
- Wenn Textual keine Bilder rendert, Preview-Slot als `[img]`, `[work]`, `[empty]` darstellen.
- Fortschritt nur als Demo markieren, wenn nicht live.

## 10. Abweichungen / Unsicherheiten

- Bild zeigt Stage `06/12`, nicht aktuelle Registry `09/15`; fachliche Zuordnung bleibt wegen Dateiname und Inhalt sicher.
- Mockup nutzt echte Thumbnails; aktuelle Textual-Umsetzung kann nur Text-Slots zeigen.
