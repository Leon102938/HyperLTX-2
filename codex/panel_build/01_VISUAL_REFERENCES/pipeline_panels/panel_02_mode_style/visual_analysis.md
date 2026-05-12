# Panel 02 Visual Analysis

## 1. Zweck des Panels

- Das Panel zeigt die Auswahl und Sperrung von Mode und Style.
- Die Pipeline-Stufe beschreibt `02 Mode & Style`.
- Im Active Workspace dient es dazu, kreative Grundrichtung, Story-Intent, Tonalitaet, Farbwelt und Skill-Zusammenfassung sichtbar zu machen.

## 2. Gesamtstruktur

- Globaler Header bleibt gleich.
- Links bleiben `SYSTEM STATUS` und `PIPELINE MAP`.
- Hauptbereich beginnt mit `CURRENT POSITION AND PIPELINE PATH`.
- Darunter liegen zwei gleichwertige Karten nebeneinander: links `MODE`, rechts `STYLE`.
- Unter diesen beiden Karten liegt eine breite `MODE / STYLE SUMMARY` Box.
- Darunter liegt die horizontale Pipeline-Komponentenleiste mit aktivem `Keyframes`-Schritt.
- Bottom-Panels zeigen Skill Health, Artifacts, Issues, Next.

## 3. Farben

- Hintergrund dunkel, Border cyan.
- Aktiver Stage-Eintrag `02 Mode & Style` gelb.
- Fertige vorherige Stages gruen.
- Ausgewaehlte Werte wie `visual_adventure`, `cinematic_nature`, `atmospheric`, `deep jungle greens` sind gruen.
- Labels und Section-Titel cyan.
- Lock/Ready-Status ist gruen.
- Fallbacks im Summary wirken weiss/grau mit cyan Bullet.

## 4. Typografie / Text-Hierarchie

- `MODE` und `STYLE` sind cyan uppercase Titel mit Icon.
- Feldlabels wie `Selected Mode`, `Intent`, `Core Goal`, `Structure`, `Status` sind hellgrau.
- Werte sind gruen und visuell dominant.
- `MODE / STYLE SUMMARY` nutzt Unterspalten mit kleinen uppercase Titeln: `MODE SKILLS`, `STYLE SKILLS`, `FALLBACKS`.
- Skilllisten sind kleiner, mit Bullet-Punkten.

## 5. Panel- und Card-Aufbau

- Zwei grosse Cards oben, beide gleich hoch und symmetrisch.
- Jede Card ist in horizontale Zeilen mit Trennlinien aufgeteilt.
- Summary-Box ist dreispaltig.
- Keine Bildvorschauen.
- Die Komponentenleiste ist eine eigene untere Box, nicht Teil der Summary.

## 6. Inhaltliche Bloecke

- `MODE`: Selected Mode, Intent, Core Goal, Structure, Status.
- `STYLE`: Selected Style, Visual Tone, Color Language, Framing, Status.
- `MODE / STYLE SUMMARY`: Mode Skills, Style Skills, Fallbacks.
- `ARTIFACTS`: selected_pipeline.json, mode_selection.json, style_direction.json, skill_plan_preview.json.
- `NEXT`: Stage 03 Skills laden, Operator bestaetigt Mode und Style.

## 7. Interaktion / Zustaende

- `locked / ready` zeigt, dass Mode und Style festgelegt sind.
- Aktive Pipeline-Stage ist gelb.
- Vergangene Stages sind gruen.
- Fallbacks sind sichtbar, aber nicht alarmierend.

## 8. Was NICHT passieren darf

- Mode und Style nicht in eine einzige flache Liste pressen.
- Keine langen Prosatexte statt kurzer Felder.
- Keine aktive Bearbeitungs-UI, wenn der Zustand locked/ready ist.
- Keine verschobene Summary-Spalten.

## 9. Umsetzungshinweise fuer Textual

- Zwei Spalten im Active Workspace nachbilden, soweit Textual-Layout es erlaubt.
- Falls nur Textboxen verfuegbar sind: `MODE` und `STYLE` als zwei getrennte Boxen nacheinander oder mit festen Spaltenbreiten.
- Zeilenlabels kurz halten.
- `locked / ready` als gruenen Wert rendern.
- Summary als drei Spalten mit kurzen Bullet-Listen.

## 10. Abweichungen / Unsicherheiten

- Icons sind visuell wichtig, muessen in Textual eventuell mit ASCII/Unicode-Symbolen ersetzt werden.
- Exakte Farbwerte sind nicht aus dem Bild messbar; Look ist klar neon-gruen/cyan auf dunklem Hintergrund.
