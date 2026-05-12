# Panel 01 Visual Analysis

## 1. Zweck des Panels

- Das Panel zeigt die Pipeline-Auswahl und den Gesamtpfad der Content Maschine.
- Die Pipeline-Stufe ist `01 Pipeline overview` beziehungsweise Pipeline-Auswahl.
- Im Active Workspace erklaert es, welche Pipeline aktiv ist, welche Assets erwartet werden und wie die 12 sichtbaren Workflow-Schritte zusammenhaengen.

## 2. Gesamtstruktur

- Oben liegt der globale Header mit Logo, Job, Pipeline, Mode, Format, Status, Zeit, Session und Operator.
- Links stehen zwei feste Sidebar-Panels: `SYSTEM STATUS` oben und `PIPELINE MAP` darunter.
- Der zentrale Workspace ist gross und dominiert die Ansicht.
- Im Workspace steht oben eine schmale Positionsleiste mit `Current Step`, `Operator focus`, `Render paused`, `Last passed` und `Next technical step`.
- Darunter liegt eine grosse Box `SELECTED PIPELINE: shortform_storyboard_v1`.
- Innerhalb dieser Box gibt es links `PIPELINE PURPOSE` und `PIPELINE ASSETS (EXPECTED)`, rechts eine breite `PIPELINE FLOW (12 STAGES)` Liste.
- Unten im Workspace steht `PIPELINE COMPONENTS` als horizontale Prozesskette mit Kacheln: Director, Creative OS, Z-Image Prompts, Keyframes, Keyframe QA, LTX Motion Prompts, LTX I2V Takes, Assembly.
- Ganz unten liegen Skill Health, Artifacts, Issues und Next als separate Bottom-Panels.

## 3. Farben

- Grundlook: sehr dunkler, fast schwarzer Hintergrund mit subtiler gruen-blauer Terminal-Anmutung.
- Panel-Borders: cyan/tuerkis fuer Standard-Panelgrenzen.
- Header-Border und Logo-Akzent: helles Neon-Gruen.
- Aktive Elemente: gelb/orange, etwa `01 Pipeline overview`, aktive Stage-Markierung und aktive Komponenten.
- Fertige/positive Elemente: Neon-Gruen mit Check-Mark.
- Wartende Elemente: cyan/blau fuer `upcoming`.
- Fehlend/unbekannt: grau oder gedimmtes Weiss.
- Textfarben: Titel cyan, Werte weiss/grau, wichtige Runtime-Werte gruen, naechste Schritte gelb.

## 4. Typografie / Text-Hierarchie

- Hauptbrand `CONTENT MASCHINE LIVE` ist sehr gross, gruen, monospaced und fett.
- Section-Titel wie `SYSTEM STATUS`, `PIPELINE MAP`, `CURRENT POSITION AND PIPELINE PATH` sind cyan, uppercase und prominent.
- Labels sind kleiner, hellgrau oder cyan.
- Werte sind weiss, gruen, gelb oder cyan je nach Status.
- Hilfstexte in Pipeline Purpose sind klein und weiss/grau.
- Stage-Titel in der Flow-Liste sind unterstrichen oder deutlich heller als Beschreibungen.

## 5. Panel- und Card-Aufbau

- Keine verschachtelten Kartenberge; stattdessen wenige grosse, klar begrenzte Boxen.
- Die zentrale Selected-Pipeline-Box ist in eine linke Info-Spalte und eine rechte Flow-Spalte geteilt.
- Pipeline Assets ist eine kompakte Liste mit kleinen Icons.
- Pipeline Components ist eine horizontale Card-Reihe mit Pfeilen zwischen Komponenten.
- Sidebar und Bottom-Panels sind eigenstaendige Panels, nicht Teil des Workspace-Inhalts.

## 6. Inhaltliche Bloecke

- `SYSTEM STATUS`: API, Director, Image Backend, Video Backend, Vision Review, Voice, Music, Subtitles.
- `PIPELINE MAP`: Stage-Liste mit aktivem gelbem Marker und grauen offenen Stages.
- `CURRENT POSITION AND PIPELINE PATH`: Current Step, Operator focus, Render paused, Last passed, Next technical step.
- `SELECTED PIPELINE`: aktive Pipeline-ID.
- `PIPELINE PURPOSE`: Kurzbeschreibung der Storyboard-/Keyframe-Pipeline.
- `PIPELINE ASSETS`: Keyframes, Voice, Music, Subtitles, Outplay, Final Output.
- `PIPELINE FLOW`: nummerierte Stages mit Zweck und Status.
- `PIPELINE COMPONENTS`: technische/konzeptionelle Bausteine der Pipeline.
- `ARTIFACTS`: zimage prompts, Keyframe Status, LTX manifest.
- `NEXT`: technischer und Operator-Next-Step.

## 7. Interaktion / Zustaende

- Aktive Stage ist gelb markiert.
- Upcoming-Stages sind cyan.
- System-Ready ist gruen.
- Ungeprueft ist grau oder Fragezeichen.
- Pipeline-Komponenten koennen aktiv, zukuenftig oder normal wirken.

## 8. Was NICHT passieren darf

- Keine reinen Textlisten ohne visuelle Pipeline-Struktur.
- Keine Umbenennung in generische Begriffe wie nur `Pipeline Selection`, wenn das Mockup `Pipeline overview` und `Pipeline path` zeigt.
- Keine fehlende Komponentenleiste.
- Keine ueberfuellten Stage-Beschreibungen; die Flow-Liste bleibt tabellarisch und scanbar.
- Keine kaputten Borders oder verschobenen Spalten.

## 9. Umsetzungshinweise fuer Textual

- Bestehende Box-/Border-Helper fuer cyan Boxen verwenden.
- Pipeline Flow als kompakte Tabellenzeilen bauen.
- Komponentenleiste mit festen Breiten und Pfeilen umsetzen.
- Lange Texte in Purpose und Flow kuerzen.
- Statuswerte aus State lesen; Demo-Werte klar als Demo behandeln.
- Sidebar/Bottom nicht in den Active Workspace duplizieren.

## 10. Abweichungen / Unsicherheiten

- Im Bild steht links bei finaler Stage teilweise `12 Final output`, obwohl andere Mockups 15 Stages zeigen; Stage-Zaehlung muss bestaetigt werden.
- Der Dateiname enthaelt `Pipline`, nicht `Pipeline`; Zuordnung ist trotzdem sicher.
