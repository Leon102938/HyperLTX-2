# Image Instructions

Diese Datei legt fest, wie spaetere Bild- und Screenshot-Referenzen fuer den Panel-Build genutzt werden.

## Grundregel

Bilddateien sind Hilfsmittel, aber keine vollstaendige Spezifikation. Jede relevante visuelle Information muss in Text uebersetzt werden, bevor Codex daraus UI-Aenderungen ableitet.

## Vorgehen bei neuen Bildern

1. Bild oder Screenshot identifizieren.
2. Sichtbare UI-Struktur in `VISUAL_SPEC.md` beschreiben.
3. Relevante Panel-Datei in `03_PANEL_SPECS` aktualisieren.
4. Bei Abweichungen vom Design-System die Entscheidung in `06_PROGRESS/OPEN_ISSUES.md` dokumentieren.

## Kanonischer Ablageort

Pipeline-Panel-Bilder werden unter `pipeline_panels/panel_XX_name/reference.png` abgelegt. Zu jedem Bild muss direkt daneben eine `visual_analysis.md` liegen. Diese Analyse ist die Arbeitsgrundlage fuer spaetere UI-Korrekturen; Code darf nicht nur anhand des Dateinamens oder einer unbeschriebenen Bilddatei geaendert werden.

## Aktuelle Pflicht vor weiterem Panel-Build

- `REFERENCE_INDEX.md` lesen.
- Die passende `visual_analysis.md` fuer das betroffene Panel lesen.
- Erst danach vorhandenen UI-Code vergleichen.
- Bei Stage `08` zuerst die Abweichungen aus `panel_08_prompt_compiler/visual_analysis.md` beruecksichtigen.

## Was beschrieben werden muss

- Welche Panels oder Arbeitsbereiche sichtbar sind
- Welche Elemente prominent, zweitrangig oder nur kontextuell sind
- Welche Inhalte interaktiv wirken
- Welche States gezeigt werden
- Welche visuellen Regeln uebernommen werden sollen
- Welche Details bewusst nicht uebernommen werden sollen

## Verbotene Ableitungen

- Keine UI-Aenderung nur aufgrund einer unbeschriebenen Bilddatei.
- Keine Pixelkopie, wenn die bestehende App-Struktur eine andere Loesung nahelegt.
- Keine rein dekorativen Elemente ohne Workflow-Zweck.
