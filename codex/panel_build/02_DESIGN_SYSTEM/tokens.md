# Design Tokens

Diese Datei definiert die Startregeln fuer Design-Tokens. Spaetere Aenderungen muessen mit dem bestehenden App-Design abgeglichen werden.

## Token-Gruppen

- Farbe: Hintergrund, Flaechen, Borders, Text, Akzent, Statusfarben
- Typografie: Font-Familie, Groessen, Gewichtungen, Zeilenhoehen
- Spacing: Panel-Padding, Gaps, Gruppenabstaende, Toolbar-Abstaende
- Radius: Buttons, Inputs, Karten, Panels
- Schatten: nur wenn im bestehenden Design vorhanden und funktional hilfreich
- Z-Index: Overlays, Menues, aktive Arbeitsbereiche

## Regeln

- Bestehende Tokens und CSS-Variablen bevorzugen.
- Neue Tokens nur einfuehren, wenn ein wiederkehrendes Muster entsteht.
- Keine Einmalfarben direkt in Panel-Komponenten verstreuen.
- Statusfarben muessen klar unterscheidbar sein: empty, loading, warning, error, success, active.

## Spaeter zu pruefen

- Wo liegen die aktuellen Tokens?
- Gibt es bereits Theme-Variablen?
- Welche Farben sind fuer Pipeline-Aktivitaet reserviert?
- Welche Spacing-Skala nutzt die App?
