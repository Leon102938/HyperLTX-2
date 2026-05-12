# 04 Hook

## Zweck

Dieses Panel definiert den Hook oder zentralen kreativen Einstiegspunkt, der die Pipeline inhaltlich fuehrt.

## Muss sichtbar sein

- Aktiver Hook
- Eingabe oder Auswahl fuer Hook-Varianten
- Verbindung zu Style, Mode und Mapping

## Verhalten

- Hook kann manuell eingegeben, aus Vorschlaegen gewaehlt oder aus bestehendem Kontext abgeleitet werden.
- Aenderungen am Hook aktualisieren die inhaltliche Ausrichtung nachfolgender Panels.
- Leerer Hook muss als unvollstaendiger Workflow sichtbar sein.

## Inputs

- `activePipelineId`
- `activeMode`
- `styleConfig`
- User-Eingabe oder generierte Vorschlaege

## Outputs

- `hookText`
- Hook-Metadaten oder Varianten
- Validierungsstatus fuer Generation und Mapping

## UI-Regeln

- Hook prominent, aber nicht uebergross darstellen.
- Varianten klar vergleichbar machen.
- Bearbeiten und Auswaehlen eindeutig trennen.

## Fehler-/Empty-State

- Empty: noch kein Hook vorhanden, mit klarer Eingabemoeglichkeit.
- Error: Hook kann nicht verarbeitet werden oder passt nicht zum Mode.

## Akzeptanzkriterien

- Der aktive Hook ist eindeutig.
- Hook-Aenderungen sind in nachfolgenden Panels nachvollziehbar.
- Der Workflow zeigt klar, wenn ein Hook fehlt.
