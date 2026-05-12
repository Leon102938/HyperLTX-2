# 09 Active Workspace

## Zweck

Dieses Panel zeigt den aktuellen aktiven Arbeitsstand nach Auswahl, Mapping, Keyframe-Erzeugung und Compiler-Vorbereitung.

## Muss sichtbar sein

- Aktiver Workspace-Kontext
- Aktuelle Pipeline und Mode
- Ergebnis-, Preview- oder Arbeitsstatus
- Naechste sinnvolle Aktion

## Verhalten

- Workspace aktualisiert sich aus den vorherigen Panels.
- Er zeigt Ergebnisse, offene Aufgaben und Fehler an einem nachvollziehbaren Ort.
- Er darf keine stillen Widersprueche zwischen Pipeline, Keyframes und Compiler-Status enthalten.

## Inputs

- `activePipelineId`
- `activeMode`
- `compilerInput`
- `keyframes`
- Ergebnis- oder Preview-Daten
- Fehler- und Statusdaten

## Outputs

- Aktueller Workspace-State
- User-Aktionen fuer Review, Export, Weiterarbeit oder Korrektur
- Rueckverweise auf Panels mit offenen Problemen

## UI-Regeln

- Der aktive Arbeitsstand steht im Vordergrund.
- Kontext und Status bleiben sichtbar.
- Aktionen muessen klar vom reinen Preview-Bereich getrennt sein.

## Fehler-/Empty-State

- Empty: noch kein aktiver Workspace, mit Hinweis auf fehlende Voraussetzung.
- Loading: laufende Aktualisierung oder Compile-Ergebnis sichtbar.
- Error: Ergebnis kann nicht dargestellt oder Workspace-State ist inkonsistent.

## Akzeptanzkriterien

- Active Workspace ist als finaler Arbeitsbereich erkennbar.
- Ruecksprung zu fehlerhaften Vorpanels ist nachvollziehbar.
- Keine kaputten oder widerspruechlichen Workspace-States.
