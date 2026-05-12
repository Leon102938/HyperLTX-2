# 07 Keyframe Generation

## Zweck

Dieses Panel erzeugt oder verwaltet Keyframes auf Basis von Pipeline, Mode, Style, Hook, Models und Mapping.

## Muss sichtbar sein

- Keyframe-Liste oder Generation Workspace
- Status je Keyframe
- Generieren-, Aktualisieren- oder Uebernehmen-Aktion

## Verhalten

- Keyframes werden aus validierten Eingaben erzeugt.
- Einzelne Keyframes koennen geprueft, angepasst oder neu generiert werden.
- Ungueltiges Mapping oder fehlende Modelle blockieren Generation nachvollziehbar.

## Inputs

- `activePipelineId`
- `activeMode`
- `styleConfig`
- `hookText`
- `modelConfig`
- `mappingConfig`

## Outputs

- `keyframes`
- Generation-Status
- Fehler- und Warnhinweise pro Keyframe

## UI-Regeln

- Keyframes muessen als Arbeitsobjekte erkennbar sein.
- Status und Aktionen pro Keyframe konsistent darstellen.
- Keine generischen Platzhalter ohne Workflow-Bezug.

## Fehler-/Empty-State

- Empty: noch keine Keyframes erzeugt, mit klarer Startaktion.
- Loading: laufende Generation prozessklar anzeigen.
- Error: fehlgeschlagene Generation mit Ursache und Retry-Moeglichkeit.

## Akzeptanzkriterien

- Keyframe Generation ist verstaendlich.
- Der User sieht, welche Inputs fehlen.
- Erfolgreiche Keyframes koennen an den Compiler weitergegeben werden.
