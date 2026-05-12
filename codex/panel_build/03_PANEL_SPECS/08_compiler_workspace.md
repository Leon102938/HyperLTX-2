# 08 Compiler Workspace

## Zweck

Dieses Panel bereitet die validierten Inputs, Mappings und Keyframes fuer den Compiler-Workflow vor.

## Muss sichtbar sein

- Compiler-Eingaben
- Zusammenfassung der verwendeten Pipeline-Konfiguration
- Compile-Status oder Vorbereitungsschritte

## Verhalten

- Workspace zeigt, welche Daten kompiliert werden.
- Fehlende Voraussetzungen blockieren Compile-Aktionen mit klarer Begruendung.
- Erfolgreiche Vorbereitung uebergibt Daten an den aktiven Workspace.

## Inputs

- `activePipelineId`
- `activeMode`
- `styleConfig`
- `hookText`
- `modelConfig`
- `mappingConfig`
- `keyframes`

## Outputs

- `compilerInput`
- Compile-Status
- Validierungs- oder Preview-Daten

## UI-Regeln

- Compiler Workspace muss nachvollziehbar sein.
- Rohdaten nur zeigen, wenn sie fuer Kontrolle oder Fehleranalyse nuetzlich sind.
- Primaere Compile-Aktion klar positionieren.

## Fehler-/Empty-State

- Empty: keine kompilierbaren Daten vorhanden.
- Loading: Compile-Vorbereitung oder Compile-Lauf sichtbar.
- Error: Compile-Voraussetzungen oder Compiler-Fehler erklaeren.

## Akzeptanzkriterien

- Der User versteht, was kompiliert wird.
- Fehlende Inputs sind eindeutig.
- Erfolgreiche Vorbereitung fuehrt logisch zu Panel `09_active_workspace`.
