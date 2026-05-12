# 05 Models

## Zweck

Dieses Panel waehlt und konfiguriert die Modelle, die fuer Prompting, Bild-/Keyframe-Erzeugung, Mapping oder Compiler-Schritte genutzt werden.

## Muss sichtbar sein

- Aktives Modell oder Modell-Set
- Modellrollen im Workflow
- Verfuegbarkeit und Konfigurationsstatus

## Verhalten

- Modellwahl beeinflusst Generation, Mapping und Compiler-Ausgabe.
- Nicht verfuegbare Modelle werden sichtbar markiert.
- Modellwechsel muss relevante Outputs invalidieren oder neu bewerten, wenn noetig.

## Inputs

- `activePipelineId`
- `activeMode`
- Verfuegbare Modelle
- Bestehende Modellkonfiguration

## Outputs

- `modelConfig`
- Modellstatus pro Workflow-Schritt
- Hinweise auf fehlende Credentials, Limits oder Inkompatibilitaeten

## UI-Regeln

- Modellrollen klar benennen.
- Keine technischen Details anzeigen, die fuer die Entscheidung nicht helfen.
- Fehler und Verfuegbarkeit nicht verstecken.

## Fehler-/Empty-State

- Empty: keine Modelle geladen oder keine Rolle belegt.
- Error: Modell nicht verfuegbar, falsch konfiguriert oder fuer Mode ungeeignet.

## Akzeptanzkriterien

- Der User versteht, welche Modelle wofuer genutzt werden.
- Fehlende oder kaputte Modellkonfiguration blockiert den Workflow nachvollziehbar.
- Downstream Panels erhalten korrekte Modellinformationen.
