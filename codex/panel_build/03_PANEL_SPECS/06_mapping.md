# 06 Mapping

## Zweck

Dieses Panel verbindet Pipeline, Mode, Style, Hook und Models mit konkreten Mapping-Regeln fuer Szenen, Inputs, Outputs oder Prompt-Bausteine.

## Muss sichtbar sein

- Aktive Mapping-Struktur
- Zuordnung von Quellen zu Zielwerten
- Validierungsstatus pro Mapping-Bereich

## Verhalten

- Mapping macht sichtbar, wie Daten durch den Workflow fliessen.
- Unvollstaendige Mappings werden markiert.
- Aenderungen muessen Keyframe Generation und Compiler Workspace informieren.

## Inputs

- `activePipelineId`
- `activeMode`
- `styleConfig`
- `hookText`
- `modelConfig`
- Vorhandene Mapping-Daten

## Outputs

- `mappingConfig`
- Validierte Zuordnungen
- Warnungen fuer fehlende oder doppelte Verbindungen

## UI-Regeln

- Mapping muss logisch lesbar sein, nicht nur als rohe Datenstruktur.
- Quellen und Ziele klar trennen.
- Kritische fehlende Zuordnungen sichtbar hervorheben.

## Fehler-/Empty-State

- Empty: noch kein Mapping erstellt, mit klarer Startaktion oder Auto-Mapping-Hinweis.
- Error: widerspruechliche, zirkulaere oder unvollstaendige Zuordnung.

## Akzeptanzkriterien

- Mapping ist nachvollziehbar.
- Folgepanels koennen Mapping-Ergebnisse nutzen.
- Fehlerhafte Zuordnungen sind sichtbar und korrigierbar.
