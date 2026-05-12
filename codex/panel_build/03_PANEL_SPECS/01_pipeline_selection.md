# 01 Pipeline Selection

## Zweck

Dieses Panel waehlt die aktive Pipeline fuer den Workflow aus und macht sichtbar, welcher Prozess gerade bearbeitet wird.

## Muss sichtbar sein

- Liste oder Auswahl der verfuegbaren Pipelines
- Aktive Pipeline
- Status der Pipeline: leer, konfiguriert, in Arbeit, fehlerhaft oder bereit

## Verhalten

- Auswahl setzt die aktive Pipeline fuer alle nachfolgenden Panels.
- Wechsel der Pipeline aktualisiert Mode, Style, Hook, Models, Mapping, Keyframes und Workspace-Kontext.
- Nicht verfuegbare Pipelines werden sichtbar deaktiviert statt versteckt.

## Inputs

- Verfuegbare Pipeline-Konfigurationen
- Aktueller Pipeline-State
- Optional: Projekt- oder Session-Kontext

## Outputs

- `activePipelineId`
- Pipeline-Metadaten fuer Folgepanels
- Statushinweise fuer Navigation und Workspace

## UI-Regeln

- Aktive Pipeline klar markieren.
- Pipeline-Auswahl kompakt und wiedererkennbar halten.
- Keine Pipeline-Karten ohne Status oder Zweck anzeigen.

## Fehler-/Empty-State

- Empty: keine Pipeline verfuegbar, mit Hinweis auf naechsten Setup-Schritt.
- Error: Pipeline konnte nicht geladen werden, mit Wiederholen- oder Diagnose-Aktion.

## Akzeptanzkriterien

- Der User erkennt jederzeit die aktive Pipeline.
- Pipeline-Wechsel fuehrt nicht zu widerspruechlichen States.
- Folgepanels erhalten den richtigen Pipeline-Kontext.
