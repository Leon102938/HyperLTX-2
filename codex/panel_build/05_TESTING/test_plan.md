# Test Plan

Dieser Testplan wird spaeter nach jedem Build-Durchlauf genutzt.

## Build und Basis

- Projekt installieren oder vorhandene Dependencies pruefen.
- Build-Kommando aus dem Repo ermitteln und ausfuehren.
- Relevante Unit-, Component- oder E2E-Tests ausfuehren.

## Workflow-Tests

- Pipeline auswaehlen und aktiven Status pruefen.
- Mode wechseln und downstream Reaktion pruefen.
- Style, Hook und Models konfigurieren.
- Mapping erstellen oder validieren.
- Keyframes generieren oder Empty/Error States pruefen.
- Compiler Workspace mit validen und invaliden Daten testen.
- Active Workspace auf konsistenten Endzustand pruefen.

## Regression

- Bestehende Panels duerfen nicht verschwinden.
- Vorhandene funktionierende Workflows duerfen nicht gebrochen werden.
- Keine neuen kaputten States akzeptieren.

## Dokumentation

Nicht ausgefuehrte Tests muessen mit Grund in `../06_PROGRESS/CHANGELOG.md` oder `OPEN_ISSUES.md` dokumentiert werden.
