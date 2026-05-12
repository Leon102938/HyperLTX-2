# Acceptance Checklist

Diese Liste wird nach jedem spaeteren Build-Durchlauf geprueft.

## Panel-Struktur

- [ ] Alle Panels `01` bis `09` sind sichtbar oder klar erreichbar.
- [ ] Die Reihenfolge ist eindeutig und entspricht dem Workflow.
- [ ] Die aktive Pipeline ist jederzeit sichtbar.
- [ ] Kein Panel wirkt wie Dummy-UI ohne Zweck.

## Inhaltliche Verstaendlichkeit

- [ ] Pipeline Selection ist verstaendlich.
- [ ] Mode ist verstaendlich.
- [ ] Style ist verstaendlich.
- [ ] Hook ist verstaendlich.
- [ ] Models sind verstaendlich.
- [ ] Mapping ist logisch und nachvollziehbar.
- [ ] Keyframe Generation ist verstaendlich.
- [ ] Compiler Workspace ist nachvollziehbar.
- [ ] Active Workspace zeigt den aktuellen Arbeitsstand klar.

## Verhalten und States

- [ ] Inputs und Outputs der Panels sind verbunden.
- [ ] Empty States geben sinnvolle naechste Schritte.
- [ ] Loading States blockieren nicht unklar.
- [ ] Error States sind sichtbar, verstaendlich und loesbar.
- [ ] Keine kaputten States oder Sackgassen im Workflow.

## Qualitaet

- [ ] Build laeuft.
- [ ] Relevante Tests laufen oder nicht ausgefuehrte Tests sind begruendet.
- [ ] Visuelle Pruefung wurde durchgefuehrt.
- [ ] Fortschritt wurde in `06_PROGRESS` aktualisiert.
