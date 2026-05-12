# Panel 06 Visual Analysis

## 1. Zweck des Panels

- Das Panel bewertet Strategie und Hook und gibt die kreative Entscheidung frei.
- Die Pipeline-Stufe ist `06 Creative Judge`.
- Im Active Workspace zeigt es Judge Input, Checks und finale Creative Decision.

## 2. Gesamtstruktur

- Workspace-Titel: `ACTIVE WORKSPACE / STAGE 06: CREATIVE JUDGE`.
- Oben eine Current-Position-Leiste.
- Darunter drei Spalten: `A) JUDGE INPUT`, `B) CREATIVE CHECKS`, `C) FINAL CREATIVE DECISION`.
- Unter den drei Spalten liegt eine Prozesskette, in der `Creative Judge` gelb hervorgehoben ist.

## 3. Farben

- Cyan fuer Struktur und Titel.
- Gruen fuer pass, approved, selected values.
- Gelb fuer Risiko/low und aktive Stage.
- Graue Trennlinien innerhalb der Checkliste.
- JSON-Preview wieder gruen auf dunklem Panel.

## 4. Typografie / Text-Hierarchie

- Judge-Input-Labels sind gruen oder gelb mit Icons.
- Checknamen sind hellgrau; Statuswerte rechts sind gruen oder gelb.
- Finale Entscheidung `APPROVED` ist gross/gruen.
- JSON Preview ist kleiner und monospaced.

## 5. Panel- und Card-Aufbau

- Judge Input ist eine schmale linke Karte mit mehreren vertikalen Feldern.
- Creative Checks ist eine mittlere Karte mit sechs Zeilen und Status rechts.
- Final Decision ist eine rechte Karte mit Entscheidungsfeldern und JSON-Preview.
- Prozesskette unten verbindet Director, Creative OS, Strategy, Beat/Hook Planner, Creative Judge, Scene Contracts.

## 6. Inhaltliche Bloecke

- Judge Input: Strategy source, Hook source, Goal, Audience, Mode/Style, Key risk.
- Creative Checks: Hook strength, Format fit, Visual feasibility, Story clarity, Artifact risk, Scroll-stop potential.
- Final Decision: Decision approved, Selected hook, Reason, Required changes, Next output.
- Output Preview JSON: decision, selected_hook, artifact_risk, next_stage.

## 7. Interaktion / Zustaende

- Pass: gruen mit Check.
- Low risk: gelb mit neutralem Symbol.
- Approved: gruen.
- Aktive Stage: gelb.

## 8. Was NICHT passieren darf

- Keine Entscheidung ohne Checkliste.
- Kein unerklaertes Approved ohne Reason.
- Risiko nicht verstecken.
- JSON nicht dominanter machen als Entscheidung.

## 9. Umsetzungshinweise fuer Textual

- Drei Spalten nach Moeglichkeit beibehalten.
- Checkliste als feste Zeilen mit rechtsbuendigem Status.
- Decision-Box mit kurzen Feldern.
- Gelbe Risikozeile separat stylen.

## 10. Abweichungen / Unsicherheiten

- Einige Icons sind visuell komplex; Textual kann sie mit einfachen Symbolen ersetzen.
- Ob `CreativeJudge` im Header ohne Leerzeichen gewollt ist, sollte bestaetigt werden.
