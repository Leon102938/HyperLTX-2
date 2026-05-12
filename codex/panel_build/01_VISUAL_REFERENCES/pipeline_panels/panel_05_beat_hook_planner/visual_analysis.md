# Panel 05 Visual Analysis

## 1. Zweck des Panels

- Das Panel plant Hook und Beat-Struktur.
- Die Pipeline-Stufe ist `05 Beat / Hook Planner`.
- Im Active Workspace werden Hook Brief, Kandidaten und ausgewaehlter Beat Plan sichtbar.

## 2. Gesamtstruktur

- Workspace-Titel: `ACTIVE WORKSPACE / STAGE 05: BEAT / HOOK PLANNER`.
- Oben eine breite Current-Position-Leiste.
- Darunter links `A) HOOK BRIEF`, mittig `B) HOOK OPTIONS / BEAT CANDIDATES`, rechts `C) SELECTED BEAT PLAN`.
- Unten im Workspace eine horizontale Prozesskette von Director bis Scene Contracts.

## 3. Farben

- Cyan Borders und Titel.
- Aktive Stage und ausgewaehlte Komponente gelb.
- Ausgewaehlter Kandidat hat cyan Umrandung und Check.
- Kandidaten 2/3 haben gelbe oder graue Nummern und Auswahlkreise.
- Gruen fuer fertige/positive Artefakte.
- JSON-Preview Text gruen.

## 4. Typografie / Text-Hierarchie

- Blocktitel cyan uppercase mit Buchstaben A-C.
- Hook Brief Labels gruen mit grossen Icons.
- Kandidaten haben nummerierte Badges.
- Candidate-Details nutzen Labels `Hook type`, `Opening visual`, `Camera idea`, `Beat feel`.
- Selected Beat Plan nutzt Beat 1/2/3 mit gruenen Titeln und hellgrauen Beschreibungen.

## 5. Panel- und Card-Aufbau

- Hook Brief ist eine schmale linke Karte.
- Kandidatenbereich ist die dominante mittlere Karte mit drei gestapelten Kandidaten.
- Jeder Kandidat hat eine kleine Preview-Box links und Textfelder rechts.
- Selected Beat Plan rechts hat drei vertikale Beat-Zeilen plus JSON-Preview.
- Prozesskette unten markiert `Beat / Hook Planner` gelb.

## 6. Inhaltliche Bloecke

- Hook Brief: Goal, Audience pull, Visual angle, Tone.
- Hook Options: Kandidat 1 selected, Kandidat 2/3 candidate.
- Selected Beat Plan: Beat 1 Hook open, Beat 2 Build tension, Beat 3 Micro payoff.
- Output Preview JSON: hook_type, opening_visual, beat_count, selected_candidate, transition_note.
- Artifacts: creative_strategy.json, beat_hook_plan.json, creative_judge.json, scene_contracts.json.

## 7. Interaktion / Zustaende

- Selected Candidate: cyan Check und selected Text.
- Candidate: leerer Kreis.
- Aktive Stage: gelb in Pipeline Map und Prozesskette.
- Noch fehlende Artefakte: graue Kreise.

## 8. Was NICHT passieren darf

- Keine simple Hook-Textbox ohne Kandidatenvergleich.
- Beat Plan nicht ohne Auswahlstatus zeigen.
- Keine echten Bildvorschauen erfinden; im Mockup sind nur geplante/concept Visual Slots.
- JSON Preview klein halten.

## 9. Umsetzungshinweise fuer Textual

- Kandidaten als drei klar getrennte Cards bauen.
- Preview-Slots als Text/ASCII-Placeholder markieren, nicht als echte Bilder.
- Selected-State deutlich markieren.
- Output Preview gekuerzt rendern.
- Prozesskette unten nur nachbilden, wenn sie in Active Workspace Scope gehoert.

## 10. Abweichungen / Unsicherheiten

- Die Bild-Preview-Slots sind im Mockup gezeichnete Platzhalter; Umsetzung darf keine echten Bilder laden.
- Exakte Kandidateninhalte koennen je nach Run variieren.
