# Panel 03 Visual Analysis

## 1. Zweck des Panels

- Das Panel zeigt Skill Loading, Skill Tree und Ladezustand der benoetigten Skill-Gruppen.
- Die Pipeline-Stufe ist `03 Skills laden`.
- Im Active Workspace macht es sichtbar, welche Core-, Pipeline-, Mode-, Style-, Creative-, Safety- und Model-Skills geladen, queued oder partial sind.

## 2. Gesamtstruktur

- Header und Sidebar bleiben erhalten.
- Workspace-Titel lautet `CURRENT POSITION AND SKILL LOADING`.
- Positionsleiste zeigt Current Step `Skills Loading`, Operator Focus `resolve skill tree`, Render Paused `no`, Last Passed `02 Mode & Style`, Next Technical Step `04 Creative Strategy`.
- Darunter drei Hauptbereiche: links `SKILL TREE V1`, mittig `MODE / STYLE CONTEXT` plus `SKILL LOADING PROGRESS`, rechts grosse `LOADING STATUS` Liste.
- Unten ueber die Breite liegt `MATCH / LOAD DETAILS`.

## 3. Farben

- Standard-Borders cyan.
- Aktive Stage gelb.
- Geladen/ok gruen.
- Loading gelb/orange mit Spinner.
- Queued cyan.
- Partial cyan mit dotted indicator.
- Muted Beschreibungen grau.
- Skill-Namen teilweise gruen, cyan oder gelb je nach Status.

## 4. Typografie / Text-Hierarchie

- Section-Titel cyan uppercase.
- Skillgruppen im Tree sind weiss/grau mit cyan Icons.
- Mode/Style-Werte sind cyan/gruen.
- Prozentwert `67%` ist gross und gruen.
- Ladebalken besteht aus gruenen Segmenten und grauen Restsegmenten.
- Loading Status hat nummerierte Kacheln und klare Statuswerte rechts.

## 5. Panel- und Card-Aufbau

- `SKILL TREE V1` ist eine vertikale Baumdarstellung mit Linien.
- `MODE / STYLE CONTEXT` ist eine kompakte Kontextkarte.
- `SKILL LOADING PROGRESS` ist eine separate kleine Karte unter dem Kontext.
- `LOADING STATUS` ist eine grosse Listenkarte mit sechs Reihen.
- `MATCH / LOAD DETAILS` ist eine breite Detailzeile mit zwei Spalten.

## 6. Inhaltliche Bloecke

- Skill Tree: Core Skills, Pipeline Skills, Mode Skills, Style Skills, Creative Skills, Safety Skills, Model Skills.
- Mode/Style Context: Mode, Style, bestaetigter Kontext.
- Skill Loading Progress: Prozent, Segmentbalken, aktuelle Ladung.
- Loading Status: Core loaded, Pipeline loaded, Mode loaded, Style loading, Creative queued, Safety + Model partial.
- Match/Load Details: Core, Mode, Style, Creative, Safety, Model.
- Bottom Artifacts: selected_pipeline.json, mode_style_selection.json, skill_match.json, skill_tree_v1.json, creative_strategy.json.

## 7. Interaktion / Zustaende

- Loaded: gruen mit Check.
- Loading: gelb mit Spinner.
- Queued: cyan mit Uhrsymbol.
- Partial: cyan mit dotted indicator.
- Vorherige Stages sind in Pipeline Map gruen abgehakt.

## 8. Was NICHT passieren darf

- Skill Loading nicht als einfache Statusliste ohne Baum darstellen.
- Progress nicht als Fake-Echtzeit ausgeben, wenn keine echten Daten vorhanden sind.
- Keine fehlenden Skillgruppen verstecken.
- Keine ueberfuellten Details in den Statuszeilen.

## 9. Umsetzungshinweise fuer Textual

- Baum mit Linienzeichen oder eingerueckten Reihen nachbilden.
- Statuszeilen mit fester Nummernspalte und rechter Statusspalte bauen.
- Fortschritt nur anzeigen, wenn Daten vorhanden oder klar Demo/Fixture.
- Match Details kurz halten und in zwei Spalten trennen.

## 10. Abweichungen / Unsicherheiten

- Exakte Spinner-Animation ist im statischen Bild nicht ableitbar.
- Einige Skillnamen sind sichtbar, aber koennen je nach Run dynamisch sein.
