# Panel 07 Visual Analysis

## 1. Zweck des Panels

- Das Panel definiert Scene Contracts als strukturierte Vorgaben fuer den Prompt Compiler.
- Die Pipeline-Stufe ist `07 Scene Contracts`.
- Im Active Workspace werden Inputs, drei Scene Contract Cards und Output Readiness gezeigt.

## 2. Gesamtstruktur

- Workspace-Titel: `ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS`.
- Oben Current-Position-Leiste.
- Darunter drei Hauptbereiche: links `A) CONTRACT INPUTS`, mittig `B) SCENE CONTRACTS`, rechts `C) OUTPUT PREVIEW / READINESS`.
- Unten Prozesskette mit aktivem `Scene Contracts`.

## 3. Farben

- Cyan Borders und Titel.
- Gruen fuer approved, aligned, ready, Checks.
- Gelb fuer drafting und aktive Stage.
- Grau fuer queued.
- Scene-Nummern sind cyan, gelb oder grau je nach Status.
- JSON Preview gruen.

## 4. Typografie / Text-Hierarchie

- Feldlabels in Contract Inputs sind gruen.
- Scene IDs sind cyan.
- Contract-Feldnamen `visual anchor`, `environment`, `action`, `camera`, `lighting`, `allowed visuals`, `forbidden visuals` sind cyan.
- Werte sind hellgrau/weiss.
- Status rechts oben pro Scene ist farbig und klein.

## 5. Panel- und Card-Aufbau

- Linke Inputs-Card mit vertikaler Liste.
- Mittlere grosse Scene-Contracts-Card mit drei horizontalen Scene Cards.
- Jede Scene Card hat links eine nummerierte Preview-Skizze, mittig Contract-Felder, rechts Status und Kreis/Check.
- Rechte Output-Preview-Card mit JSON und darunter Readiness-Checkliste.

## 6. Inhaltliche Bloecke

- Contract Inputs: Creative Strategy approved, Beat/Hook Plan approved, Creative Judge aligned, Mode/Style, Risk Policy, Scene Count.
- Scene 01: misty canopy reveal, sunrise jungle canopy, slow opening reveal, controlled push-in, warm shafts through mist, allowed/forbidden visuals, contract ready.
- Scene 02: suspense jungle trail, narrow dense path, cautious forward motion, low forward glide, filtered green-gold light, drafting.
- Scene 03: golden path payoff, opening jungle corridor, reveal into brighter destination, steady forward push, queued.
- Output Preview: scene_contracts.json, scene_count 3, continuity_rule, text_policy, ready_for image_prompt_compiler.
- Readiness: strategy merged, hook merged, scene rules locked, prompt compilation next.

## 7. Interaktion / Zustaende

- Ready: gruen mit Check.
- Drafting: gelb mit leerem Kreis.
- Queued: grau mit leerem Kreis.
- Aktive Stage in Map und Prozesskette gelb.

## 8. Was NICHT passieren darf

- Scene Contracts nicht als eine zusammengefasste Zeile darstellen.
- Forbidden visuals muessen sichtbar sein.
- Keine echten Bilder; die Sketch-Previews sind Platzhalter/Icons.
- Scene Cards muessen getrennt bleiben.

## 9. Umsetzungshinweise fuer Textual

- Drei Scene Cards mit klaren Trennlinien bauen.
- Contract-Felder kurz und pro Scene gleich strukturiert halten.
- Output Preview rechts oder darunter mit JSON-Auszug.
- Readiness-Checkliste sichtbar halten.
- Lange Visual-Listen kuerzen.

## 10. Abweichungen / Unsicherheiten

- Die kleinen Sketches sind stilisierte Platzhalter; Umsetzung soll keine Bildgenerierung starten.
- Einige Felder sind im bestehenden Fixture eventuell nicht vorhanden und muessen dann `not_checked` bleiben.
