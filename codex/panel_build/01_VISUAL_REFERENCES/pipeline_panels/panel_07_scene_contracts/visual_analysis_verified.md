# Stage 07 Scene Contracts - Verified Visual Analysis

## 1. Kann das Bild visuell ausgewertet werden?

- ja
- Begruendung: Das Bild `/workspace/codex/panel_build/01_VISUAL_REFERENCES/pipeline_panels/panel_07_scene_contracts/reference.png` wurde visuell geoeffnet und ist lesbar. Die Hauptstruktur, Titel, Blocknamen, viele Feldlabels, Statusanzeigen, Farben und die relative Anordnung sind sichtbar.

## 2. Rohbeschreibung des Bildes

### A) Wirklich im Bild sichtbar

- Das Bild zeigt ein dunkles Cockpit-UI mit einem grossen globalen Header oben, linker Sidebar, zentralem Active Workspace und Bottom-Panels.
- Oben links steht ein gruenes `CM`-Logo in einem eckigen Rahmen.
- Im Header steht gross `CONTENT MASCHINE LIVE`, darunter kleiner `CREATIVE OS COCKPIT`.
- Im mittleren Headerbereich stehen Job, Pipeline, Mode, Format und Status.
- Sichtbare Headerwerte:
  - `Job: creative-os-jungle-001`
  - `Pipeline: shortform_storyboard_v1`
  - `Mode: visual_adventure • jungle safari at sunrise`
  - `Format: portrait • 512x768 • 9.0s • 3 scenes`
  - `Status: running • stage 07/15 • SceneContracts`
- Rechts im Header stehen `TIME (UTC)`, `2025-05-22 12:47:31`, `SESSION a1f9c2b7`, `OPERATOR cmd_center`.
- Links oben in der Sidebar steht `SYSTEM STATUS`.
- Darunter steht in der Sidebar `PIPELINE MAP`.
- Im Pipeline Map ist Stage `07 Scene Contracts` gelb aktiv markiert.
- Stages `00` bis `06` sind gruen abgehakt.
- Stages `08` bis `15` sind grau/offen.
- Der zentrale Bereich hat den Titel `ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS`.
- Direkt darunter steht eine Box `CURRENT POSITION AND PIPELINE PATH`.
- In dieser Position-Box stehen:
  - `Current Step: SceneContracts`
  - `Operator focus: lock scene-level production rules`
  - `Render paused: yes`
  - `Last passed: ✓ 06 Creative Judge`
  - `Next technical step: ○ 08 Image Prompt Compiler`
- Unter der Position-Box liegen drei Hauptspalten:
  - links `A) CONTRACT INPUTS`
  - mittig `B) SCENE CONTRACTS`
  - rechts `C) OUTPUT PREVIEW / READINESS`
- Unter diesen drei Spalten liegt eine horizontale Prozesskette mit mehreren Step-Karten.
- In der Prozesskette sind sichtbar: `Director`, `Creative OS`, `Strategy`, `Beat / Hook Planner`, `Creative Judge`, `Scene Contracts`, `Image Prompt Compiler`.
- Die Karte `Scene Contracts` ist gelb hervorgehoben.
- Unten sind vier Bottom-Bereiche sichtbar: `SKILL HEALTH`, `ARTIFACTS`, `ISSUES`, `NEXT`.

### Sichtbare Labels und Inhalte in `A) CONTRACT INPUTS`

- `Creative Strategy: approved`
- `Beat / Hook Plan: approved`
- `Creative Judge: aligned`
- `Mode / Style: visual_adventure • cinematic_nature`
- `Risk Policy: text_safe / artifact_avoidance`
- `Scene Count: 3`
- Die Punkte sind vertikal angeordnet und durch gestrichelte horizontale Linien getrennt.
- Zu jedem Punkt gibt es links ein gruenes Icon.

### Sichtbare Inhalte in `B) SCENE CONTRACTS`

- Es gibt drei untereinander angeordnete Scene Cards.
- Jede Scene Card hat:
  - links eine nummerierte Box
  - daneben eine quadratische Sketch-/Placeholder-Flache mit gestricheltem Rand
  - rechts davon strukturierte Contract-Felder
  - ganz rechts eine Statusanzeige und einen Kreis/Check
- Scene 1:
  - Nummer `1` cyan
  - `scene_01`
  - `visual anchor: misty canopy reveal`
  - `environment: sunrise jungle canopy`
  - `action: slow opening reveal through leaves`
  - `camera: controlled push-in`
  - `lighting: warm shafts through mist`
  - `allowed visuals: leaves, vines, depth haze`
  - `forbidden visuals: text, logos, extra animals`
  - Status rechts: `status: contract ready`
  - Rechts unten: gruenes Check-Symbol im Kreis.
- Scene 2:
  - Nummer `2` gelb
  - `scene_02`
  - `visual anchor: suspense jungle trail`
  - `environment: narrow dense path`
  - `action: cautious forward motion`
  - `camera: low forward glide`
  - `lighting: filtered green-gold light`
  - `allowed visuals: wet leaves, foliage depth, trail`
  - `forbidden visuals: captions, UI text, crowd elements`
  - Status rechts: `status: contract drafting`
  - Rechts unten: gelber leerer Kreis.
- Scene 3:
  - Nummer `3` grau
  - `scene_03`
  - `visual anchor: golden path payoff`
  - `environment: opening jungle corridor`
  - `action: reveal into brighter destination`
  - `camera: steady forward push`
  - `lighting: strong golden end light`
  - `allowed visuals: path, sun rays, layered foliage`
  - `forbidden visuals: signage, duplicate subjects, overlays`
  - Status rechts: `status: queued`
  - Rechts unten: grauer leerer Kreis.

### Sichtbare Inhalte in `C) OUTPUT PREVIEW / READINESS`

- Oben steht `Output Preview (JSON)`.
- Darunter liegt eine innere dunkle JSON-Box.
- In der JSON-Box ist sichtbar:
  - `"file": "scene_contracts.json"`
  - `"scene_count": 3`
  - `"continuity_rule":`
  - `"consistent jungle sunrise progression"`
  - `"text_policy": "no readable text inside generated imagery"`
  - `"ready_for": "image_prompt_compiler"`
- Unter der JSON-Box steht eine Readiness-Liste:
  - `strategy merged` mit gruenem Check
  - `hook merged` mit gruenem Check
  - `scene rules locked` mit gruenem Check
  - `prompt compilation next` mit grauem/offenem Kreis

### Sichtbare Bottom-Bereiche

- `SKILL HEALTH` zeigt `ok`, `loaded 8`, `fallbacks 2`, `missing optional 1`.
- `ARTIFACTS` listet:
  - `creative_strategy.json`
  - `beat_hook_plan.json`
  - `creative_judge.json`
  - `scene_contracts.json`
- `scene_contracts.json` ist gelb markiert.
- `ISSUES` zeigt `none blocking`.
- `NEXT` zeigt:
  - `Technical: Stage 08 compile image prompts`
  - `Operator: review scene contracts and continue`

### B) Nur aus `visual_analysis.md`

- Die Formulierung, dass das Panel "strukturierte Vorgaben fuer den Prompt Compiler" definiert, stammt aus der bestehenden Analyse.
- Die Einordnung als "keine echten Bilder; Sketch-Previews sind Platzhalter/Icons" stammt aus der bestehenden Analyse, ist aber visuell plausibel.
- Die Umsetzungshinweise zu Textual stammen aus der bestehenden Analyse.

### C) Interpretation

- Die gestrichelten Sketch-Flaechen in den Scene Cards wirken wie geplante visuelle Platzhalter, nicht wie echte generierte Bilder.
- `contract ready`, `contract drafting` und `queued` bilden wahrscheinlich den Fortschritt der Scene-Contract-Erstellung.
- `prompt compilation next` zeigt, dass Stage 07 die direkte Vorstufe fuer Stage 08 ist.
- Die gelbe Markierung von `scene_contracts.json` im Artifacts-Panel wirkt wie "aktuell erzeugt/neu/relevant", nicht wie final gruen abgeschlossen.

## 3. Exakte Layout-Struktur

- Global Frame
  - Position: ganzes Bild
  - Inhalt: Header, Sidebar, Active Workspace, Bottom-Band
  - Farbe/Border: dunkler Hintergrund, gruene Header-Border, cyan Panel-Borders

- Header
  - Position: ganz oben, volle Breite
  - Groesse: etwa 15 Prozent der Bildhoehe
  - Inhalt: Logo/Brand links, Run-Metadaten mittig, Zeit/Session/Operator rechts
  - Prioritaet: globaler Kontext, nicht Stage-spezifisches Hauptpanel
  - Farbe/Border: gruener Rahmen, gruene Brand-Schrift, weisse/graue Metadaten
  - Ausrichtung: horizontal

- Left Sidebar
  - Position: links unter Header
  - Groesse: etwa 22 Prozent der Breite
  - Inhalt: `SYSTEM STATUS` oben, `PIPELINE MAP` darunter
  - Prioritaet: Navigation und Systemkontext
  - Farbe/Border: cyan Borders, dunkle Panels
  - Ausrichtung: vertikal

- Outer Active Workspace Panel
  - Position: rechts neben Sidebar, unter Header
  - Groesse: etwa 76 Prozent der Breite und rund 64 Prozent der Bildhoehe
  - Inhalt: Stage-07-Titel, Position Box, drei Hauptspalten, Prozesskette
  - Prioritaet: dominant
  - Farbe/Border: cyan Aussenborder, dunkler Hintergrund
  - Ausrichtung: vertikal

- Top Bar / Title Row im Active Workspace
  - Position: ganz oben im Active Workspace
  - Inhalt: `ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS`
  - Farbe: cyan
  - Prioritaet: hoch

- Current Position Box
  - Position: direkt unter Title Row
  - Groesse: schmaler horizontaler Balken
  - Inhalt: Current Step, Operator focus, Render paused, Last passed, Next technical step
  - Farbe/Border: graue Innenlinien, cyan Titel, gelbe und gruene Statuswerte
  - Ausrichtung: horizontale Spalten

- Block A: `A) CONTRACT INPUTS`
  - Position: linke Spalte unter Current Position
  - Groesse: schmal, etwa 20 Prozent der Active-Workspace-Breite
  - Inhalt: sechs Input-Statuszeilen
  - Prioritaet: mittel; liefert Voraussetzungen
  - Farbe/Border: cyan Border, gruen betonte Werte
  - Ausrichtung: vertikal
  - Cards: keine Untercards, sondern getrennte Zeilen

- Block B: `B) SCENE CONTRACTS`
  - Position: mittlere Spalte
  - Groesse: groesster Block, etwa 50 Prozent der Active-Workspace-Breite
  - Inhalt: drei Scene Contract Cards
  - Prioritaet: hoechste in Stage 07
  - Farbe/Border: cyan Border; Scene-Status gruen/gelb/grau
  - Ausrichtung: drei Cards untereinander
  - Cards: Scene 01, Scene 02, Scene 03

- Block C: `C) OUTPUT PREVIEW / READINESS`
  - Position: rechte Spalte
  - Groesse: etwa 28 Prozent der Active-Workspace-Breite
  - Inhalt: JSON Preview und Readiness-Liste
  - Prioritaet: mittel bis hoch; erklaert Handoff an Stage 08
  - Farbe/Border: cyan Border, gruenes JSON, gruene Checks
  - Ausrichtung: vertikal

- Bottom Process Chain im Active Workspace
  - Position: unter den drei Hauptspalten
  - Groesse: voller Workspace-Breite, niedrige Hoehe
  - Inhalt: Director -> Creative OS -> Strategy -> Beat/Hook Planner -> Creative Judge -> Scene Contracts -> Image Prompt Compiler
  - Prioritaet: Workflow-Kontext
  - Farbe/Border: einzelne Karten mit grauen/cyan Borders, aktive Scene-Contracts-Karte gelb
  - Ausrichtung: horizontal nebeneinander

- Bottom Band
  - Position: ganz unten unter Workspace/Sidebar
  - Inhalt: Skill Health, Artifacts, Issues, Next
  - Prioritaet: sekundaerer Status
  - Farbe/Border: cyan Borders
  - Ausrichtung: vier Panels nebeneinander

## 4. Farbsystem aus dem Bild

- Hintergrund: fast schwarz mit leicht gruenlich-blauem Glow.
- Hauptpanel: dunkles Schwarz/Blau.
- Innere Panels: ebenfalls dunkel, minimal heller als Hintergrund.
- Borders: cyan/tuerkis fuer Standard-Panels und innere Boxen.
- Header-Border: neon-gruen.
- Aktive Elemente: gelb/orange, sichtbar bei `07 Scene Contracts`, Prozessketten-Karte und naechstem Schritt.
- Ready/done Elemente: neon-gruen, sichtbar bei Checks, `approved`, `aligned`, `contract ready`.
- Pending/demo Elemente: grau fuer queued/offen, gelb fuer drafting/in progress.
- Warn-/Risiko-Elemente: im Stage-07-Bild keine roten Fehler sichtbar; Risiko-Policy ist gruen/approved, drafting ist gelb.
- Haupttext: hellgrau bis weiss.
- Labels: cyan fuer Feldnamen in Scene Contracts; gruen fuer Contract Inputs.
- Werte: gruen fuer approved/ready, weiss/grau fuer descriptive Werte, gelb fuer drafting/active.

## 5. Text-Hierarchie

- Groesster Titel: global `CONTENT MASCHINE LIVE`.
- Groesster Stage-Titel: `ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS`.
- Section-Titel: `CURRENT POSITION AND PIPELINE PATH`, `A) CONTRACT INPUTS`, `B) SCENE CONTRACTS`, `C) OUTPUT PREVIEW / READINESS`.
- Card-Titel: Scene IDs `scene_01`, `scene_02`, `scene_03` innerhalb der Scene Cards.
- Labels: `visual anchor`, `environment`, `action`, `camera`, `lighting`, `allowed visuals`, `forbidden visuals`.
- Werte: `misty canopy reveal`, `sunrise jungle canopy`, `slow opening reveal through leaves`, etc.
- Kleine Hinweise: Statuswerte rechts in Scene Cards, Bottom `Operator` und `Technical`.
- Status-Texte: `contract ready`, `contract drafting`, `queued`, `approved`, `aligned`, `none blocking`.

## 6. Inhaltliche Bloecke

- `CURRENT POSITION AND PIPELINE PATH`
  - Zweck: Stage-Kontext und Handoff zeigen.
  - Sichtbare Felder: Current Step, Operator focus, Render paused, Last passed, Next technical step.
  - Nicht drin: keine Scene-Felder, keine JSON-Daten.
  - Dominanz: mittel, horizontaler Kontextbalken.

- `A) CONTRACT INPUTS`
  - Zweck: Voraussetzungen fuer Scene Contracts anzeigen.
  - Sichtbare Felder: Creative Strategy, Beat/Hook Plan, Creative Judge, Mode/Style, Risk Policy, Scene Count.
  - Nicht drin: keine einzelnen Scene Details, keine Prompttexte.
  - Dominanz: klein bis mittel.

- `B) SCENE CONTRACTS`
  - Zweck: konkrete Scene-Regeln und visuelle Vorgaben pro Scene zeigen.
  - Sichtbare Felder: scene_id, visual anchor, environment, action, camera, lighting, allowed visuals, forbidden visuals, status.
  - Nicht drin: keine finalen Image Prompts, keine Backend-Werte, keine echten Renderbilder.
  - Dominanz: groesster Hauptblock.

- `C) OUTPUT PREVIEW / READINESS`
  - Zweck: `scene_contracts.json` und Bereitschaft fuer Stage 08 zeigen.
  - Sichtbare Felder: JSON file, scene_count, continuity_rule, text_policy, ready_for, strategy merged, hook merged, scene rules locked, prompt compilation next.
  - Nicht drin: keine Prompt Compiler Branches, keine Image Generation Cards.
  - Dominanz: mittel.

- Process Chain
  - Zweck: Stage 07 im Workflow einordnen.
  - Sichtbare Felder: Step-Namen.
  - Nicht drin: keine Detaildaten.
  - Dominanz: niedrig bis mittel.

- Bottom Panels
  - Zweck: globaler Status, Artefakte, Issues, Next.
  - Sichtbare Felder: Skill Health, Artifacts, Issues, Next.
  - Nicht drin: keine Stage-Hauptstruktur.
  - Dominanz: sekundaer.

## 7. Was Stage 07 NICHT zeigen darf

- Keine finalen Image Prompts.
- Keine echten Bilder.
- Keine Image Generation Cards.
- Keine riesigen JSON-Bloecke.
- Keine Prompt Compiler Struktur.
- Keine Video/Audio/Music Compiler Branches.
- Keine Stage-08/09-Inhalte vorwegnehmen.
- Keine Backend-/Render-Fortschritte.
- Keine LTX-, Audio- oder Music-Ausgaben.
- Keine Prompt Payload Preview ausser der kleinen `scene_contracts.json` Output Preview.
- Keine zusammengequetschte Ein-Zeilen-Zusammenfassung aller Scenes.

## 8. Umsetzung als Textual-Panel

- Bestehende Helper:
  - `_section_box` fuer `CURRENT POSITION`, `CONTRACT INPUTS`, `SCENE CONTRACTS`, `OUTPUT PREVIEW / READINESS`.
  - `_position_grid` fuer die obere Positionsleiste.
  - `_label_value_line` fuer kurze Label/Wert-Zeilen.
  - `_plain_line` fuer JSON Preview und Prozesshinweise.
  - `_truncate_text` indirekt ueber Box-Helper zur Ueberlaufvermeidung.
- Render-Reihenfolge:
  1. Stage Header `ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS`.
  2. `CURRENT POSITION AND PIPELINE PATH`.
  3. `A) CONTRACT INPUTS`.
  4. `B) SCENE CONTRACTS`.
  5. `C) OUTPUT PREVIEW / READINESS`.
  6. Prozesskette oder kompakte Handoff-Zeile.
- Exakte Labels, die sichtbar bleiben sollten:
  - `Creative Strategy`
  - `Beat / Hook Plan`
  - `Creative Judge`
  - `Mode / Style`
  - `Risk Policy`
  - `Scene Count`
  - `visual anchor`
  - `environment`
  - `action`
  - `camera`
  - `lighting`
  - `allowed visuals`
  - `forbidden visuals`
  - `status`
  - `Output Preview (JSON)`
  - `ready_for`
- Text kurz halten:
  - Scene-Felder pro Zeile kurz und gekuerzt.
  - Allowed/forbidden visuals maximal wenige Items.
  - JSON Preview nur 5-7 Zeilen.
- Demo/Fixture darf sein:
  - Beispielwerte fuer Scene 01-03, wenn als Fixture/Demo-Kontext erkennbar.
  - Status `contract ready`, `contract drafting`, `queued`, wenn als Fixture oder aus State ableitbar.
- Spaeter aus State:
  - Scene Contracts, Beat/Hook Plan, Judge-Status, Mode/Style, Risk Policy, Artifacts.
- Besser statisch:
  - Prozesskette, sofern keine eigene Navigationslogik existiert.
  - Labels und Grundstruktur der drei Hauptbloecke.

## 9. Abgleich mit aktuellem Stage-07-Panel

### Gelesener Code

- Es wurde nur `/workspace/agent_core/creative_os/cockpit/panels/active_workspace_panel.py` gelesen.
- Aktueller Stage-07-Code ist `_scene_contracts_workspace`.
- Der aktuelle Code ruft `_stage_detail_workspace(...)` mit Section `SCENE CONTRACTS` auf.
- Aktuell werden nur zusammenfassende Felder gerendert:
  - `Scenes`
  - `Environment`
  - `Action`
  - `Camera`
  - `Lighting`
  - `Risk Controls`
  - Expected Output `scene_contracts.json`
  - Next Action `Inspect scene requirements before image prompt compilation.`

### Was aktueller Stage-07-Build falsch macht

- Er bildet nicht die drei Hauptspalten `A) CONTRACT INPUTS`, `B) SCENE CONTRACTS`, `C) OUTPUT PREVIEW / READINESS` ab.
- Er zeigt keine Current-Position-Leiste im Bildstil.
- Er zeigt keine drei getrennten Scene Cards.
- Er zeigt keine `visual anchor`, `allowed visuals` oder `forbidden visuals`.
- Er zeigt keinen Status pro Scene (`contract ready`, `contract drafting`, `queued`).
- Er zeigt keine Output Preview mit `scene_contracts.json`, `continuity_rule`, `text_policy`, `ready_for`.
- Er zeigt keine Readiness-Checkliste (`strategy merged`, `hook merged`, `scene rules locked`, `prompt compilation next`).
- Er zeigt keine Prozesskette mit aktivem `Scene Contracts`.

### Welche Bloecke fehlen

- `A) CONTRACT INPUTS`
- `B) SCENE CONTRACTS` als Card-Liste, nicht nur Summary
- `C) OUTPUT PREVIEW / READINESS`
- Current Position im spezifischen Stage-07-Bildstil
- Prozesskette/Handoff-Leiste

### Welche Bloecke zu viel sind

- Der generische `_stage_detail_workspace`-Block mit `Current Status`, `Purpose`, `Expected Output`, `Next Action` entspricht nicht dem Bild.
- Die generische `ARTIFACTS` Box aus `_stage_detail_workspace` ist im Active Workspace des Bildes nicht an dieser Stelle sichtbar; Artefakte liegen unten im globalen Bottom-Band.

### Welche Reihenfolge falsch ist

- Aktuell kommt eine generische Detailbox und danach Artifacts.
- Im Bild kommt zuerst Position, dann drei Hauptspalten, dann Prozesskette.

### Welche Bezeichnungen falsch sind

- Aktuell: `SCENE CONTRACTS`, `Current Status`, `Purpose`, `Expected Output`, `Next Action`, `ARTIFACTS`.
- Bild: `CURRENT POSITION AND PIPELINE PATH`, `A) CONTRACT INPUTS`, `B) SCENE CONTRACTS`, `C) OUTPUT PREVIEW / READINESS`.

### Welche Struktur nicht dem Bild entspricht

- Aktuell ist Stage 07 ein allgemeines Detail-Panel.
- Das Bild verlangt ein operatives Stage-Workspace-Layout mit drei nebeneinander liegenden Arbeitsbloecken und drei Scene Cards.

## 10. Minimaler Korrekturplan

- Schritt 1: `_scene_contracts_workspace` aus dem generischen `_stage_detail_workspace` herausloesen und eine eigene Stage-07-Workspace-Struktur mit Stage Header und Position Box bauen.
- Schritt 2: Drei Hauptbloecke rendern: `A) CONTRACT INPUTS`, `B) SCENE CONTRACTS`, `C) OUTPUT PREVIEW / READINESS`.
- Schritt 3: In `B) SCENE CONTRACTS` drei getrennte Scene Cards mit visual anchor, environment, action, camera, lighting, allowed visuals, forbidden visuals und status bauen.
- Schritt 4: Output Preview als kurze `scene_contracts.json` JSON-Vorschau plus Readiness-Checkliste rendern.
- Schritt 5: Tests nur fuer Stage 07 aktualisieren/ergaenzen und Stage 08/09-Erhalt absichern.
