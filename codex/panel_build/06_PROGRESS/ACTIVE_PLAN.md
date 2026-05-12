# Active Plan

Stand: 2026-05-11, Active-Workspace-Scroll-Pass abgeschlossen.

## Aktueller Build-Stand

Der enge Build-Schritt wurde korrigiert: Stage `07 Scene Contracts` ist jetzt ein eigenstaendiges Active-Workspace-Panel und folgt der verifizierten Bildanalyse unter `01_VISUAL_REFERENCES/pipeline_panels/panel_07_scene_contracts/visual_analysis_verified.md`.

Nach dem ersten Visual-Correction-Pass wurde das Layout noch einmal stabilisiert:

- A/B/C-Spalten haben jetzt fixe, gemessene Breiten.
- Jede Stage-07-Box-Zeile wird auf 132 Zeichen begrenzt.
- Current Position nutzt zwei kompakte Reihen mit kurzen Labels.
- Lange Werte werden vor der Border gekuerzt.
- Sidebar wurde minimal verdichtet, damit der Workspace visuell dominanter bleibt.

Zusaetzlich wurde der Active Workspace als generischer Scroll-Host umgesetzt:

- Der aeussere `ACTIVE WORKSPACE` Rahmen bleibt erhalten.
- `#workspace` ist jetzt ein `ScrollableContainer`.
- Der renderbare Stage-Inhalt liegt in `#workspace-content`.
- Alle Stage-Views `00` bis `15` laufen durch denselben Scrollbereich.
- Der Scroll-Host ist nicht fokussierbar, damit Stage-Navigation und Stage-09-Toggle erhalten bleiben.

Gebaut wurde nur im engen Scope:

- Active Workspace Stage `07`
- `ACTIVE WORKSPACE / STAGE 07: SCENE CONTRACTS`
- `CURRENT POSITION AND PIPELINE PATH`
- `A) CONTRACT INPUTS`
- `B) SCENE CONTRACTS`
- `C) OUTPUT PREVIEW / READINESS`
- `HANDOFF PATH`
- Cockpit-Tests fuer Stage 07 sowie Stage-08-/Stage-09-Erhalt aktualisiert

Nicht geaendert:

- keine Stage-ID-Neuordnung
- kein Cockpit-Redesign
- keine neue Pipeline-Integration
- keine Render-, API- oder n8n-Arbeit
- Header und Bottom Panels unveraendert
- System Status und Pipeline Map nur minimal in Hoehe/Padding verdichtet
- Stage `08` Prompt Compiler bleibt erhalten
- Stage `09` Image Cards bleiben erhalten

## Was Stage 07 jetzt sichtbar macht

- Stage `07` ist nicht mehr das generische Detailpanel.
- Current Position zeigt Scene Contracts, Creative-Judge-Vorstufe und Stage-08-Handoff.
- Contract Inputs zeigen Creative Strategy, Beat/Hook, Creative Judge, Mode/Style, Risk Policy und Artifact Policy.
- Drei Scene-Contract-Cards zeigen `scene_01`, `scene_02`, `scene_03` mit Visual Anchor, Environment, Action, Camera/Lighting, erlaubten und verbotenen Visuals.
- Output Preview zeigt kompakt `scene_contracts.json` und `ready_for: image_prompt_compiler`.
- Handoff Path erklaert den Uebergang von Creative Judge zu Stage `08 Image Prompt Compiler`.

Fehlende Details bleiben `missing` oder `not_checked`. Fixture-/Demo-Szenenwerte werden als Demo-Kontext dargestellt, nicht als echte Livewerte. Stage 07 zeigt keine finalen Image Prompts, keine echten Bilder, keine Image Generation Cards und keine Compiler-Branches.

## Teststatus

- `python3 -m unittest /workspace/tests/test_creative_os_cockpit.py -v`: gruen, 19 Tests
- `python3 -m unittest /workspace/tests/test_creative_os_status.py -v`: gruen, 13 Tests

## Naechster enger Build-Schritt

Stage `07` im laufenden Textual-Cockpit visuell gegen `reference.png` pruefen.
