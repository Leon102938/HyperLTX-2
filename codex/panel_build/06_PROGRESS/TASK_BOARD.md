# Task Board

Stand: 2026-05-11, nach Active-Workspace-Scroll-Pass.

## Done

- [x] Aufgabenordner `panel_build` angelegt.
- [x] Startstruktur fuer Regeln, Specs, Verhalten, Tests und Progress angelegt.
- [x] Bestehende Cockpit-Dateien identifiziert.
- [x] `app.py`, `layout.py`, `stage_registry.py`, `panel_registry.py`, `state_adapter.py`, `panels/` und `tests/test_creative_os_cockpit.py` gelesen.
- [x] Panel-Spezifikationen `01` bis `09` gegen vorhandene Stage-Struktur abgeglichen.
- [x] Placeholder und vorbereitete Stage-Panels grob getrennt.
- [x] Gap-Analyse erstellt.
- [x] Stage `09 Image / Keyframe Generation` gegen Spec `07_keyframe_generation` nachgeschaerft.
- [x] Stage `09` um klare Readiness-/Input-Zusammenfassung ergaenzt.
- [x] Backend, Run Type, Session Mode und Keyframe Summary in Stage `09` sichtbar gemacht.
- [x] Bestehende Image Cards und Keyboard-Toggle per Test abgesichert.
- [x] Stage `08 Image Prompt Compiler` als direkte Compiler-/Prompt-Vorstufe fuer Stage `09` nachgeschaerft.
- [x] Stage `08` zeigt Compiler Readiness, Compiler Family, Scene Prompt Cards, Model Rules und Output/Next.
- [x] Stage `08` zeigt keine echten Bilder und bleibt read-only.
- [x] Stage `08` anhand `panel_08_prompt_compiler/reference.png` und `visual_analysis.md` korrigiert.
- [x] Stage `08` zeigt jetzt Scope-Leiste, dominante Image-Compiler-Zone, Scene Prompt Summaries, Final Prompt Payload und Branch-Kontext.
- [x] Stage `07 Scene Contracts` anhand `visual_analysis_verified.md` korrigiert.
- [x] Stage `07` vom generischen Detailpanel auf eine eigene Scene-Contracts-Struktur umgestellt.
- [x] Stage `07` zeigt Current Position, Contract Inputs, Scene Contract Cards, Output Preview und Handoff Path.
- [x] Stage `07` vermeidet finale Image Prompts, echte Bilder, Image Generation Cards und Compiler-Branches.
- [x] Stage `07` Layout nachkorrigiert: stabile 3-Spalten-Boxen, flacher Current-Position-Block, harte Truncation gegen Border-Ueberlauf.
- [x] Linke Sidebar minimal verdichtet: System Status niedriger, Pipeline Map kompakter gepolstert, Workspace breiter.
- [x] Active Workspace als generischen Scroll-Host umgesetzt.
- [x] `#workspace` scrollt alle Stage-Views `00` bis `15` ueber `#workspace-content`.
- [x] Stage-Navigation und Stage-09-Image-Card-Toggle nach Scroll-Einbau per Test abgesichert.

## In Progress

- [ ] Kein aktiver Build-Durchlauf. Active-Workspace-Scroll-Pass ist abgeschlossen.

## Next Build Backlog

- [ ] Stage `07` im laufenden Textual-Cockpit visuell gegen `reference.png` pruefen.
- [ ] Stage `06 Creative Judge` anhand der Visual Analysis pruefen und bei Bedarf gezielt nachschaerfen.
- [ ] Mapping-Quelle zwischen Scene Contracts, Prompts und Keyframes noch klarer machen.
- [ ] Prompt Audit und Artifact Policy besser aus echten Artefakten lesen, falls verfuegbar.

## Later Backlog

- [ ] Stage `01 Pipeline wählen` gegen Spec `01_pipeline_selection` nachschaerfen.
- [ ] Stage `02 Mode & Style` gegen Specs `02_mode` und `03_style` nachschaerfen.
- [ ] Hook-Daten in Stage `04`/`05` konsistenter darstellen.
- [ ] Models als read-only Modellrollen in vorhandenen Stages sichtbar machen.
- [ ] Spec `09_active_workspace` als zusammenfassenden Workspace-Zustand definieren, ohne neue Architektur zu erzwingen.

## Blocked / Needs Decision

- [ ] Soll die Zielstruktur `01-09` exakt als Stage IDs umgesetzt werden oder als Spec-Mapping auf die bestehende Stage-Reihenfolge gelten?
- [ ] Welche Artefakte gelten langfristig als Quelle fuer `modelConfig` und `mappingConfig` im echten Run?
