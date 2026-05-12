# Open Issues

Stand: 2026-05-11, nach Stage-07-Visual-Correction.

## Offene Fragen

- Stage `08` ist jetzt deutlich naeher an `panel_08_prompt_compiler/reference.png`, bleibt aber wegen Textual-Einschraenkung ohne echte Thumbnails und ohne Stage-ID-Unterpunkte `08.1` bis `08.4`.
- Die Referenz fuer Stage `08` zeigt `08.1 Image Compiler` innerhalb eines Prompt-Compiler-Familienpanels. Aktuell wird das ohne Stage-ID-Neuordnung als `Active Branch: Image Compiler` und `COMPILER FAMILY / BRANCHES` abgebildet.
- Die Referenz fuer Stage `09` zeigt Stage `06/12 TextToImage`, waehrend die aktuelle Registry Stage `09/15 Image / Keyframe Generation` nutzt. Fachliche Zuordnung ist sicher, aber Nummerierung muss bei UI-Korrekturen bewusst behandelt werden.
- Die Spezifikation definiert Panels `01` bis `09`, der Code definiert Stages `00` bis `15`. Es ist weiterhin zu entscheiden, ob die Spezifikation exakt in Stage IDs uebersetzt werden soll oder als fachliche Checkliste fuer vorhandene Stages gilt.
- Spec `02 Mode` und Spec `03 Style` sind aktuell in Stage `02 Mode & Style` zusammengelegt. Eine Trennung wuerde Tests und Stage Map beeinflussen.
- Spec `05 Models` existiert nicht als eigenes Stage-Panel. Stage `08` und `09` zeigen jetzt Backend/Rules/Handoff, aber noch keine vollstaendige Modellrollen-Uebersicht.
- Spec `06 Mapping` existiert nicht als eigenes Stage-Panel. Stage `08` zeigt Scene Contract Source pro Prompt Card, aber der volle Datenfluss ist noch nicht als eigene Mapping-Ansicht sichtbar.
- `prompt_audit.json` fehlt in der aktuellen Fixture; Stage `08` und `09` zeigen deshalb korrekt `prompt_audit.json missing`.
- Artifact Policy ist nur sichtbar, wenn ein Audit-Feld vorhanden ist; sonst bleiben Regeln als `missing` oder `not_checked` markiert.
- Scene Contracts in der Fixture enthalten aktuell nur `scene_id`; Environment, Action, Camera und Lighting bleiben deshalb korrekt `not_checked`.
- Stage `07` nutzt fuer die drei Scene Cards Demo-/Fixture-Ergaenzungen, solange echte Contract-Felder fehlen. Diese Werte sind UI-Kontext, keine Live-Run-Werte.
- Stage `07` ist jetzt als explizite Mapping-Quelle fuer Stage `08` sichtbar, aber die langfristige echte Datenquelle fuer vollstaendige Scene Contracts muss noch definiert werden.

## Risiken

- Stage IDs umzubauen waere ein groesserer Eingriff, weil Tests Stage `09` als Image/Keyframe Generation erwarten.
- Neue eigene Panels fuer Models oder Mapping koennten die vorhandene Cockpit-Architektur unnoetig vergroessern.
- Wenn Mapping nur implizit bleibt, kann der User den Datenfluss von Pipeline/Mode/Style/Hook/Models zu Keyframes schwer nachvollziehen.
- Stage `09` hat Keyboard-Interaktion fuer Image Jobs; weitere Aenderungen muessen Navigation `j/k/space/enter` weiter schuetzen.
- `state_adapter.py` liest je nach Run-Typ unterschiedliche Artefakte. Neue Felder muessen fuer Fixture, Creative-OS, Agent-Core, Missing und Unknown Run robust bleiben.

## Keine aktuellen Blocker

- Stage-07-Scene-Contracts-Pass ist gebaut.
- Geforderte und optionale Tests sind gruen.
