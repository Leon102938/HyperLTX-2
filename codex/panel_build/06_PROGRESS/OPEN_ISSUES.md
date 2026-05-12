# Open Issues

Stand: 2026-05-12, nach finalem Panel-Polish Stage 00-08.

## Offene Fragen

- Textual muss auf der stabilen 0.89.x-Linie bleiben. Runtime aktuell `0.89.1`, Pin `textual>=0.89,<1.0`.
- Stage `00` bis `08` sind stabile Terminal-Panels, aber keine perfekte Pixel-Kopie der Referenzbilder.
- Stage `08` bleibt ohne echte Thumbnails und ohne echte Unterstufen `08.1` bis `08.4`; die Referenzstruktur wird als stabile Textual-Ansicht abgebildet.
- Stage `05` zeigt bewusst keine Fake-Bilder oder Keyframes, sondern nur textbasierte Hook-/Beat-Kandidaten.
- Die Spezifikation definiert Panels `01` bis `09`, der Code definiert Stages `00` bis `15`. Die fachliche Zuordnung bleibt bewusst erhalten, ohne Stage-ID-Neuordnung.
- Spec `02 Mode` und Spec `03 Style` bleiben in Stage `02 Mode & Style` zusammengelegt.
- Spec `05 Models` und Spec `06 Mapping` existieren weiterhin nicht als eigene Stage-Panels.
- `prompt_audit.json` fehlt in der aktuellen Fixture; entsprechende Policy-/Audit-Felder bleiben deshalb read-only und knapp.
- Scene Contracts in der Fixture enthalten nur wenige echte Felder; Demo-/Fixture-Ergaenzungen sind UI-Kontext, keine Live-Run-Werte.

## Risiken

- Dependency-Drift auf Textual 8.x wuerde Scroll-/Navigation-Performance erneut gefaehrden.
- Stage-ID-Umbauten waeren ein groesserer Eingriff, weil Tests und Navigation die bestehende `00` bis `15` Registry erwarten.
- Weitere Modell-/Mapping-Panels ohne echte Datenquelle wuerden die UI wieder generisch wirken lassen.
- Stage `09` hat Keyboard-Interaktion fuer Image Jobs; weitere Aenderungen muessen `j/k/space/enter` und Pfeilnavigation weiter schuetzen.

## Keine aktuellen Blocker

- Stage `00` bis `08` rendern im Fixture-Smoke.
- Stage `08` hat weiterhin einen sichtbaren, geschlossenen Image-Compiler-Hauptbereich.
- Stage `09` wurde nicht umgebaut.
- Cockpit- und Status-Tests sind gruen.
