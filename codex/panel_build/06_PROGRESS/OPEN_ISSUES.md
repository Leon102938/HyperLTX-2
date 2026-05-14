# Open Issues

Stand: 2026-05-14, nach Phase-1-Live-Orchestrator V3 Stage 00-09.

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
- Phase-1-Run kann Stage `00` bis `08` lokal erzeugen und Stage `09` als Image-Job-Manifest anlegen.
- Live-Run kann Stage `00` bis `09` sequenziell starten und `live_status.json`/`stage_events.jsonl` schreiben; es ist weiterhin keine Stage `10+` Runtime gebaut.
- `--open-cockpit` ist bewusst entschärft: kein Textual-Background-Prozess im selben TTY. Der CLI gibt zwei sichere Terminal-Befehle aus.
- Disabled/missing Backend darf Stage `09` nie gruen machen; V2 setzt Stage `09=error` und completed nur `00` bis `08`.
- Stage `10` bis `15` bleiben fuer Phase-1-Runs out-of-scope/not built und duerfen nie gruen werden.
- Stage `09` Manifest wird waehrend echter Bildgenerierung aktualisiert; trotzdem bleibt die UI read-only und startet keine eigenen Backend-Aktionen.
- Echte Stage-09-Bilder entstehen nur, wenn das lokale Z-Image-HTTP-Backend erreichbar ist. Fehlt es, bleibt der Run korrekt auf `phase1_paused_missing_image_backend`.
- Stage `09` zeigt im Terminal keine echten Bild-Thumbnails, sondern echte Preview-Pfade plus Datei-Status. Visuelle Bildinspektion laeuft ueber die PNG-Dateien oder `keyframe_gallery.html`.
- Stage `10+` ist nach fertiger Phase 1 ausdruecklich `not built yet` und darf nicht als naechster Live-Schritt erscheinen.
- Stage `09` hat einen engen Retry/Resume-Befehl: `creative-os retry-keyframes`. Er ist kein genereller Rebuild und schreibt Stage `00` bis `08` nicht neu.
- Naechster fachlicher Engpass bleibt ein echtes Keyframe-Review/Gate nach Stage `09`; Stage `10+`, LTX und API/n8n bleiben ausserhalb dieses Schritts.

## Risiken

- Dependency-Drift auf Textual 8.x wuerde Scroll-/Navigation-Performance erneut gefaehrden.
- Stage-ID-Umbauten waeren ein groesserer Eingriff, weil Tests und Navigation die bestehende `00` bis `15` Registry erwarten.
- Weitere Modell-/Mapping-Panels ohne echte Datenquelle wuerden die UI wieder generisch wirken lassen.
- Stage `09` hat Keyboard-Interaktion fuer Image Jobs; weitere Aenderungen muessen `j/k/space/enter` und Pfeilnavigation weiter schuetzen.
- Live-State kann in sehr schnellen No-Image-Smokes alle Stages im selben Sekunden-Timestamp abschliessen; die Event-Reihenfolge in `stage_events.jsonl` ist dann die belastbare Reihenfolge.
- `--stage-delay-seconds` ist nur Debug-Sichtbarkeit; Produktionsdefault bleibt `0`.
- `keyframe_contact_sheet.png` wird nicht erzeugt; aktuell gibt es nur die HTML-Gallery ohne neue Dependencies.
- Es gibt noch kein Keyframe-Review/Gate nach Stage `09`; das ist der naechste sinnvolle Schritt vor Stage `10+`.

## Keine aktuellen Blocker

- Stage `00` bis `09` rendern im Fixture-/Phase-1-Smoke.
- Stage `08` hat weiterhin einen sichtbaren, geschlossenen Image-Compiler-Hauptbereich.
- Cockpit- und Status-Tests sind gruen.
- Live-Smoke `live-smoke-20260514` ist vorhanden und Cockpit/Status-Tests sind gruen.
- V2-Smokes `live-v2-smoke-20260514` und `live-v2-smoke-images-20260514` sind vorhanden.
- V3-Smokes `live-v3-smoke-noimages-20260514` und `live-v3-smoke-images-20260514` sind vorhanden.
