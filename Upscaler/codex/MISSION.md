# MISSION.md

## Hauptmission
Baue einen headless Video-Agent-Core fuer RunPod.

## Zielbild
Das System soll Videojobs intern im Pod annehmen, planen, ausfuehren und Ergebnisse sauber speichern.
Dabei soll es lokale Modelle, lokale Skripte, lokale Tools und vorhandene lokale APIs nutzen koennen.

## Kernprinzipien
- modularer Aufbau
- Agent-Core vor API und n8n
- kein ComfyUI-Zwang
- klare Trennung von Planung, State, Adaptern und Ausfuehrung
- spaeter API-faehig
- spaeter n8n-faehig
- keine unnoetige GUI
- keine Demo-only-Architektur

## Nicht das Ziel
- Monolith ohne klare Schnittstellen
- aufgeblasener Multi-Agent-Schwarm ohne Nutzen
- starre Pipeline ohne Job-Konfigurierbarkeit
- stillschweigende Zielverschiebung

## Erfolg bedeutet
- ein Job kann sauber angenommen, validiert, geplant, ausgefuehrt und persistiert werden
- vorhandene Backends koennen ueber Adapter angebunden werden
- neue Modi wie TTS aus, Musik statt TTS, Storyboard aus, Hochformat oder anderer Renderpfad koennen spaeter ergaenzt werden, ohne den Kern zu zerlegen
