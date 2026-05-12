# Component Rules

Diese Datei beschreibt Startregeln fuer Komponenten im spaeteren Panel-Build.

## Wiederverwendung

- Vorhandene Komponenten zuerst suchen und verwenden.
- Neue Komponenten nur anlegen, wenn kein passendes bestehendes Pattern existiert.
- Props und State-Vertraege an bestehende Konventionen anpassen.

## Controls

- Auswahl: Segmented Controls, Tabs oder Selects je nach bestehendem Pattern.
- Binaere Optionen: Toggle oder Checkbox.
- Numerische Werte: Slider, Stepper oder Input.
- Aktionen: klare Buttons mit Icon, falls im bestehenden System ueblich.

## States

Jede relevante Komponente muss diese States sinnvoll darstellen koennen:

- default
- hover/focus
- selected/active
- disabled
- loading
- error
- empty

## Vermeiden

- Keine inhaltslosen Karten.
- Keine doppelten Controls mit gleicher Funktion.
- Keine Komponenten, die nur optisch existieren und keinen Workflow-Zweck haben.
