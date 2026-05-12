# State Logic

Diese Datei beschreibt die Startlogik fuer spaetere State-Entscheidungen.

## Zentrale State-Abhaengigkeiten

- Pipeline ist die oberste Abhaengigkeit.
- Mode haengt von Pipeline ab.
- Style und Hook haengen von Pipeline und Mode ab.
- Models haengen von Pipeline und Mode ab.
- Mapping haengt von Pipeline, Mode, Style, Hook und Models ab.
- Keyframes haengen von validiertem Mapping und Models ab.
- Compiler Workspace haengt von Keyframes und Mapping ab.
- Active Workspace haengt vom Compiler- und Ergebnisstatus ab.

## Invalidation

Wenn ein frueherer State geaendert wird, muessen downstream Daten als stale, invalid oder review-needed markiert werden, falls sie nicht mehr sicher passen.

## Mindest-States pro Bereich

- empty
- partial
- valid
- stale
- loading
- error
- ready

## Dokumentationspflicht

Spaetere State-Aenderungen muessen in `06_PROGRESS/CHANGELOG.md` dokumentiert werden, wenn sie Verhalten oder Datenfluss beeinflussen.
