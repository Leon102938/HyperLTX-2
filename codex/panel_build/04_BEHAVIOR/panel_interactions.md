# Panel Interactions

Diese Datei beschreibt, wie Panels spaeter miteinander interagieren sollen.

## Auswahl und Propagation

- Panel `01` setzt den Pipeline-Kontext.
- Panel `02` setzt den Mode-Kontext.
- Panels `03` bis `05` liefern kreative und technische Parameter.
- Panel `06` verbindet Parameter zu nutzbaren Mappings.
- Panel `07` erzeugt Keyframes aus validierten Inputs.
- Panel `08` bereitet Compiler-Daten vor.
- Panel `09` zeigt den aktiven Arbeitsstand.

## Sichtbare Rueckmeldungen

- Jede Aenderung mit downstream Auswirkung braucht sichtbares Feedback.
- Gesperrte Aktionen muessen den Grund zeigen.
- Fehler sollen auf das Panel zurueckverweisen, in dem sie behoben werden koennen.

## Bedienregeln

- Primaere Aktionen pro Panel begrenzen.
- Sekundaere Aktionen kontextnah platzieren.
- Navigation und Korrekturwege duerfen nicht versteckt sein.
