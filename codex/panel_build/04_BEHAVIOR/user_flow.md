# User Flow

Der spaetere finale Workflow folgt dieser Reihenfolge:

1. Pipeline auswaehlen.
2. Mode bestimmen.
3. Style konfigurieren.
4. Hook festlegen.
5. Models auswaehlen.
6. Mapping pruefen oder erstellen.
7. Keyframes generieren.
8. Compiler Workspace vorbereiten.
9. Active Workspace pruefen und weiterarbeiten.

## Grundprinzip

Jedes Panel soll den naechsten sinnvollen Schritt sichtbar machen. Wenn ein Schritt noch nicht moeglich ist, muss das Panel erklaeren, welche Voraussetzung fehlt.

## Rueckwaertsbewegung

Der User darf zu frueheren Panels zurueckkehren. Wenn eine Aenderung downstream Ergebnisse invalidiert, muss das sichtbar markiert werden.

## Abschluss

Der Workflow ist erst abgeschlossen, wenn der Active Workspace einen konsistenten Stand aus Pipeline, Mode, Style, Hook, Models, Mapping, Keyframes und Compiler-Daten zeigt.
