# 02 Mode

## Zweck

Dieses Panel bestimmt den Arbeitsmodus der aktiven Pipeline, zum Beispiel Image-Erstellung, Keyframe-Planung, Mapping oder Compiler-Vorbereitung.

## Muss sichtbar sein

- Aktiver Mode
- Verfuegbare Modes
- Kurzer Status, ob der Mode vollstaendig konfiguriert ist

## Verhalten

- Mode-Auswahl beeinflusst sichtbare Folgeoptionen.
- Unpassende Optionen in spaeteren Panels werden deaktiviert oder erklaert.
- Mode-Wechsel darf bestehende Eingaben nicht stillschweigend zerstoeren.

## Inputs

- `activePipelineId`
- Liste erlaubter Modes
- Bestehende Mode-Konfiguration

## Outputs

- `activeMode`
- Validierungsstatus fuer Folgepanels
- Hinweise auf erforderliche Inputs

## UI-Regeln

- Modes als klare, schnelle Auswahl darstellen.
- Aktiven Mode visuell stabil markieren.
- Keine langen Erklaertexte in der Panel-Flaeche.

## Fehler-/Empty-State

- Empty: Pipeline gewaehlt, aber keine Modes geladen.
- Error: Mode-Konfiguration ist ungueltig oder inkompatibel mit der Pipeline.

## Akzeptanzkriterien

- Der aktive Mode ist verstaendlich.
- Folgepanels reagieren nachvollziehbar auf Mode-Aenderungen.
- Ungueltige Kombinationen sind sichtbar blockiert.
