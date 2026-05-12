# 03 Style

## Zweck

Dieses Panel definiert den visuellen oder narrativen Stil, der fuer Image-, Keyframe- und Mapping-Ergebnisse verwendet wird.

## Muss sichtbar sein

- Aktiver Style
- Style-Auswahl oder Style-Parameter
- Hinweise auf fehlende oder widerspruechliche Style-Werte

## Verhalten

- Style-Auswahl beeinflusst Prompts, Keyframes und Compiler-Ausgaben.
- Style kann aus Vorlagen oder manuellen Parametern entstehen.
- Aenderungen muessen in downstream Panels sichtbar werden.

## Inputs

- `activePipelineId`
- `activeMode`
- Style-Presets
- Manuelle Style-Felder

## Outputs

- `styleConfig`
- Style-Zusammenfassung fuer Mapping und Generation
- Validierungsstatus

## UI-Regeln

- Style nicht als reine Dekoration behandeln.
- Presets und manuelle Anpassungen klar trennen.
- Aktive Werte kompakt zusammenfassen.

## Fehler-/Empty-State

- Empty: kein Style gewaehlt, mit neutralem Default oder klarer Aufforderung.
- Error: unvollstaendige oder widerspruechliche Style-Konfiguration.

## Akzeptanzkriterien

- Der gewaehlte Style ist sichtbar und nachvollziehbar.
- Style-Werte fliessen logisch in nachfolgende Panels.
- Fehlende Style-Daten erzeugen keine kaputten States.
