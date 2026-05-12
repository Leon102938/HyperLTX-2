# Error, Empty, Loading States

## Empty States

Empty States muessen einen Zweck haben. Sie zeigen:

- was fehlt
- warum es fehlt
- welcher naechste Schritt sinnvoll ist

## Loading States

Loading States muessen klar machen:

- welcher Prozess laeuft
- ob einzelne Panels oder der ganze Workflow blockiert sind
- ob Abbrechen, Retry oder Weiterarbeiten moeglich ist

## Error States

Error States muessen enthalten:

- kurze Ursache
- betroffener Workflow-Schritt
- konkrete Aktion zur Behebung oder Diagnose

## Vermeiden

- Keine generischen "Something went wrong"-Meldungen ohne Kontext.
- Keine leeren Flaechen ohne Hinweis.
- Keine endlosen Loader ohne Timeout- oder Fehlerpfad.
