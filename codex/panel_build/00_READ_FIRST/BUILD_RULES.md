# Build Rules

Diese Regeln gelten fuer jeden spaeteren Panel-Build-Durchlauf.

1. Erst lesen, dann aendern.
   - Relevante vorhandene Panels, Komponenten, State-Dateien und Tests muessen vor jeder Aenderung untersucht werden.
   - Keine Annahmen ueber die App-Struktur treffen, ohne Dateien zu pruefen.

2. Kein Blind-Refactor.
   - Nur Aenderungen machen, die direkt zur finalen Panel-Struktur beitragen.
   - Bestehende funktionierende Patterns beibehalten, wenn sie den Anforderungen nicht widersprechen.

3. Keine unnoetigen neuen Libraries.
   - Vorhandene Design-, State- und UI-Werkzeuge bevorzugen.
   - Neue Dependencies nur einbauen, wenn der konkrete Nutzen dokumentiert und alternativlos ist.

4. Maximal 3 Qualitaetsdurchlaeufe.
   - Durchlauf 1: Struktur und Sichtbarkeit herstellen.
   - Durchlauf 2: Verhalten, States und Datenfluss pruefen.
   - Durchlauf 3: visuelles Finetuning und Teststabilitaet.

5. Nach jedem Build-Durchlauf `ACCEPTANCE_CHECKLIST.md` pruefen.
   - Erfuellte Punkte markieren.
   - Offene Punkte in `../06_PROGRESS/OPEN_ISSUES.md` eintragen.

6. Fortschritt in `06_PROGRESS` aktualisieren.
   - `ACTIVE_PLAN.md`: aktueller Fokus und naechste Schritte
   - `TASK_BOARD.md`: Status der Aufgaben
   - `CHANGELOG.md`: konkrete Aenderungen pro Session
   - `OPEN_ISSUES.md`: Blocker, Risiken und offene Entscheidungen

7. Keine kaputten Zwischenstaende absichtlich stehen lassen.
   - Wenn ein Durchlauf Code aendert, muss der relevante Build/Test geprueft oder der Grund fuer nicht ausgefuehrte Tests dokumentiert werden.
