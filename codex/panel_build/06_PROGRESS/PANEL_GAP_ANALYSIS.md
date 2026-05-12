# Panel Gap Analysis

Stand: 2026-05-11. Diese Analyse vergleicht `03_PANEL_SPECS/01-09` mit dem aktuellen Cockpit unter `/workspace/agent_core/creative_os/cockpit`.

## Kurzfazit

Die Cockpit-Basis ist vorhanden: Stage Registry, Pipeline Map, Active Workspace Router, Stage-Detail-Renderer, Stage-09 Image Jobs und Tests existieren. Die Spezifikation `01-09` passt fachlich, aber nicht eins zu eins auf die aktuelle Stage-ID-Struktur. Der kleinste sinnvolle Build ist deshalb kein Architekturumbau, sondern ein gezieltes Nachschaerfen der vorhandenen Active-Workspace-Views, zuerst Stage `09`.

## Stage-/Spec-Abgleich

| Spec | Ziel laut Aufgabenpaket | Aktueller Code-Stand | Gap |
| --- | --- | --- | --- |
| 01 Pipeline Selection | Aktive Pipeline, verfuegbare Pipelines, Status | Stage `01 Pipeline wählen`, View `pipeline_select` vorhanden | Preview-only, keine echte Auswahl/Propagation |
| 02 Mode | Aktiver Mode, erlaubte Modes, Validierung | Teil von Stage `02 Mode & Style` | Nicht getrennt, wenig Validierungslogik sichtbar |
| 03 Style | Style-Konfiguration fuer Generation/Mapping | Teil von Stage `02 Mode & Style` | Style nur als Hint/Meta, keine Presets/Parameter |
| 04 Hook | Aktiver Hook und Varianten | Stage `04 Creative Strategy` und `05 Beat / Hook Planner` lesen Hook-Felder | Auf zwei Stages verteilt, keine klare Hook-Auswahl |
| 05 Models | Modellrollen, Verfuegbarkeit, Config | Kein eigenes Stage-Panel; Provider/Backend teils in Stage `08`/`09` sichtbar | Modellrollen fehlen als klare read-only Uebersicht |
| 06 Mapping | Datenfluss/Zuordnung Quellen zu Zielen | Kein eigenes Stage-Panel; Mapping implizit ueber Scene Contracts, Prompts, Artifacts | Datenfluss nicht explizit nachvollziehbar |
| 07 Keyframe Generation | Keyframe-Liste, Status, Generation, States | Stage `09 Image / Keyframe Generation` mit Image Job Cards vorhanden | Gute Basis, aber Inputs/Outputs/Voraussetzungen nach Spec noch nicht klar genug |
| 08 Compiler Workspace | Compiler-Eingaben, Compile-Status, Preview | Stage `08 Image Prompt Compiler` vorhanden | Als Compiler Workspace noch nicht klar benannt/verbunden |
| 09 Active Workspace | Aktiver Arbeitsstand nach vorherigen Panels | `active_workspace` ist Hauptpanel und Router | Kein eigener finaler Summary-State nach Specs 01-08 |

## Vorhandene echte Panels / Views

- `pipeline_map_panel.render`: echte Stage Map mit Auswahl-, Current-, Passed- und Pending-Markern.
- `active_workspace_panel.render`: echter Active-Workspace-Router.
- Stage `01`: eigene Pipeline-Select-View.
- Stage `02`: eigene Mode-&-Style-View.
- Stage `03`: eigene Skills-View.
- Stage `04` bis `08`: konkrete Detail-Views, keine reinen Placeholder mehr.
- Stage `09`: konkrete Image-/Keyframe-Job-View mit Cards, Auswahl, Expansion, Status und Progress.
- Stage `10` bis `15`: konkrete Detail-Views fuer Folgepipeline.

## Placeholder / Preview-only Stellen

- Stage `00 Command Center`: read-only command composer, kein Ausfuehren.
- Stage `01 Pipeline wählen`: preview-only, kein echter Pipeline-Wechsel.
- Stage `02 Mode & Style`: read-only Anzeige, keine echte Mode-/Style-Auswahl.
- Stage `05 Models`: fachlich nicht als eigenes Panel vorhanden.
- Stage `06 Mapping`: fachlich nicht als eigenes Panel vorhanden.
- Stage `08 Compiler Workspace`: nur als `Image Prompt Compiler` mit Artifact-Auswertung vorhanden.
- Spec `09 Active Workspace`: im Code kein eigener finaler Stage-Zustand, sondern Container/Router.

## Noetige Aenderungen bis Image/Keyframe Generation

Kleinster umsetzbarer Weg:

1. Stage `09` als Fokus behalten, weil bestehende Tests und Default-Auswahl darauf aufbauen.
2. In Stage `09` die aktuelle `CURRENT POSITION` und `PROMPTS / IMAGE JOBS` Struktur um eine kompakte Readiness-/Input-Zusammenfassung erweitern.
3. Diese Zusammenfassung soll Pipeline, Mode, Style/Hook-Hinweis, Prompt/Mapping-Quelle, Modell/Backend und Keyframe-Artefaktstatus zeigen.
4. Empty-State fuer fehlende Scenes/Jobs beibehalten und ggf. mit fehlenden Voraussetzungen anreichern.
5. Tests fuer Stage `09` nur um neue erwartete Texte ergaenzen.

## Dateien mit wahrscheinlichem Spaeter-Impact

- `/workspace/agent_core/creative_os/cockpit/panels/active_workspace_panel.py`
- `/workspace/agent_core/creative_os/cockpit/state_adapter.py`
- `/workspace/agent_core/creative_os/cockpit/stage_registry.py`
- `/workspace/agent_core/creative_os/cockpit/panels/pipeline_map_panel.py`
- `/workspace/tests/test_creative_os_cockpit.py`

## Empfehlung

Nicht mit einer Neuordnung von Stage IDs starten. Zuerst Stage `09` spec-konform schaerfen, weil dort der groesste Nutzwert fuer Image/Keyframe Generation liegt und die bestehende Architektur bereits passt.
