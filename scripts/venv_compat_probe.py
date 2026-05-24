#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INTERPRETERS = {
    "opt_venv": Path("/opt/venv/bin/python"),
    "qwen3_tts": Path("/workspace/venvs/qwen3-tts/bin/python"),
}

PACKAGES = [
    "torch",
    "transformers",
    "diffusers",
    "accelerate",
    "tokenizers",
    "safetensors",
    "numpy",
    "scipy",
    "qwen",
    "qwen_tts",
    "librosa",
    "soundfile",
    "torchaudio",
    "torchvision",
    "einops",
    "peft",
    "sentencepiece",
    "modelscope",
    "funasr",
    "huggingface_hub",
    "ace_step",
    "acestep",
    "acestep.handler",
    "acestep.inference",
    "acestep.llm_inference",
    "torchao",
    "xformers",
    "loguru",
    "toml",
    "vector_quantize_pytorch",
    "matplotlib",
    "fastapi",
    "uvicorn",
    "xxhash",
]

INTERNAL_MODULES = [
    "app.qwen_tts",
    "app.ace_step_1_5",
]

PROBE_CODE = r"""
import importlib
import json
import os
import sys

packages = __PACKAGES__
internal_modules = __INTERNAL_MODULES__

result = {
    "python": sys.version,
    "prefix": sys.prefix,
    "executable": sys.executable,
    "packages": {},
    "internal_modules": {},
}

for name in packages:
    item = {"ok": False, "version": None, "error": None}
    try:
        mod = importlib.import_module(name)
        item["ok"] = True
        item["version"] = getattr(mod, "__version__", "unknown")
        if name == "torch":
            item["cuda"] = getattr(mod.version, "cuda", None)
            item["cuda_available"] = bool(mod.cuda.is_available())
        if name == "transformers":
            try:
                import transformers.configuration_utils as cu
                item["has_layer_type_validation"] = hasattr(cu, "layer_type_validation")
            except Exception as exc:
                item["configuration_utils_error"] = repr(exc)
    except Exception as exc:
        item["error"] = repr(exc)
    result["packages"][name] = item

for name in internal_modules:
    item = {"ok": False, "version": None, "error": None}
    try:
        mod = importlib.import_module(name)
        item["ok"] = True
        item["version"] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        item["error"] = repr(exc)
    result["internal_modules"][name] = item

print(json.dumps(result, indent=2, sort_keys=True))
"""


def _run_interpreter(name: str, python_path: Path) -> dict[str, Any]:
    if not python_path.exists():
        return {"ok": False, "error": f"missing interpreter: {python_path}"}
    code = (
        PROBE_CODE.replace("__PACKAGES__", repr(PACKAGES))
        .replace("__INTERNAL_MODULES__", repr(INTERNAL_MODULES))
    )
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = "/workspace:/workspace/ACE-Step-1.5"
    proc = subprocess.run(
        [str(python_path), "-c", code],
        cwd="/workspace",
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    payload: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
    }
    try:
        payload.update(json.loads(proc.stdout))
    except json.JSONDecodeError:
        payload["stdout"] = proc.stdout
        payload["error"] = "invalid_json_from_probe"
    return payload


def _du(path: Path) -> str:
    proc = subprocess.run(["du", "-sh", str(path)], text=True, capture_output=True)
    if proc.returncode != 0:
        return "missing"
    return proc.stdout.split()[0]


def _status(item: dict[str, Any] | None) -> str:
    if not item:
        return "missing"
    if item.get("ok"):
        version = item.get("version")
        if version and version != "unknown":
            return f"ok {version}"
        return "ok"
    return f"missing/error: {item.get('error')}"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    opt = report["interpreters"].get("opt_venv", {})
    qwen = report["interpreters"].get("qwen3_tts", {})
    lines: list[str] = []
    lines.append("# Venv Compatibility Probe: /opt/venv vs qwen3-tts")
    lines.append("")
    lines.append("## 1. Ziel")
    lines.append("- Read-only pruefen, ob Qwen TTS und ACE-Step perspektivisch auf `/opt/venv` laufen koennten.")
    lines.append("- Keine venvs, Requirements, Dockerfile oder App-Code aendern.")
    lines.append("")
    lines.append("## 2. Interpreter")
    lines.append("")
    lines.append("| Interpreter | Größe | Python | Zweck |")
    lines.append("|---|---:|---|---|")
    lines.append(
        f"| `/opt/venv/bin/python` | {report['sizes'].get('opt_venv', 'unknown')} | {opt.get('python', 'unknown').splitlines()[0] if opt.get('python') else 'unknown'} | Service/API/Jupyter Default |"
    )
    lines.append(
        f"| `/workspace/venvs/qwen3-tts/bin/python` | {report['sizes'].get('qwen3_tts', 'unknown')} | {qwen.get('python', 'unknown').splitlines()[0] if qwen.get('python') else 'unknown'} | Qwen TTS + ACE Runtime |"
    )
    lines.append("")
    lines.append("## 3. Import Matrix")
    lines.append("")
    lines.append("| Paket | /opt/venv | qwen3-tts venv | Bewertung |")
    lines.append("|---|---|---|---|")
    for pkg in PACKAGES:
        opt_item = (opt.get("packages") or {}).get(pkg)
        qwen_item = (qwen.get("packages") or {}).get(pkg)
        opt_ok = bool(opt_item and opt_item.get("ok"))
        qwen_ok = bool(qwen_item and qwen_item.get("ok"))
        if opt_ok and qwen_ok:
            rating = "beide ok"
        elif (not opt_ok) and qwen_ok:
            rating = "Luecke in /opt/venv"
        elif opt_ok and not qwen_ok:
            rating = "nur /opt/venv"
        else:
            rating = "in beiden nicht verfuegbar"
        lines.append(f"| `{pkg}` | {_status(opt_item)} | {_status(qwen_item)} | {rating} |")
    lines.append("")
    lines.append("## 4. Kritische Lücken in /opt/venv")
    opt_gaps = []
    for pkg in PACKAGES:
        opt_item = (opt.get("packages") or {}).get(pkg)
        qwen_item = (qwen.get("packages") or {}).get(pkg)
        if qwen_item and qwen_item.get("ok") and not (opt_item and opt_item.get("ok")):
            opt_gaps.append(pkg)
    if opt_gaps:
        for pkg in opt_gaps:
            lines.append(f"- `{pkg}` fehlt oder importiert nicht in `/opt/venv`.")
    else:
        lines.append("- Keine Paketluecke gefunden, die in qwen3-tts vorhanden und in `/opt/venv` fehlend ist.")
    lines.append("")
    lines.append("## 5. Versionsabweichungen")
    diffs = []
    for pkg in PACKAGES:
        opt_item = (opt.get("packages") or {}).get(pkg) or {}
        qwen_item = (qwen.get("packages") or {}).get(pkg) or {}
        if opt_item.get("ok") and qwen_item.get("ok") and opt_item.get("version") != qwen_item.get("version"):
            diffs.append((pkg, opt_item.get("version"), qwen_item.get("version")))
    if diffs:
        for pkg, opt_v, qwen_v in diffs:
            lines.append(f"- `{pkg}`: `/opt/venv`={opt_v}, qwen3-tts={qwen_v}")
    else:
        lines.append("- Keine Versionsabweichung bei beidseitig importierbaren Probe-Paketen.")
    lines.append("")
    lines.append("## 6. Qwen TTS Kompatibilität")
    qwen_required = ["qwen_tts", "torch", "transformers", "torchaudio", "librosa", "soundfile", "einops", "sentencepiece", "huggingface_hub"]
    missing_qwen = [p for p in qwen_required if not ((opt.get("packages") or {}).get(p) or {}).get("ok")]
    app_qwen = (opt.get("internal_modules") or {}).get("app.qwen_tts", {})
    lines.append(f"- Import moeglich: {'ja' if not missing_qwen and app_qwen.get('ok') else 'vielleicht' if not missing_qwen else 'nein'}")
    lines.append(f"- fehlende Pakete: {', '.join(missing_qwen) if missing_qwen else 'keine in der Probe'}")
    lines.append(f"- App-Modul `app.qwen_tts` mit `/opt/venv`: {_status(app_qwen)}")
    lines.append("- Risiko: niedrig bis mittel; Qwen-Kernpakete sind in `/opt/venv` vorhanden, aber App-Code referenziert aktuell hart die Workspace-venv fuer Worker-Aufrufe.")
    lines.append("- Urteil: technisch wahrscheinlich konsolidierbar, aber nicht ohne gezielte Worker-Smoke-Tests und Code-/init.sh-Anpassung.")
    lines.append("")
    lines.append("## 7. ACE-Step Kompatibilität")
    ace_required = ["torch", "transformers", "diffusers", "loguru", "toml", "modelscope", "torchvision", "torchao", "matplotlib", "scipy", "soundfile", "einops", "fastapi", "uvicorn", "vector_quantize_pytorch", "xxhash"]
    missing_ace = [p for p in ace_required if not ((opt.get("packages") or {}).get(p) or {}).get("ok")]
    app_ace = (opt.get("internal_modules") or {}).get("app.ace_step_1_5", {})
    lines.append(f"- Import moeglich: {'ja' if not missing_ace and app_ace.get('ok') else 'vielleicht' if len(missing_ace) <= 3 else 'nein'}")
    lines.append(f"- fehlende Pakete: {', '.join(missing_ace) if missing_ace else 'keine in der Probe'}")
    lines.append(f"- App-Modul `app.ace_step_1_5` mit `/opt/venv`: {_status(app_ace)}")
    lines.append("- Risiko: mittel; ACE hat Zusatzpakete in qwen3-tts, und `torchao` meldet mit Torch 2.7 einen C++ Extension Skip.")
    lines.append("- Urteil: Konsolidierung ist nur mit vorherigem gezieltem Ergänzen/Validieren der fehlenden ACE-Pakete sicher.")
    lines.append("")
    lines.append("## 8. Kann qwen3-tts venv ersetzt werden?")
    if missing_ace:
        answer = "vielleicht"
        reason = "Qwen sieht weitgehend bereit aus, ACE hat aber Luecken in `/opt/venv` oder braucht weitere Smoke-Tests."
    else:
        answer = "vielleicht"
        reason = "Import-Matrix ist gut, aber Worker-Pfade sind hart auf die Workspace-venv verdrahtet."
    lines.append("Antwort:")
    lines.append(f"- {answer}")
    lines.append(f"- Begruendung: {reason}")
    lines.append("")
    lines.append("## 9. Sicherster nächster Build")
    lines.append("- Fehlende reine Python-Pakete fuer ACE in `/opt/venv` gezielt im Image ergaenzen und danach Qwen/ACE Worker-Smoke-Tests gegen `/opt/venv` ausfuehren. Erst danach `QWEN_VENV`/`ACE_STEP_PYTHON` optional auf `/opt/venv` konsolidieren.")
    lines.append("")
    lines.append("## 10. Nicht getan")
    lines.append("- keine venv geloescht")
    lines.append("- keine requirements geaendert")
    lines.append("- kein pip install")
    lines.append("- kein pip uninstall")
    lines.append("- kein init.sh geaendert")
    lines.append("")
    lines.append("## 11. Probe-Artefakte")
    lines.append("- JSON: `/workspace/status/venv_compat_probe.json`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    status_dir = Path("/workspace/status")
    status_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interpreters": {},
        "sizes": {
            "opt_venv": _du(Path("/opt/venv")),
            "qwen3_tts": _du(Path("/workspace/venvs/qwen3-tts")),
        },
        "packages_tested": PACKAGES,
        "internal_modules_tested": INTERNAL_MODULES,
    }
    for name, python_path in INTERPRETERS.items():
        report["interpreters"][name] = _run_interpreter(name, python_path)

    json_path = status_dir / "venv_compat_probe.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(report, Path("/workspace/VENV_OPT_COMPAT_PROBE.md"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
