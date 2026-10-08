"""Fail if any deliverable prose has an em dash or a spaced en dash (style rule)."""
import _bootstrap  # noqa: F401
import json
import sys

from src.config import ROOT

BAD = {"—": "em dash", " – ": "spaced en dash"}
files = [ROOT / f for f in ["REPORT.md", "README.md", "AI_USE.md"]]
nb = ROOT / "analysis.ipynb"
problems = []
for f in files:
    if f.exists():
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for ch, name in BAD.items():
                if ch in line:
                    problems.append(f"{f.name}:{i}: {name}: {line.strip()[:80]}")
if nb.exists():
    for c in json.loads(nb.read_text())["cells"]:
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            for ch, name in BAD.items():
                if ch in src:
                    problems.append(f"analysis.ipynb markdown: {name}: {src[:60]}")
print("\n".join(problems) if problems else "prose lint OK: no em dashes")
sys.exit(1 if problems else 0)
