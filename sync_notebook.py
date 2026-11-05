"""Regenerate the project notebook from the executable percent-format script."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
script = (ROOT / "pairs_pipeline.py").read_text(encoding="utf-8").splitlines(keepends=True)
path = ROOT / "pairs_trading_final_model.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))

cells = [notebook["cells"][0]]  # Kaggle dependency-install cell
kind = None
lines = []


def append_cell():
    if not lines:
        return
    if kind == "markdown":
        source = [line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                  for line in lines]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
    else:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": lines.copy()})


for line in script:
    if line.startswith("# %%"):
        append_cell()
        lines = []
        kind = "markdown" if "[markdown]" in line else "code"
    else:
        lines.append(line)
append_cell()
notebook["cells"] = cells
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Updated {path.name}: {len(cells)} cells")
