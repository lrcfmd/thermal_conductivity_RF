# Auto-generated from model.json - DO NOT EDIT
# Regenerate with: uv run model-types <path-to-model.json> -o src/model/types.py

from pathlib import Path
from typing import TypedDict, Literal

class InstanceInputs(TypedDict, total=False):
    # Chemical formula, e.g. "Bi2Te3" or "Ag9 Tl1 Te5"
    composition: str

class DatasetInputs(TypedDict, total=False):
    # CSV file with a 'composition' column of chemical formulas; other columns are passed through
    file: tuple[Path, str]