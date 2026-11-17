# Auto-generated from model.json - DO NOT EDIT
# Regenerate with: uv run model-types <path-to-model.json> -o src/model/types.py

from pathlib import Path
from typing import TypedDict, Literal

class InstanceInputs(TypedDict, total=False):
    # Chemical formula using case-sensitive element symbols, e.g. "Bi2Te3" or "Ag9 Tl1 Te5"
    composition: str

class DatasetInputs(TypedDict, total=False):
    # CSV file with a header row and one material per row: a 'composition' column of chemical formulas (e.g. Bi2Te3); any other columns are copied to the output unchanged
    file: tuple[Path, str]