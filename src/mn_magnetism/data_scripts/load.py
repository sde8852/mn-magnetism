# src/mn_magnetism/data_scripts/load.py

# Load raw data

import gzip
import json
import pandas as pd
from pathlib import Path

# -------------------------
# FILEPATHS
# -------------------------

# Find project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Path to raw data
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# -------------------------
# DATA LOADING
# -------------------------

def load_data(filename: str) -> pd.DataFrame:
    """Load a gzip-compressed JSON dataset."""

    mn_gz_file_path = RAW_DIR / filename

    if not mn_gz_file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {mn_gz_file_path}")
    
    with gzip.open(mn_gz_file_path, "rt") as fh:
        materials = json.load(fh)

    return pd.DataFrame(materials)