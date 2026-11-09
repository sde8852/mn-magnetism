# src/mn_magnetism/features/ground_state.py


import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def ground_state_extraction(
    input_file="all_orderings.json",
    output_file="ground_states.json",
) -> pd.DataFrame:
    """Extract the lowest-energy configuration per reduced formula."""

    # 1. Load interim dataset
    input_path = INTERIM_DIR / input_file

    with open(input_path, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)

    # 2. Extract energy per atom directly
    df["energy_per_atom"] = pd.to_numeric(
        df["ordering"].apply(
            lambda ordering: ordering.get("energy_per_atom")
            if isinstance(ordering, dict) else None
        ),
        errors="coerce",
    )

    # Remove missing or invalid energies
    df["energy_per_atom"] = df["energy_per_atom"].replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=["energy_per_atom"])

    # 3. Select lowest-energy ordering per reduced formula
    ground_idx = (
        df.groupby("reduced_formula")["energy_per_atom"]
        .idxmin()
    )

    ground_df = (
        df.loc[ground_idx]
        .sort_values("reduced_formula")
        .reset_index(drop=True)
    )

    # 4. Save ground states
    output_path = INTERIM_DIR / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ground_df.to_json(
        output_path,
        orient="records",
        indent=2,
    )

    print(f"Extracted {len(ground_df)} ground states.")
    print(f"Saved to {output_path}")

    return ground_df

