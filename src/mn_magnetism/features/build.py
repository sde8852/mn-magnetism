
# src/mn_magnetism/features/build.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from mn_magnetism.features.composition import (
    get_composition_features,
)
from mn_magnetism.features.structure import (
    get_structure_features,
)
from mn_magnetism.features.magnetic import (
    get_magnetic_features,
)


ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "data" / "interim" / "ground_states.json"
OUTPUT = ROOT / "data" / "processed" / "features.csv"
DIAGNOSTICS = (
    ROOT / "data" / "interim" / "magnetic_diagnostics.json"
)

# Metadata that must not become model input features
ID_COLUMNS = [
    "reduced_formula",
    "source_material_id",
    "data_id",
]

TARGET_COLUMN = "magnetic_order"


def extract_features(record: dict) -> tuple[dict, dict]:
    """
    Convert one ground-state record into:
      1. ML features + target
      2. Magnetic diagnostics
    """

    ordering = record["ordering"]

    structure = Structure.from_dict(ordering["structure"])

    # Independently calculated descriptor groups
    composition = get_composition_features(structure)
    geometry = get_structure_features(structure)
    magnetic = get_magnetic_features(structure)

    identifiers = {
        "reduced_formula": record["reduced_formula"],
        "source_material_id": record.get("source_material_id"),
        "data_id": record.get("data_id"),
    }

    # Model-ready information
    features = {
        **identifiers,
        **composition,
        **geometry,
        TARGET_COLUMN: magnetic[TARGET_COLUMN],
    }

    # Magnetic properties are diagnostics, not input features
    diagnostics = {
        **identifiers,
        "energy_per_atom": ordering.get("energy_per_atom"),
        **magnetic,
    }

    return features, diagnostics


def validate_features(df: pd.DataFrame) -> None:
    """Check the resulting ML feature table."""

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    required = set(ID_COLUMNS + [TARGET_COLUMN])

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["reduced_formula"].isna().any():
        raise ValueError("Missing reduced formulas.")

    if df["reduced_formula"].duplicated().any():
        raise ValueError("Duplicate reduced formulas detected.")

    valid_labels = {"FM", "AFM", "FiM", "NM"}

    if not df[TARGET_COLUMN].isin(valid_labels).all():
        raise ValueError("Invalid magnetic-ordering labels.")

    # Check physical features without treating missing values
    # as necessarily fatal.
    feature_cols = [
        c for c in df.columns
        if c not in ID_COLUMNS + [TARGET_COLUMN]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    values = df[feature_cols].to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ValueError("Infinite feature values detected.")


def build_features(
    input_path: Path = None,
    output_path: Path = None,
    diagnostics_path: Path = None,
) -> pd.DataFrame:
    """Build, validate, and save the ML feature dataset."""

    input_path = Path(input_path) if input_path is not None else INPUT
    output_path = Path(output_path) if output_path is not None else OUTPUT
    diagnostics_path = (
        Path(diagnostics_path)
        if diagnostics_path is not None
        else DIAGNOSTICS
    )

    # 1. Load selected ground states
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of ground states.")

    feature_rows = []
    diagnostic_rows = []

    # 2. Calculate descriptors
    for i, record in enumerate(records):
        try:
            features, diagnostics = extract_features(record)

        except Exception as exc:
            raise ValueError(
                f"Failed at record {i}: "
                f"{record.get('reduced_formula', 'unknown')}"
            ) from exc

        feature_rows.append(features)
        diagnostic_rows.append(diagnostics)

    # 3. Construct and validate DataFrame
    df = pd.DataFrame(feature_rows)

    validate_features(df)

    # 4. Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(output_path, index=False)

    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(
            diagnostic_rows,
            f,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )

    # 5. Summary
    print(f"\nProcessed {len(df)} materials.")
    print(f"Feature dataset: {output_path}")
    print(f"Diagnostics: {diagnostics_path}")

    print("\nMagnetic ordering distribution:")
    print(df[TARGET_COLUMN].value_counts())

    print("\nMissing feature values:")
    print(df.isna().sum()[lambda s: s > 0])

    return df


def _json_default(obj):
    """Serialize NumPy values and other common scalar types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    raise TypeError(f"Not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    features_df = build_features()
    print(features_df.head())
