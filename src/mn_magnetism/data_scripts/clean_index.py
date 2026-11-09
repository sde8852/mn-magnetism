# src/mn_magnetism/data_scripts/clean_index.py

import pandas as pd
from pymatgen.core import Structure

# ===================================================================

def extract_row(raw_df_row: pd.Series) -> pd.DataFrame:
    """Extract all magnetic configurations for one material, which is one row of raw_df.
    raw_df_row is a row from the raw dataframe, which contains a list of magnetic configurations.
    Each configuration is a dictionary with a 'structure' key, which contains the structure data.

    Outputs a dictionary with the reduced formula and the ordering for each configuration.
    """

    extracted = []

    for ordering in raw_df_row:

        # Skip missing configurations
        if not isinstance(ordering, dict):
            continue

        # Reconstruct pymatgen Structure
        structure = Structure.from_dict(ordering["structure"])

        # Extract reduced formula
        reduced_formula = structure.composition.reduced_formula

        extracted.append({
            "reduced_formula": reduced_formula,
            "ordering": ordering
        })

    return extracted

# ===================================================================

def extract_all_rows(raw_df):
    records = []

    for idx, row in raw_df.iterrows():
        for record in extract_row(row):
            record["source_material_id"] = idx
            records.append(record)

    return pd.DataFrame(records)

# ===================================================================

def match_data_ids(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Match reduced formulas to their database IDs."""

    reference = reference_df.rename(
        columns={
            "path": "database_id",
            "formula": "reduced_formula"
        }
    ).copy()

    # Extract ID after the first 10 characters
    reference["data_id"] = reference["database_id"].str[10:]

    # Combine multiple IDs for the same reduced formula
    grouped = (
        reference.groupby("reduced_formula")["data_id"]
        .agg(lambda ids: ", ".join(ids.dropna().unique()))
        .reset_index()
    )

    # Match IDs to each magnetic configuration
    return df.merge(
        grouped,
        on="reduced_formula",
        how="left",
        validate="many_to_one"
    )

# ===================================================================

