
# tests/test_build.py

import json

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from mn_magnetism.features.build import build_features


@pytest.fixture
def sample_ground_states(tmp_path):
    """Create a small synthetic ground-state dataset."""

    structure = Structure(
        Lattice.cubic(4.0),
        ["Mn", "O"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ],
        site_properties={
            "magmom": [2.0, 0.0],
        },
    )

    records = [
        {
            "reduced_formula": "MnO",
            "source_material_id": 0,
            "data_id": "test-001",
            "ordering": {
                "structure": structure.as_dict(),
                "energy_per_atom": -5.0,
            },
        }
    ]

    input_path = tmp_path / "ground_states.json"

    with input_path.open("w") as f:
        json.dump(records, f)

    return input_path


def test_build_features(sample_ground_states, tmp_path):
    """Verify the complete feature-building pipeline."""

    output = tmp_path / "features.csv"
    diagnostics = tmp_path / "magnetic_diagnostics.json"

    df = build_features(
        input_path=sample_ground_states,
        output_path=output,
        diagnostics_path=diagnostics,
    )

    # Check output files
    assert output.exists()
    assert diagnostics.exists()

    # Check number of materials
    assert len(df) == 1

    # Check identifiers
    assert df.iloc[0]["reduced_formula"] == "MnO"
    assert df.iloc[0]["data_id"] == "test-001"

    # Check known descriptors
    assert df.iloc[0]["mn_fraction"] == pytest.approx(0.5)
    assert df.iloc[0]["mn_mn_min_dist"] == pytest.approx(4.0)

    # Check magnetic target
    assert df.iloc[0]["magnetic_order"] == "FM"

    # Prevent target leakage
    assert "total_magmom" not in df.columns
    assert "energy_per_atom" not in df.columns

    # Verify CSV can be reloaded
    saved = pd.read_csv(output)

    assert len(saved) == 1
    assert saved.iloc[0]["magnetic_order"] == "FM"

    # Verify magnetic diagnostics
    with diagnostics.open("r") as f:
        diagnostic_records = json.load(f)

    assert len(diagnostic_records) == 1

    assert diagnostic_records[0]["total_magmom"] == pytest.approx(
        2.0
    )
