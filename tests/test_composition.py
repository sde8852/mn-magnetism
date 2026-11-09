
import pytest
from pymatgen.core import Lattice, Structure

from mn_magnetism.features.composition import (
    get_composition_features,
)


def test_composition_features():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Mn", "Mn", "O"],
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        ],
    )

    features = get_composition_features(structure)

    assert features["mn_fraction"] == pytest.approx(2 / 3)
    assert features["n_elements"] == 2
    assert features["mean_atomic_number"] == pytest.approx(
        (2 * 25 + 8) / 3
    )
    assert features["composition_entropy"] > 0
