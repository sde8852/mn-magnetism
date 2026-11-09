
import pytest
from pymatgen.core import Lattice, Structure

from mn_magnetism.features.structure import (
    get_mn_indices,
    get_structure_features,
)


def test_mn_indices():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Mn", "O", "Mn"],
        [
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
        ],
    )

    assert get_mn_indices(structure) == [0, 2]


def test_periodic_mn_distances():
    # One Mn per cubic cell.
    # The nearest Mn atoms are periodic images 4 Å away.
    structure = Structure(
        Lattice.cubic(4.0),
        ["Mn"],
        [[0, 0, 0]],
    )

    features = get_structure_features(structure)

    assert features["mn_mn_min_dist"] == pytest.approx(4.0)
    assert features["mn_mn_mean_nn_dist"] == pytest.approx(4.0)
    assert features["mn_first_shell_cn"] == pytest.approx(6.0)
    assert features["mn_second_shell_dist"] == pytest.approx(
        4.0 * 2**0.5
    )

    assert features["volume_per_atom"] == pytest.approx(64.0)
    assert features["space_group"] == 221
