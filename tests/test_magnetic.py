
import numpy as np

from mn_magnetism.features.magnetic import (
    clean_magmoms,
    get_spin_pattern,
    classify_magnetic_ordering,
)


def test_clean_magmoms():
    moments = [2.5, -2.4, 0.02]

    result = clean_magmoms(moments)

    np.testing.assert_array_equal(
        result,
        [2.5, -2.4, 0.0],
    )


def test_spin_pattern():
    moments = [2.5, -2.4, 0.02]

    result = get_spin_pattern(moments)

    np.testing.assert_array_equal(
        result,
        [1, -1, 0],
    )


def test_magnetic_classification():
    assert classify_magnetic_ordering([0.01, 0.02]) == "NM"

    assert classify_magnetic_ordering([2.0, 2.1]) == "FM"

    assert classify_magnetic_ordering([2.0, -2.0]) == "AFM"

    assert classify_magnetic_ordering([2.0, -1.0]) == "FiM"
