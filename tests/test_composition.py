# tests/tests_composition.py

import pytest

from mn_magnetism.features.composition import get_mn_fraction


def test_mn_fraction():
    assert get_mn_fraction("MnO") == 0.5
    assert get_mn_fraction("Mn2O3") == 2 / 5
    assert get_mn_fraction("FeMnO3") == 1 / 5
    assert get_mn_fraction("Fe2MnO3") == 1 / 6
    assert get_mn_fraction("Fe2O3") == 0.0
