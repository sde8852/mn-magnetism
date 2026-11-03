# src/mn_magnetism/features/composition.py

from pymatgen.core import Composition


def get_mn_fraction(formula: str) -> float:
    """Calculate the atomic fraction of Mn in a formula."""
    composition = Composition(formula)

    return composition.get_atomic_fraction("Mn")
