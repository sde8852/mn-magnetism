# src/mn_magnetism/features/composition.py

import numpy as np
from pymatgen.core import Structure


def get_composition_features(structure: Structure) -> dict:
    """Calculate composition-based descriptors."""

    comp = structure.composition
    elements = comp.element_composition

    n_atoms = len(structure)
    n_mn = sum(
        site.specie.symbol == "Mn"
        for site in structure
    )

    fractions = np.array([
        elements.get_atomic_fraction(el)
        for el in elements.elements
    ])

    return {
        "mn_fraction": n_mn / n_atoms,
        "n_elements": len(elements.elements),
        "mean_atomic_number": float(
            sum(
                el.Z * elements.get_atomic_fraction(el)
                for el in elements.elements
            )
        ),
        "composition_entropy": float(
            -np.sum(fractions * np.log(fractions))
        ),
    }
