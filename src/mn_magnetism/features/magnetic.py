
# src/mn_magnetism/features/magnetic.py

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.magnetism.analyzer import (
    CollinearMagneticStructureAnalyzer,
)


MAGMOM_TOL = 0.1  # Bohr magnetons
FIM_TOL = 0.1     # Bohr magnetons


def clean_magmoms(magmoms, tol=MAGMOM_TOL):
    """Set negligible collinear moments to zero."""

    m = np.asarray(magmoms, dtype=float).copy()

    if m.ndim != 1:
        raise ValueError(
            "Expected collinear scalar magnetic moments."
        )

    m[np.abs(m) < tol] = 0.0

    return m


def get_spin_pattern(magmoms, tol=MAGMOM_TOL):
    """Return -1, 0, +1 spin pattern."""

    return np.sign(
        clean_magmoms(magmoms, tol)
    ).astype(int)


def classify_magnetic_ordering(
    magmoms,
    magmom_tol=MAGMOM_TOL,
    fim_tol=FIM_TOL,
):
    """Classify collinear magnetic order from moments."""

    m = clean_magmoms(magmoms, magmom_tol)
    nonzero = m[m != 0]

    if len(nonzero) == 0:
        return "NM"

    has_pos = np.any(nonzero > 0)
    has_neg = np.any(nonzero < 0)

    if has_pos != has_neg:
        return "FM"

    positive = np.sum(nonzero[nonzero > 0])
    negative = np.sum(np.abs(nonzero[nonzero < 0]))

    imbalance = abs(positive - negative)

    if imbalance <= fim_tol:
        return "AFM"

    return "FiM"


def get_magnetic_features(structure: Structure) -> dict:
    """Extract magnetic labels and diagnostic descriptors."""

    if "magmom" not in structure.site_properties:
        raise ValueError("Structure has no magmom site property.")

    magmoms = np.asarray(
        structure.site_properties["magmom"],
        dtype=float,
    )

    mn_indices = [
        i for i, site in enumerate(structure)
        if site.specie.symbol == "Mn"
    ]

    mn_magmoms = magmoms[mn_indices]

    if not len(mn_magmoms):
        raise ValueError("Structure contains no Mn atoms.")

    abs_mn = np.abs(mn_magmoms)

    # Custom classifications
    final_ordering = classify_magnetic_ordering(magmoms)
    mn_ordering = classify_magnetic_ordering(mn_magmoms)

    # Optional pymatgen comparison
    try:
        analyzer = CollinearMagneticStructureAnalyzer(
            structure,
            threshold=MAGMOM_TOL,
        )

        pymatgen_ordering = analyzer.ordering.value
        n_unique_sites = (
            analyzer.number_of_unique_magnetic_sites()
        )

    except (ValueError, TypeError, AttributeError):
        pymatgen_ordering = None
        n_unique_sites = None

    return {
        "magnetic_order": final_ordering,
        "mn_magnetic_order": mn_ordering,
        "pymatgen_order": pymatgen_ordering,
        "n_unique_magnetic_sites": n_unique_sites,
        "total_magmom": float(np.sum(magmoms)),
        "mn_net_magmom": float(np.sum(mn_magmoms)),
        "mn_mean_abs_magmom": float(np.mean(abs_mn)),
        "mn_max_abs_magmom": float(np.max(abs_mn)),
        "mn_min_abs_magmom": float(np.min(abs_mn)),
        "mn_spin_pattern": get_spin_pattern(
            mn_magmoms
        ).tolist(),
    }
