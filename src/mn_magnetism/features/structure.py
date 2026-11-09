
# src/mn_magnetism/features/structure.py

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


SHELL_TOL = 0.05  # Angstrom
INITIAL_RADIUS = 4.0
RADIUS_STEP = 2.0


def get_mn_indices(structure: Structure) -> list[int]:
    """Return indices of Mn sites."""
    return [
        i for i, site in enumerate(structure)
        if site.specie.symbol == "Mn"
    ]


def get_mn_neighbor_shells(
    structure: Structure,
    mn_indices: list[int],
    shell_tol: float = SHELL_TOL,
) -> dict:
    """Calculate periodic first- and second-shell Mn descriptors."""

    if not mn_indices:
        raise ValueError("Structure contains no Mn sites.")

    first_distances = []
    second_distances = []
    nn_distances = []
    coordination_numbers = []

    max_radius = max(
        20.0,
        1.5 * max(structure.lattice.abc),
    )

    for i in mn_indices:
        radius = INITIAL_RADIUS
        distances = []

        while radius <= max_radius:
            neighbors = structure.get_neighbors(
                structure[i], radius
            )

            distances = sorted(
                float(nn.nn_distance)
                for nn in neighbors
                if nn.specie.symbol == "Mn"
                and nn.nn_distance > 1e-8
            )

            # Need at least two distinct shells
            if distances:
                d_min = distances[0]
                has_second = any(
                    d > d_min + shell_tol
                    for d in distances
                )
                if has_second:
                    break

            radius += RADIUS_STEP

        if not distances:
            raise ValueError(
                f"No periodic Mn neighbors for site {i}"
            )

        d_min = distances[0]
        nn_distances.append(d_min)

        # First shell: all bonds near nearest distance
        first = [
            d for d in distances
            if abs(d - d_min) <= shell_tol
        ]

        first_distances.extend(first)
        coordination_numbers.append(len(first))

        # Second distinct shell
        second_candidates = [
            d for d in distances
            if d > d_min + shell_tol
        ]

        if second_candidates:
            d_second = min(second_candidates)

            second = [
                d for d in second_candidates
                if abs(d - d_second) <= shell_tol
            ]

            second_distances.extend(second)

    first_mean = float(np.mean(first_distances))

    second_mean = (
        float(np.mean(second_distances))
        if second_distances else np.nan
    )

    return {
        "mn_mn_min_dist": float(np.min(nn_distances)),
        "mn_mn_mean_nn_dist": float(np.mean(nn_distances)),
        "mn_mn_std_nn_dist": float(np.std(nn_distances)),
        "mn_first_shell_dist": first_mean,
        "mn_first_shell_cn": float(
            np.mean(coordination_numbers)
        ),
        "mn_second_shell_dist": second_mean,
        "mn_shell_ratio": (
            second_mean / first_mean
            if np.isfinite(second_mean) else np.nan
        ),
    }


def get_crystal_features(structure: Structure) -> dict:
    """Calculate lattice and symmetry descriptors."""

    lattice = structure.lattice
    a, b, c = lattice.abc

    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=0.1,
    )

    return {
        "volume_per_atom": float(
            structure.volume / len(structure)
        ),
        "density": float(structure.density),
        "lattice_anisotropy": float(
            max(a, b, c) / min(a, b, c)
        ),
        "space_group": int(
            analyzer.get_space_group_number()
        ),
        "crystal_system": analyzer.get_crystal_system(),
    }


def get_structure_features(structure: Structure) -> dict:
    """Combine crystallographic and Mn geometry features."""

    mn_indices = get_mn_indices(structure)

    return {
        **get_mn_neighbor_shells(structure, mn_indices),
        **get_crystal_features(structure),
    }
