#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
from ase import Atom
from ase.build import graphene
from ase.io import write
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def build_graphene_sheet(nx: int, ny: int, vacuum: float):
    atoms = graphene(formula="C2", a=2.46, size=(nx, ny, 1), vacuum=vacuum)
    atoms.center(vacuum=vacuum / 2.0, axis=2)
    return atoms


def carbon_indices(atoms):
    return [i for i, atom in enumerate(atoms) if atom.symbol == "C"]


def add_hydroxyl_group(atoms, carbon_index: int, above: bool = True):
    c_pos = atoms[carbon_index].position.copy()
    direction = 1.0 if above else -1.0

    o_pos = c_pos + np.array([0.0, 0.0, 1.43 * direction])
    h_pos = o_pos + np.array([0.0, 0.0, 0.97 * direction])

    atoms.append(Atom("O", o_pos))
    atoms.append(Atom("H", h_pos))


def find_cc_pairs(atoms, max_dist: float = 1.7):
    c_inds = carbon_indices(atoms)
    pairs = []

    for i, a in enumerate(c_inds):
        for b in c_inds[i + 1:]:
            d = atoms.get_distance(a, b, mic=True)
            if d < max_dist:
                pairs.append((a, b))
    return pairs


def add_epoxide_group(atoms, c1: int, c2: int, above: bool = True):
    p1 = atoms[c1].position
    p2 = atoms[c2].position
    midpoint = 0.5 * (p1 + p2)

    direction = 1.0 if above else -1.0
    o_pos = midpoint + np.array([0.0, 0.0, 1.20 * direction])

    atoms.append(Atom("O", o_pos))


def pick_nonoverlapping_carbons(atoms, count: int, seed: int):
    rng = random.Random(seed)
    candidates = carbon_indices(atoms)
    rng.shuffle(candidates)

    chosen = []
    for idx in candidates:
        too_close = False
        for j in chosen:
            if atoms.get_distance(idx, j, mic=True) < 2.5:
                too_close = True
                break
        if not too_close:
            chosen.append(idx)
        if len(chosen) == count:
            break

    if len(chosen) < count:
        raise RuntimeError(f"Could only place {len(chosen)} hydroxyl sites, needed {count}")

    return chosen


def pick_nonoverlapping_pairs(atoms, count: int, seed: int):
    rng = random.Random(seed)
    pairs = find_cc_pairs(atoms)
    rng.shuffle(pairs)

    chosen = []
    used = set()

    for a, b in pairs:
        if a in used or b in used:
            continue
        chosen.append((a, b))
        used.add(a)
        used.add(b)
        if len(chosen) == count:
            break

    if len(chosen) < count:
        raise RuntimeError(f"Could only place {len(chosen)} epoxide sites, needed {count}")

    return chosen


def build_go(nx: int, ny: int, vacuum: float, epoxides: int, hydroxyls: int, seed: int):
    atoms = build_graphene_sheet(nx, ny, vacuum)

    epoxide_pairs = pick_nonoverlapping_pairs(atoms, epoxides, seed)
    hydroxyl_sites = pick_nonoverlapping_carbons(atoms, hydroxyls, seed + 1)

    rng = random.Random(seed)

    for a, b in epoxide_pairs:
        add_epoxide_group(atoms, a, b, above=rng.choice([True, False]))

    for c in hydroxyl_sites:
        add_hydroxyl_group(atoms, c, above=rng.choice([True, False]))

    return atoms


def build_sic4h_si_terminated(
    nx: int = 6,
    ny: int = 6,
    vacuum: float = 15.0,
    a: float = 3.08,
    layer_spacing: float = 1.9,
    h_bond: float = 1.50,
):
    """
    Approximate Si-terminated 4H-SiC slab.

    For nx=6, ny=6:
    - 72 Si
    - 72 C
    - 72 H passivation atoms
    - 216 atoms total

    Top surface is Si-terminated.
    Bottom surface is C-terminated and H-passivated.
    """

    lattice = Lattice([
        [nx * a, 0.0, 0.0],
        [-0.5 * ny * a, np.sqrt(3) * 0.5 * ny * a, 0.0],
        [0.0, 0.0, 4 * layer_spacing + vacuum],
    ])

    z_offset = vacuum / 2.0

    layers = [
        ("C", 0.0, (0.0, 0.0)),
        ("Si", layer_spacing, (1 / 3, 2 / 3)),
        ("C", 2 * layer_spacing, (2 / 3, 1 / 3)),
        ("Si", 3 * layer_spacing, (0.0, 0.0)),
    ]

    species = []
    coords = []

    top_si_positions = []
    bottom_c_positions = []

    for symbol, z, shift in layers:
        sx, sy = shift

        for i in range(nx):
            for j in range(ny):
                frac_x = (i + sx) / nx
                frac_y = (j + sy) / ny

                cart = lattice.get_cartesian_coords([frac_x, frac_y, 0.0])
                cart[2] = z + z_offset

                species.append(symbol)
                coords.append(cart)

                if symbol == "Si" and z == 3 * layer_spacing:
                    top_si_positions.append(cart.copy())

                if symbol == "C" and z == 0.0:
                    bottom_c_positions.append(cart.copy())

    # H-passivate exposed top Si atoms
    for pos in top_si_positions:
        h_pos = pos.copy()
        h_pos[2] += h_bond
        species.append("H")
        coords.append(h_pos)

    # H-passivate exposed bottom C atoms
    for pos in bottom_c_positions:
        h_pos = pos.copy()
        h_pos[2] -= h_bond
        species.append("H")
        coords.append(h_pos)

    return Structure(
        lattice,
        species,
        coords,
        coords_are_cartesian=True,
    )


def write_output(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, Structure):
        CifWriter(obj, symprec=None).write_file(str(path))
        print(f"[info] Formula: {obj.composition}")
        print(f"[info] Natoms: {len(obj)}")
        print(f"[info] Lattice:\n{obj.lattice}")
    else:
        write(str(path), obj)
        print(f"[info] Formula: {obj.get_chemical_formula()}")
        print(f"[info] Natoms: {len(obj)}")
        print(f"[info] Cell:\n{obj.cell}")


def main():
    parser = argparse.ArgumentParser(description="Build a periodic adsorbent slab and export CIF")

    parser.add_argument(
        "--template",
        choices=["graphene", "go", "sic4h_si"],
        required=True,
    )
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--vacuum", type=float, default=15.0)

    # GO options
    parser.add_argument("--epoxides", type=int, default=0)
    parser.add_argument("--hydroxyls", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # SiC options
    parser.add_argument("--sic-a", type=float, default=3.08)
    parser.add_argument("--sic-layer-spacing", type=float, default=1.9)
    parser.add_argument("--sic-h-bond", type=float, default=1.50)

    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.template == "graphene":
        structure = build_graphene_sheet(args.nx, args.ny, args.vacuum)

    elif args.template == "go":
        structure = build_go(
            nx=args.nx,
            ny=args.ny,
            vacuum=args.vacuum,
            epoxides=args.epoxides,
            hydroxyls=args.hydroxyls,
            seed=args.seed,
        )

    elif args.template == "sic4h_si":
        structure = build_sic4h_si_terminated(
            nx=args.nx,
            ny=args.ny,
            vacuum=args.vacuum,
            a=args.sic_a,
            layer_spacing=args.sic_layer_spacing,
            h_bond=args.sic_h_bond,
        )

    else:
        raise ValueError(f"Unknown template: {args.template}")

    out_path = Path(args.out)
    write_output(out_path, structure)

    print(f"[done] Wrote periodic adsorbent to {out_path}")


if __name__ == "__main__":
    main()