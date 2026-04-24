#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
from ase import Atom
from ase.build import graphene
from ase.io import write


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


def main():
    parser = argparse.ArgumentParser(description="Build a periodic adsorbent slab and export CIF")
    parser.add_argument("--template", choices=["graphene", "go"], required=True)
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--vacuum", type=float, default=15.0)
    parser.add_argument("--epoxides", type=int, default=0)
    parser.add_argument("--hydroxyls", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.template == "graphene":
        atoms = build_graphene_sheet(args.nx, args.ny, args.vacuum)
    elif args.template == "go":
        atoms = build_go(
            nx=args.nx,
            ny=args.ny,
            vacuum=args.vacuum,
            epoxides=args.epoxides,
            hydroxyls=args.hydroxyls,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown template: {args.template}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms)

    print(f"[done] Wrote periodic adsorbent to {out_path}")
    print(f"[info] Formula: {atoms.get_chemical_formula()}")
    print(f"[info] Cell:\n{atoms.cell}")


if __name__ == "__main__":
    main()