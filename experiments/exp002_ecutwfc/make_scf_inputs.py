"""Exp 002 step 2: build fixed-geometry SCF inputs from the relaxed TFA run.

Takes the relax input (pfas.in) and the final coordinates from its output
(pfas.out), switches calculation to 'scf' and writes one input per cutoff
with ecutrho = 10 x ecutwfc, matching the pipeline presets.
"""

import re
import sys
from pathlib import Path

CUTOFFS = (40, 60, 80)


def final_positions(out_text: str) -> str:
    m = re.search(
        r"Begin final coordinates.*?(ATOMIC_POSITIONS.*?)\nEnd final coordinates",
        out_text,
        re.S,
    )
    if not m:
        sys.exit("No 'Begin final coordinates' block found; relax not converged?")
    return m.group(1).rstrip() + "\n"


def replace_positions(in_text: str, positions: str) -> str:
    # ATOMIC_POSITIONS runs until the next card (K_POINTS) or end of file.
    return re.sub(
        r"ATOMIC_POSITIONS.*?(?=^\s*K_POINTS|\Z)",
        lambda _: positions,
        in_text,
        flags=re.S | re.M,
    )


def set_param(text: str, key: str, value: str) -> str:
    new, n = re.subn(
        rf"^(\s*){key}\s*=\s*[^,\n]*,?", rf"\g<1>{key}={value},", text, flags=re.M
    )
    if n != 1:
        sys.exit(f"Expected exactly one '{key}' in input, found {n}")
    return new


def main() -> None:
    relax_dir = Path(sys.argv[1])
    scf_root = Path(sys.argv[2])
    in_text = (relax_dir / "pfas.in").read_text()
    positions = final_positions((relax_dir / "pfas.out").read_text())
    base = replace_positions(in_text, positions)
    base = set_param(base, "calculation", "'scf'")

    for ecut in CUTOFFS:
        d = scf_root / f"ecut{ecut}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "Outputs").mkdir(exist_ok=True)
        pseudo = d / "Pseudopotentials"
        if not pseudo.exists():
            pseudo.symlink_to((relax_dir / "Pseudopotentials").resolve())
        text = set_param(base, "ecutwfc", str(ecut))
        text = set_param(text, "ecutrho", str(ecut * 10))
        text = set_param(text, "prefix", f"'tfa_ecut{ecut}'")
        (d / "tfa_scf.in").write_text(text)
        print(f"[exp02] wrote {d / 'tfa_scf.in'}")


if __name__ == "__main__":
    main()
