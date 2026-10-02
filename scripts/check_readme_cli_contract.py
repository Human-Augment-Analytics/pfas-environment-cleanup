#!/usr/bin/env python3
"""README-CLI contract check.

Extracts every --flag used in the README's dft_wrapper.py example block and
asserts each one is declared by scripts/dft_wrapper.py's argparse. Exits 1
with a precise message on any drift. Stdlib only.
"""

import re
import sys
from pathlib import Path

README = Path("README.md")
WRAPPER = Path("scripts/dft_wrapper.py")

readme_text = README.read_text(encoding="utf-8")
wrapper_src = WRAPPER.read_text(encoding="utf-8")

# 1. Locate the fenced block that invokes dft_wrapper.py
block = None
for m in re.finditer(r"```(?:\w+)?\n(.*?)```", readme_text, re.S):
    if "dft_wrapper.py" in m.group(1):
        block = m.group(1)
        break
if block is None:
    sys.exit("FAIL: no fenced block invoking dft_wrapper.py found in README")

# 2. Extract flags used in the README block (strip trailing backslashes/values)
readme_flags = sorted(set(re.findall(r"(--[a-z0-9-]+)", block)))

# 3. Extract flags declared by the wrapper's argparse
declared = sorted(set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', wrapper_src)))

# 4. Contract: every README flag must be declared
undeclared = [f for f in readme_flags if f not in declared]
if undeclared:
    sys.exit(f"FAIL: README uses flags dft_wrapper.py does not accept: {undeclared}")

# 5. Regression guard: four flags the wrapper never accepted must not reappear
bogus = ["--ecutwfc", "--ecutrho", "--input-dft", "--kpts"]
resurrected = [f for f in bogus if f in readme_flags]
if resurrected:
    sys.exit(f"FAIL: bogus flags re-introduced in README block: {resurrected}")

print(f"OK: {len(readme_flags)} README flags all declared by {WRAPPER}")
print("     declared set:", " ".join(declared))
