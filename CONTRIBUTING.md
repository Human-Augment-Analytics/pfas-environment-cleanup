# Contributing

## Setting up

The repository ships three conda environments. Create only the one you need
for the part you are working on:

| Environment | File | Use |
|---|---|---|
| `pfas` | `environment.yaml` | Data pipeline (`scripts/fetch_data.py`, tabular ML preprocessing) |
| `qe` | `qe_environment.yaml` | DFT pipeline python dependencies (`qespresso_pipeline/`) |
| `pfas_gnn_env` | `basic_molecule_gnn/environment.yml` | Molecule GNN demo (`basic_molecule_gnn/`) |

```
conda env create -f environment.yaml        # or qe_environment.yaml, or the GNN file
conda activate pfas                          # or qe / pfas_gnn_env
```

## Running DFT

Note: when running DFT, be sure to save the results from all runs,
including the input config files. Every run — including failed and
intermediate ones — should be preserved somewhere durable (your cluster
storage or an archive you control), so the exact inputs that produced a
result can always be recovered. Bulky run outputs still do not belong in
git (see below); save them, don't commit them.

## Before you open a pull request

Two GitHub Actions workflows run on every pull request:

- **CI** — shellcheck on `scripts/*.sh`, a ruff/compileall syntax pass over
  the python files, the README CLI contract check (every flag shown in the
  README's `dft_wrapper.py` example must exist in the script), a lint of the
  conda environment files, and a link check of the README.
- **Smoke** — rebuilds quantum-espresso inputs from SMILES strings through
  the MOL → CIF → input-file chain, including an ionic row, to catch
  breakage in the preparation chain.

You can run the python-side checks locally before pushing:

```
python scripts/check_readme_cli_contract.py
ruff check --select E9,F63,F7,F82 .
python -m compileall -q .
```

## Pull request checklist

- CI and Smoke workflows are green on the PR.
- If your change alters a script's flags or behavior, update the README to
  match (the contract check will fail otherwise for `dft_wrapper.py`).
- If the PR involved DFT runs, the results and input configs from all runs
  are saved, and referenced or linked in the PR description.
- Do not commit generated run outputs: `dft_cases/`, `compounds/`,
  `dft_runs/`, and `logs/` are gitignored — keep them that way. Metrics
  JSONs and captured console logs under `models/` are the exception and are
  committed deliberately.
- Keep commit messages in plain language describing what changed and why.

## Reporting problems

If a DFT or pipeline run fails, or you cannot get one of the environments
installed, open an issue — the repository provides two issue templates
("DFT run help" and "Environment setup") that walk you through the
information we need.
