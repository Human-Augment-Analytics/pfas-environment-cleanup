# Automated DFT adsorption pipeline

This repo drives Quantum ESPRESSo adsorption calculations on PACE-ICE for any
number of cases — from an entire CSV dataset down to a single adsorbent/PFAS
pair — with one command per campaign. The pipeline was validated live on
PACE-ICE with a 4-case pilot spanning different adsorbent clusters (all cases
completed, every E_ads recomputed exactly from its three energy legs).

Three layers, each usable on its own:

| Layer | Script | Use when |
|---|---|---|
| Campaign | `scripts/dft_seed.py` | you have a CSV of many adsorbents (the normal path) |
| Single case | `scripts/dft_single.py` | you want one adsorbent/PFAS pair |
| Power knobs | `scripts/dft_wrapper.py` | you need every individual flag |

## Prerequisites (once per machine)

1. SSH key auth to PACE-ICE working (`ssh <gatech-user>@login-ice.pace.gatech.edu`
   logs in without a password).
2. Repo deployed to the cluster (SLURM jobs run `scripts/run_dft_workflow.sh`
   from the cluster copy):

   ```
   bash scripts/deploy_to_cluster.sh --user <gatech-user>
   ```

   Default cluster root is `/storage/ice-shared/cs8903onl/mussmann-pfas`;
   pass `--cluster-root` if you use your own (e.g.
   `/storage/ice1/6/4/<gatech-user>/pfas`). The script refuses to clobber a
   cluster copy that has drifted from your checkout unless you pass `--force`.

## Run a whole dataset (campaign)

For any CSV with columns `ID,Name,SMILES,Category` (see
`scripts/seed_campaign_v1.csv` for the format) — one row per case, no row
limit:

```
python scripts/dft_seed.py submit  --user <gatech-user> --csv <seed.csv> [--cluster-root <root>]
python scripts/dft_seed.py status  --user <gatech-user> --csv <seed.csv> [--cluster-root <root>]
python scripts/dft_seed.py harvest --fetch --user <gatech-user> --csv <seed.csv> [--cluster-root <root>]
```

What the campaign driver handles for every case automatically:

- Production QE settings (TFA reference PFAS `FC(F)(F)C(=O)O`, molecule
  systems, `conv_thr=1e-07`, `forc_conv_thr=1e-04`, `ecutwfc=80 Ry`, BFGS
  relaxation), ice-cpu/coc/coc-ice scheduling, 18 h, 8 CPUs.
- Memory sized per case from SMILES heavy-atom counts (32/96/128 GB bands).
- OOM recovery per case: automatic resubmit with doubled memory (cap 256 GB,
  max 2 escalations). Finished legs are reused via the workflow's
  `compounds/` cache, so only the failed leg re-runs.
- PFAS reference-leg reuse across the whole campaign: the first completed
  case's TFA energy is read back from the cluster and passed to every later
  case (`pfas_energy_source=provided` in results), so only two legs run per
  case. `--no-pfas-reuse` disables this.
- Pacing and idempotency: default 10 new jobs per `submit` call, stops at 10
  of your jobs in flight, submits smallest systems first (`--order size`).
  Re-running any command never duplicates work — results exist → skip,
  job queued/running → monitor it, otherwise → submit. State lives in
  `<local-cache>/seed_state.json`; to resume after any interruption, rerun
  the same command.
- Range control for split datasets: `--only s0129,s0312` restricts to exact
  IDs; `--start-after s0312` skips everything up to an ID — so a team
  dividing a dataset just points each member at their slice.
- `harvest` writes `seed_campaign_results.csv` (one row per case: energies,
  verdict, provenance) plus a verdict rollup, and with `--fetch` downloads
  every completed case's evidence bundle into `./data/dft_cache/`.

## Single case

```
python scripts/dft_single.py run --user <gatech-user> --name "CTAB" --smiles "<SMILES>"
```

Same automation as above (settings, memory sizing, monitoring, OOM retry,
fetch, verdict) for one pair. Useful for ad-hoc checks outside a campaign.
Check or download an existing case:

```
python scripts/dft_single.py status --user <gatech-user> --case CTAB_TFA
python scripts/dft_single.py fetch --user <gatech-user> --case CTAB_TFA
```

`--no-wait` submits and prints the exact resume command.

## Reading the results

E_ads = E_complex − E_adsorbent − E_PFAS (negative = binding). Verdicts:

| E_ads (eV) | Verdict |
|---|---|
| ≥ 0 | NOT BINDING |
| −1..0 | PLAUSIBLE (weak) — expected for small neutrals without dispersion correction |
| −4..−1 | PLAUSIBLE (typical) |
| < −4 | SUSPECT — check ecutwfc/conv_thr/pseudopotentials before trusting |

Each fetched case dir holds `results.json` (all three leg energies + E_ads +
provenance), `meta.json`, the slurm log under `outputs/`, and the QE input and
output for the complex leg under `complex/` — everything needed to recompute
the numbers by hand.

Two things worth checking when a case finishes:

- BFGS step cap: if `complex/complex.out` ends with "The maximum number of
  steps has been reached", the geometry relaxed to the step limit rather than
  the force threshold. Final-step energy drift is usually a few meV (fine for
  screening); rerun with more steps if the case matters.
- OOM escalations are recorded in `meta.json` / the campaign state; a case
  that needed a memory bump is still valid — only the failed leg was redone.

Exit codes (both drivers): 0 finished · 1 completed but no results.json ·
2 failed / timeout / escalation exhausted · 3 attach target not found ·
130 interrupted (jobs keep running; rerun the same command to resume).

## Verification

```
python -m ruff format --check scripts/ tests/
python -m ruff check scripts/ tests/
python -m pytest tests/ -q -m "not slow"
```

All tests are offline (no SSH, no network).
