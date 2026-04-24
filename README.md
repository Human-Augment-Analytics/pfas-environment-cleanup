# PFAS Adsorbent Search

## Setup

### For fetching the data

#### Installing the required packages
conda env create -f environment.yaml

conda activate pfas

#### Updating the environment.yaml files

conda env update -f environment.yml --prune

## Scripts 

### Running the script to fetch data
Fetch the data from the PACE ICE cluster from the directory `/storage/ice-shared/cs8903onl/mussmann-pfas/data/` and copied into the `data/` directory.

```
cd pfas-environment-cleanup
python3 scripts/fetch_data.py
```

If you would like to fetch the data from the ICE cluster:

```
cd pfas-environment-cleanup
python3 scripts/fetch_data.py --from-ice
```

This would prompt you to enter your username and password. Make sure to be connected to the VPN.

## DFT Calculation Process

### Quantum Espresso Input Production
An automated pipeline for automatically converting a compound of interest into a Quantum Espresso input is in development, but this is the manual process for the time being. 
1. Using [PubChem](https://pubchem.ncbi.nlm.nih.gov/), find the SMILES format of the compound of interest.
2. Using [OpenBabel](https://openbabel.org/index.html), convert the SMILES string into an MDL Molfile (.mol).
3. Using [VESTA](https://jp-minerals.org/vesta/en/), convert the MDL Molfile into a Crystallographic Information File (.cif), inputting and modifying crystal orientations/morphologies as necessary. 
4. Using [cif2cell](https://pypi.org/project/cif2cell/), convert the Crystallographic Information File into a Quantum Espresso input file (.in).
5. Modify the input file with pseudopotentials for constituent atoms and standard run parameters based on those atoms.
6. Run PWSCF simulation with input file via [Quantum Espresso](https://www.quantum-espresso.org/Doc/INPUT_PW.html) (pw.x). 
Note: This method represents a slight workaround from the typical DFT calculation process designed around VASP POSCAR files, used with VASP instead of Quantum Espresso. It is possible that some information is lost or improperly assumed in this conversion process, particularly at the .mol to .cif file conversion step with VESTA.


The current pipeline automates the full process from SMILES to adsorption energy using Quantum ESPRESSO.

**Steps:**

1. Convert SMILES → `.mol` using Open Babel  
2. Convert `.mol` → `.cif` using pymatgen  
3. Convert `.cif` → QE input (`.in`) using `cif2cell`  
4. Patch QE input with:
   - PBE functional
   - Cutoffs (e.g., `ecutwfc`, `ecutrho`)
   - `K_POINTS = 1 1 1` (molecular system)
   - Pseudopotential paths
5. Run `pw.x` for:
   - Adsorbent
   - PFAS
   - Adsorbent–PFAS complex
6. Parse total energies and compute:
E_ads = E_complex - E_adsorbent - E_PFAS

### Running a Case
```
python qespresso_pipeline/run_adsorption_case.py \
  --case-name <case_name> \
  --adsorbent-name <adsorbent_name> \
  --pfas-name <pfas_name> \
  --adsorbent-smiles "<SMILES>" \
  --pfas-smiles "<SMILES>" \
  --compound-root compounds \
  --workdir dft_cases \
  --pseudo-dir qespresso_pipeline/Pseudopotentials \
  --mode cluster \
  --pw-command "pw.x"
```

#### Reusing Existing Calculations

You can skip parts of the workflow if outputs already exist:

```
--skip-ads → reuse adsorbent
--skip-pfas → reuse PFAS
--skip-complex → reuse complex
```
- Providing PFAS Energy Directly

#### If PFAS energy is already known:
```
--pfas-energy-ry <value>
```
- Skips PFAS calculation
- Uses provided energy in adsorption calculation

#### Directory Structure
```
/storage/ice-shared/cs8903onl/mussmann-pfas/
├── compounds/
│   ├── adsorbents/
│   │   └── <adsorbent_name>/
│   └── pfas/
│       └── <pfas_name>/
├── dft_cases/
│   └── <case_name>/
│       └── complex/
├── dft_runs/
│   └── <case_name>/
│       ├── job.sbatch
│       ├── meta.json
│       └── outputs/
├── qespresso_pipeline/
├── scripts/
├── qe_environment.yaml
└── run_dft_workflow.sh
```

#### Important Scripts

- `run_adsorption_case.py`
Runs a full adsorption-energy case.

It prepares and runs:

Adsorbent
PFAS
Adsorbent–PFAS complex

Then it parses total energies and computes adsorption energy.

*Molecular adsorbent example*

```
python qespresso_pipeline/run_adsorption_case.py \
  --case-name imidazolium_tfa \
  --adsorbent-name imidazolium \
  --pfas-name tfa \
  --adsorbent-source smiles \
  --adsorbent-smiles "ADSORBENT_SMILES_HERE" \
  --pfas-smiles "PFAS_SMILES_HERE" \
  --system-type molecule \
  --mode production \
  --compound-root compounds \
  --workdir dft_cases \
  --pseudo-dir qespresso_pipeline/Pseudopotentials \
  --pw-command "pw.x"
```

*Periodic Adsorbent example*

```
python qespresso_pipeline/run_adsorption_case.py \
  --case-name pfoa_go \
  --adsorbent-name go_barker \
  --pfas-name pfoa \
  --adsorbent-source cif \
  --adsorbent-cif qespresso_pipeline/benchmarks/go_barker_like.cif \
  --pfas-smiles "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F" \
  --system-type periodic \
  --mode production \
  --compound-root compounds \
  --workdir dft_cases \
  --pseudo-dir qespresso_pipeline/Pseudopotentials \
  --pw-command "pw.x"
```

- `run_dft_workflow.sh`

Cluster-side workflow script.

This is the entrypoint used by SLURM jobs. It:

loads Anaconda
creates or updates the QE conda environment
reads environment variables from the SLURM job
runs run_adsorption_case.py

Usually, you do not run this manually. It is called by dft_wrapper.py.

- `dft_wrapper.py`

Submits DFT jobs to the cluster.

It:

connects to the cluster
creates a case directory under dft_runs/
writes a SLURM job script
exports required variables
submits the job using sbatch

Example for PFOA on GO:

```
python3 scripts/dft_wrapper.py \
  --user arai304 \
  --cluster login-ice.pace.gatech.edu \
  --case-name pfoa_go_check \
  --adsorbent-name go_barker \
  --adsorbent-source cif \
  --adsorbent-cif /storage/ice-shared/cs8903onl/mussmann-pfas/qespresso_pipeline/benchmarks/go_barker_like.cif \
  --pfas-name pfoa \
  --pfas-smiles "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F" \
  --system-type periodic \
  --mode production \
  --ecutwfc 40 \
  --ecutrho 200 \
  --input-dft vdW-DF-cx \
  --kpts 2 2 1 \
  --cluster-root /storage/ice-shared/cs8903onl/mussmann-pfas \
  --runs-subdir dft_runs \
  --submit-if-missing \
  --cpus 4 \
  --mem-gb 64 \
  --time 12:00:00 \
  --workflow-script /storage/ice-shared/cs8903onl/mussmann-pfas/run_dft_workflow.sh
```