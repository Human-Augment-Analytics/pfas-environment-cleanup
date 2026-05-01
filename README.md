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

### Manual DFT Simulation

For tuning purposes, it will likely be necessary to manually create a DFT input file from a CIF file, created either via ase, pymatgen, or sourced from a crystallographic database. You begin by running a command of this following structure to create an input file:

```
cif2cell [input_file].cif -p quantum-espresso -o [output_file].in
```

This gives you a generic input file. Depending on if you have a molecular or periodic/metallic structure, we then need to append a certain number of things. For molecular, the structure may look something like this:

```
&CONTROL
  calculation='relax', ! to denote the type of calculation you want to run
  outdir='./Outputs', ! where it sends metadata, run results, etc.
  prefix='tfanoh_isolated', ! title of results
  pseudo_dir='./Pseudopotentials', ! where it sources pseudopotential files for the run
  verbosity='low',
  tprnfor=.true., ! additional force/stress calculations
  tstress=.true.,
  forc_conv_thr=7.7D-4, ! global convergence thresholds
  etot_conv_thr=7.3D-7
/
&SYSTEM
  ibrav = 1
  A = 15.0  ! for molecules, you need vacuum on the sides to prevent spurious interactions
  nat = 7 ! that means rescaling the ATOMIC_POSITIONS parameters
  ntyp = 3
  tot_charge = -1.0 ! specifying the ionic nature 
  assume_isolated = 'martyna-tuckerman' ! this is a correction factor for isolated molecular systems
  ecutwfc=60, ! energy cutoffs, the higher the more accurate, but the more memory/time required
  ecutrho=480,
  input_dft='pbe',
  occupations='fixed',
  vdw_corr='grimme-d3' ! van der waals force correction
/
&ELECTRONS
  conv_thr=1d-07, ! self-consistent convergence threshold
  mixing_beta=0.3, ! mixing factor (akin to a learning rate, too high is unstable, too low is slow)
/
&IONS
  ion_dynamics='bfgs', ! relaxation dynamics for ions
/
CELL_PARAMETERS {angstrom}
  15.00000000000000   0.000000000000000   0.000000000000000 
  0.000000000000000   15.00000000000000   0.000000000000000 
  0.000000000000000   0.000000000000000   15.00000000000000 
ATOMIC_SPECIES
   F   18.99800   F.UPF ! PAW pseudopotential files in the given directory
   O   15.99900   O.UPF
   C   12.01060   C.UPF
ATOMIC_POSITIONS {angstrom}
F   8.071326   5.984002   7.794029 ! note that these parameters are clearly bounded by vacuum
F   6.411133   7.035002   8.740828 ! note that the cell goes from [0,15], and this is centred
F   6.351331   6.595001   6.598528
O   7.353828   9.388298   7.092531
O   9.255626   8.192599   7.341530
C   7.891528   8.311999   7.321427
C   7.165231   6.993100   7.611128
K_POINTS gamma ! equivalent to a 1 1 1 grid, appropriate for isolated systems
```

Conversely, if you have a periodic/metallic structure, you will need some additions, such as:
```
&CONTROL ! these only describe additions that must be made
  tefield=.true., ! electric field potential for lattices
  dipfield=.true., ! dipole correction factor for lattices
  max_seconds=40000, ! periodic structures run longer, so this sets a save point for restarts
&SYSTEM
  occupations='smearing', ! Gaussian smoothing filter, necessary for metals
  smearing='mv', ! specific form of cold smearing
  degauss=0.01,
  nspin=2, ! spin-polarization, due to magnetism of system
  edir=3, ! electric field/dipole correction axis
  emaxpos=0.9, ! maximum electric potential, placed in vacuum above all atoms
  eopreg=0.1, ! zone of decline for dipole factor, placed in vacuum under all atoms
  starting_magnetization(1)=0.3 ! initial magnetization for metal (in this case iron)
&ELECTRONS
  mixing_mode='local-TF', ! inhomogeniety correction for adsorbants
  electron_maxstep=200 ! extending calculation steps to promote convergence
/
&IONS
  wfc_extrapolation='second_order', ! wave function and potential optimizers
  pot_extrapolation='second_order', ! increases runtime drastically, trading small portion of accuracy
/
ATOMIC_POSITIONS {crystal} ! note the additional numbers added after the positions
Fe   0.833333333333333   0.333333333333333   0.355649309796759 0 0 0 ! 0s represent fixed structure 
 C   0.523961155391464   0.283367120111382   0.752928148328609 1 1 1 ! 1s represent free movement
 K_POINTS {automatic} ! we need fixed layers to simulate bulk behaviour 
  5 5 1 0 0 0 ! periodic structures need K-mesh for accurate energy calc, 5x5x1 is good for slabs
```

These files can be run on PACE ICE with parallelization as follows:
```
module load quantum-espresso
module load openmpi
mpirun -np [number_of_processors] pw.x -in [input_file].in > [output_file].out
```
