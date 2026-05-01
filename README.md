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




# ML Models Description



1. **Benchmark validation on public molecular datasets**  
   These scripts test whether the GNN and tabular model designs are at least directionally reasonable on established tasks before we trust them on PFAS/QE data.

2. **BO/UCB validation on benchmark datasets**  
   These scripts simulate a Bayesian Optimization loop on known public datasets to check whether the acquisition logic, uncertainty estimates, and ranking behavior make sense.

3. **Placeholder PFAS-specific models and BO scaffolds**  
   These scripts are the PFAS-facing prototypes. They currently use placeholder data and/or hidden placeholder labels, so they are best treated as workflow scaffolds and internal comparisons, not final scientific conclusions.

### Datasets:
  - `ESOL` - water solubility (in log solubility)
  - `Lipo` - water distribution coefficient (logD at pH 7.4)
  - `FreeSolv` - hydration free energy of small molecules
  - `AQSOL` - aqueous solubility (LogS)

---

## 1) Benchmark validation scripts



### `benchmark_gnn_regression.py`
**Category:** benchmark regression / GNN validation

**What it does**
- Trains a **GCN** or **GAT** regressor on benchmark molecular datasets:
  - `ESOL`
  - `Lipo`
  - `FreeSolv`
  - `AQSOL`
- Uses PyTorch Geometric molecular graphs directly.
- Builds train/val/test splits:
  - `AQSOL`: uses predefined splits
  - `ESOL`, `Lipo`, `FreeSolv`: uses seeded random splits
- Tracks validation MAE for early stopping.
- Saves a metrics JSON with train settings, best epoch, validation history, and test metrics.

**Why it exists**
- This is the main **graph-based sanity check**.
- It tells us whether simple molecular GNNs can learn real chemistry/property benchmarks before we use similar architectures in PFAS workflows.

**Output**
- `benchmark_outputs/{dataset}_{model}_metrics.json`

**Run one specific case**
```
python benchmark_gnn_regression.py --dataset ESOL --model GAT
python benchmark_gnn_regression.py --dataset AQSOL --model GCN
```


**Run all GNN benchmark regression cases**
```
$datasets = "ESOL","Lipo","FreeSolv","AQSOL"
$models   = "GCN","GAT"

foreach ($dataset in $datasets) {
  foreach ($model in $models) {
    python benchmark_gnn_regression.py `
      --dataset $dataset `
      --model $model `
      --data-root data `
      --outdir benchmark_outputs
  }
}
```

---

### `benchmark_ml_regression.py`
**Category:** benchmark regression / tabular baseline validation

**What it does**
- Trains tabular regressors on the **same datasets** as the GNN benchmark:
  - `ESOL`
  - `Lipo`
  - `FreeSolv`
  - `AQSOL`
- Converts each molecule graph into a **graph-summary feature vector**.
- Supports:
  - `rf` = Random Forest
  - `extratrees` = Extra Trees
  - `hgb` = HistGradientBoostingRegressor
- Performs a small randomized hyperparameter search.
- Saves a metrics JSON with best params and test metrics.

**Why it exists**
- This is the main **non-GNN baseline**.
- It tells us whether simpler models on derived features outperform or match the GNNs.
- It is especially useful because strong tabular baselines can be hard to beat on small or medium molecular datasets.

**Output**
- `benchmark_outputs/{dataset}_{model}_tabular_metrics.json`

**Run one specific case**
```bash
python benchmark_ml_regression.py --dataset ESOL --model extratrees
python benchmark_ml_regression.py --dataset AQSOL --model hgb
```

**Run all tabular benchmark regression cases**
```
$root = ".\pfas-environment-cleanup\benchmarks"

$datasets = @("ESOL", "FreeSolv", "Lipo", "AQSOL")
$mlModels = @("rf", "hgb", "xgb")
$gnnModels = @("GCN", "GAT")

foreach ($dataset in $datasets) {
    foreach ($model in $mlModels) {
        python "$root\benchmark_ml_regression.py" --dataset $dataset --model $model
    }
}

foreach ($dataset in $datasets) {
    foreach ($model in $gnnModels) {
        python "$root\benchmark_gnn_regression.py" --dataset $dataset --model $model
    }
}
```



---

## 2) BO/UCB benchmark scripts

### `benchmark_gnn_bo_ucb.py`
**Category:** benchmark BO/UCB validation / GNN surrogate

**What it does**
- Treats a benchmark dataset as a **closed candidate pool**.
- Hides labels until selected, so the script can mimic an active-learning / BO loop.
- Uses a **GNN ensemble** (`gcn` or `gat`) to produce:
  - predictive mean `mu`
  - predictive uncertainty `sigma`
- Supports acquisition modes:
  - `ucb`
  - `greedy`
  - `random`
- Supports objectives:
  - `maximize`
  - `minimize`
- Saves round-by-round ranked pools, selected batches, BO history, and summary metrics.

**Why it exists**
- This is the main **graph-based BO sanity check**.
- It tests whether a GNN + uncertainty + acquisition loop behaves sensibly before applying the same logic to PFAS candidate selection.

**Outputs**
- `benchmark_bo_outputs/{dataset}_{model}_{acquisition}_bo_metrics.json`
- `benchmark_bo_outputs/{dataset}_{model}_{acquisition}_bo_history.csv`
- `benchmark_bo_outputs/{dataset}_{model}_{acquisition}_bo_selected_all.csv`
- plus per-round ranked/selected CSVs

**Run one specific case**
```bash
python benchmark_gnn_bo_ucb.py --dataset ESOL --model gat --acquisition ucb
python benchmark_gnn_bo_ucb.py --dataset FreeSolv --model gcn --acquisition greedy
python benchmark_gnn_bo_ucb.py --dataset AQSOL --model gat --acquisition random
```


**Run all GNN BO/UCB cases**
```powershell
$root = ".\pfas-environment-cleanup\benchmarks"

$datasets = @("ESOL", "FreeSolv", "Lipo", "AQSOL")
$mlModels = @("rf", "hgb", "xgb")
$gnnModels = @("gcn", "gat")
$acqs = @("random", "greedy", "ucb")

foreach ($dataset in $datasets) {
    foreach ($model in $mlModels) {
        foreach ($acq in $acqs) {
            python "$root\benchmark_ml_bo_ucb.py" --dataset $dataset --model $model --acquisition $acq
        }
    }
}

foreach ($dataset in $datasets) {
    foreach ($model in $gnnModels) {
        foreach ($acq in $acqs) {
            python "$root\benchmark_gnn_bo_ucb.py" --dataset $dataset --model $model --acquisition $acq
        }
    }
}
```

**How to run a specific “case”**
- A “case” here usually means a **dataset + model + acquisition setting**.
- If you want a minimization case, just switch the objective:
```bash
python benchmark_gnn_bo_ucb.py --dataset ESOL --model gat --acquisition ucb --objective minimize
```

---

### `benchmark_ml_bo_ucb.py`
**Category:** benchmark BO/UCB validation / tabular surrogate

**What it does**
- Runs the same **offline BO pool search idea** as the GNN BO script, but with tabular surrogate models.
- Supports:
  - `rf`
  - `extratrees`
  - `hgb`
- Builds an ensemble from bootstrapped tabular models to estimate `mu` and `sigma`.
- Supports:
  - `ucb`
  - `greedy`
  - `random`
- Supports:
  - `maximize`
  - `minimize`

**Why it exists**
- This is the **tabular BO baseline** for the active-learning logic.
- It tells us whether the BO ranking behavior works better with simpler surrogate families.

**Outputs**
- `benchmark_bo_outputs/{dataset}_{model}_{acquisition}_bo_metrics.json`
- `benchmark_bo_outputs/{dataset}_{model}_{acquisition}_bo_history.csv`
- `benchmark_bo_outputs/{dataset}_{model}_{acquisition}_bo_selected_all.csv`
- plus per-round ranked/selected CSVs

**Run one specific case**
```bash
python benchmark_ml_bo_ucb.py --dataset ESOL --model rf --acquisition ucb
python benchmark_ml_bo_ucb.py --dataset AQSOL --model hgb --acquisition greedy
python benchmark_ml_regression.py --dataset ESOL --model xgb
```

**Run all tabular BO/UCB cases**
```powershell
$datasets     = "ESOL","Lipo","FreeSolv","AQSOL"
$models       = "rf","xgb","hgb"
$acquisitions = "ucb","greedy","random"

foreach ($dataset in $datasets) {
  foreach ($model in $models) {
    foreach ($acquisition in $acquisitions) {
      python benchmark_ml_bo_ucb.py `
        --dataset $dataset `
        --model $model `
        --acquisition $acquisition `
        --objective maximize `
        --data-root data `
        --outdir benchmark_bo_outputs
    }
  }
}
```

**How to run a specific “case”**
```bash
python benchmark_ml_bo_ucb.py --dataset AQSOL --model rf --acquisition ucb --objective minimize
```
### run both BO benchmarks:
```
$datasets = @("ESOL", "FreeSolv", "Lipo", "AQSOL")
$mlModels = @("rf", "xgb", "hgb")
$gnnModels = @("gcn", "gat")
$acqs = @("ucb", "greedy", "random")

foreach ($dataset in $datasets) {
    foreach ($model in $mlModels) {
        foreach ($acq in $acqs) {
            python .\pfas-environment-cleanup\benchmarks\benchmark_ml_bo_ucb.py --dataset $dataset --model $model --acquisition $acq
        }
    }
}

foreach ($dataset in $datasets) {
    foreach ($model in $gnnModels) {
        foreach ($acq in $acqs) {
            python .\pfas-environment-cleanup\benchmarks\benchmark_gnn_bo_ucb.py --dataset $dataset --model $model --acquisition $acq
        }
    }
}
```

---

## 3) Placeholder PFAS-specific scripts

### `per_pfas_rf_model.py`
**Category:** placeholder PFAS regression / tabular per-PFAS baseline

**What it does**
- Loads `quantum_espress_placeholder.csv`.
- Cleans numeric and boolean columns.
- Uses a **per-PFAS split** instead of one shared global model.
- Trains:
  1. a single RF per PFAS
  2. an RF ensemble per PFAS
- Reports weighted overall MAE/R² and saves JSON summaries.

**Why it exists**
- This is the first PFAS-facing tabular scaffold.
- It answers: “If we split by PFAS identity and use simple descriptor-based models, do we get a stable baseline workflow?”

**Important limitation**
- It uses **`y_placeholder`**, so it is a workflow/prototype benchmark, not final QE-grounded science.

**Outputs**
- `models/per_pfas_rf_metrics.json`
- `models/per_pfas_rf_ensemble_metrics.json`

**Run**
```bash
python per_pfas_rf_model.py
```

**How it should be interpreted**
- Use it as a **PFAS workflow baseline**.
- Compare it against the public benchmark behavior, but do **not** treat it as official adsorption-performance evidence.

---

### `per_pfas_gnn_model.py`
**Category:** placeholder PFAS regression / graph per-PFAS baseline

**What it does**
- Loads PFAS rows from a CSV with `SMILES`, `pfas_id`, and `y_placeholder`.
- Converts each SMILES string into a molecular graph.
- Trains **single-model or ensemble per-PFAS GNNs**.
- Supports:
  - `GCN`
  - `GAT`
- Saves separate JSON files for single-model and ensemble runs.

**Why it exists**
- This is the GNN counterpart to `per_pfas_rf_model.py`.
- It tests whether graph-based PFAS-specific modeling is workable at the pipeline level before official labels arrive.

**Outputs**
- `models/per_pfas_{model}_metrics.json`
- `models/per_pfas_{model}_ensemble_metrics.json`

**Run one specific case**
```bash
python per_pfas_gnn_model.py --model GAT
python per_pfas_gnn_model.py --model GCN
```

**More explicit examples**
```bash
python per_pfas_gnn_model.py \
  --data data/quantum_espress_placeholder.csv \
  --outdir models \
  --model GAT \
  --ensemble-M 3
```

---

### `bo_ucb_selector.py`
**Category:** placeholder PFAS BO scaffold / offline selection prototype

**What it does**
- Uses the **RF ensemble** from `per_pfas_rf_model.py` as the surrogate.
- Splits placeholder PFAS data into labeled seed and unlabeled pool.
- Ranks pool items with:
  - `ucb`
  - `greedy`
  - `random`
- Selects a batch per PFAS.
- Evaluates selections **offline** by reading the hidden `y_placeholder` value.
- Saves ranked pools, selected batches, and BO history.

**Why it exists**
- This is the earliest PFAS-specific BO selection prototype.
- It checks whether the active-learning loop mechanics work end-to-end on placeholder data.

**Important limitation**
- This script is **not yet a general CLI benchmark runner**.
- It has hardcoded paths and currently sets:
```python
acquisition = "greedy"
```
inside `main()`.

**Outputs**
- `proposals/{acquisition}_ranked_pool_round{round}.csv`
- `proposals/{acquisition}_selected_batch_round{round}.csv`
- `proposals/bo_history_{acquisition}.json`
- `proposals/bo_history_{acquisition}.csv`

**Run**
```bash
python bo_ucb_selector.py
```

**How it should evolve**
- Replace the offline evaluator with actual QE job submission + result parsing.
- Make acquisition and path settings command-line arguments.
- Replace `y_placeholder` with the official target once available.

---

### `fast_tree_based_training_demo.py`
**Category:** quick prototype / simple tabular baseline

**What it does**
- Trains a fast tabular model on the placeholder CSV.
- Supports:
  - `hgb`
  - `extratrees`
  - `rf`
- Uses stratified regression splitting and a small randomized search.
- Saves a joblib bundle and a metrics JSON.

**Why it exists**
- This is a **fast experimentation script**.
- It is useful for quickly checking whether engineered features are learnable before running the more structured per-PFAS scripts.

**Run**
```bash
python fast_tree_based_training_demo.py --in data/quantum_espress_placeholder.csv --model hgb
python fast_tree_based_training_demo.py --in data/quantum_espress_placeholder.csv --model extratrees
python fast_tree_based_training_demo.py --in data/quantum_espress_placeholder.csv --model rf
```

---

## 4) Result artifact files currently included

The uploaded JSON metric files are not new model definitions; they are **saved outputs** from the benchmark scripts.

- `esol_gat_metrics.json`
- `esol_gcn_metrics.json`
- `esol_xgb_tabular_metrics.json`
- `esol_rf_tabular_metrics.json`
- `esol_hgb_tabular_metrics.json`
- `aqsol_gat_metrics.json`
- `aqsol_gcn_metrics.json`
- `aqsol_xgb_tabular_metrics.json`
- `aqsol_rf_tabular_metrics.json`
- `aqsol_hgb_tabular_metrics.json`
- `freesolv_gat_metrics.json`
- `freesolv_extratrees_tabular_metrics.json`

### High-level snapshot from the uploaded benchmark outputs

| Dataset | Best uploaded model | Test R² | Takeaway |
|---|---:|---:|---|
| ESOL | ExtraTrees | 0.885 | Strong tabular baseline; GAT is competitive but not best |
| AQSOL | HGB | 0.608 | Tabular models clearly beat current GNN runs here |
| FreeSolv | ExtraTrees | 0.775 | Uploaded tabular result beats uploaded GAT result |

### Practical interpretation
Right now, the benchmark evidence points to this working conclusion:

- **Tabular baselines are the strongest default benchmark family so far.**
- **GAT can be competitive on some easier benchmark settings (like ESOL),** but the GNNs are not consistently ahead.
- So when we move to PFAS/QE data, the safest comparison framing is:
  1. benchmark GNN vs benchmark tabular on public datasets
  2. placeholder PFAS GNN vs placeholder PFAS tabular on the mock target
  3. official PFAS GNN vs official PFAS tabular once real adsorption labels arrive

That gives us a clean “same question, different data maturity” comparison path.

---

## 5) How these categories fit together

### A. Testing whether the model design is “semi right”
Use:
- `benchmark_gnn_regression.py`
- `benchmark_ml_regression.py`

These answer:
- Can the chosen architectures learn known molecular-property tasks?
- Is the graph model competitive?
- Are simple tabular baselines already very strong?

### B. Testing whether the BO/UCB logic is “semi right”
Use:
- `benchmark_gnn_bo_ucb.py`
- `benchmark_ml_bo_ucb.py`

These answer:
- Does uncertainty-aware ranking improve over greedy/random?
- Does the acquisition loop find high-value candidates faster?
- Does the BO logic work better for some surrogate families than others?

### C. Testing whether the PFAS-specific workflow is wired correctly
Use:
- `per_pfas_rf_model.py`
- `per_pfas_gnn_model.py`
- `bo_ucb_selector.py`
- `fast_tree_based_training_demo.py`

These answer:
- Can the PFAS pipeline ingest PFAS data, split it by PFAS family, train models, and score candidates?
- Do the uncertainty and ranking mechanics run end-to-end?
- What should be replaced once QE outputs become official?

---

## 6) Plan for when PFAS data comes:

### Minimal required edits
1. **Replace the input CSV**
   - Swap out `quantum_espress_placeholder.csv` for the official data file.

2. **Replace `y_placeholder`**
   - Change the target column to the real adsorption/binding/formation-energy target you want to model.

3. **Revisit feature columns**
   - For tabular PFAS models, confirm which descriptor columns are still valid.
   - Add or remove engineered columns as needed.

4. **Replace offline BO evaluation**
   - In `bo_ucb_selector.py`, replace:
     - hidden truth lookup from `y_placeholder`
   - with:
     - QE job submission / result collection / parser logic.

5. **Add CLI control where needed**
   - `bo_ucb_selector.py` especially should be upgraded to accept:
     - acquisition mode
     - beta
     - rounds
     - batch size
     - input path
     - output path

6. **Re-run the same comparison ladder**
   - official tabular PFAS baseline
   - official GNN PFAS baseline
   - official BO/UCB loop with each surrogate family

---
