# PFAS GNN ML/BO Project Handoff

## Purpose

> This repository contains a prototype ML + Bayesian Optimization pipeline for PFAS candidate screening. Because final Quantum Espresso adsorption labels were not yet available, the current PFAS-specific scripts use `y_placeholder` as a stand-in target. The placeholder target is used only to validate data loading, per-PFAS model training, uncertainty estimation through ensembles, and offline BO/UCB candidate selection. Scientific conclusions should not be drawn from placeholder-target performance. Once real QE outputs are available, `y_placeholder` should be replaced by the parsed adsorption/binding target and `evaluate_selected_offline()` should be replaced by the QE job submission/parser step.

---

## What has been completed

### 1. Literature and project framing

The project direction came from the idea that PFAS simulation is expensive, so ML should act as a surrogate model and BO should decide which candidates deserve the next expensive simulations. This matches the original proposal: start with molecular featurization, train surrogate models, use uncertainty-aware Bayesian Optimization, and compare against random and greedy selection.

The relevant PFAS/DFT/ML literature supports this direction. The PFAS adsorption paper shows that DFT and ML can be combined to accelerate PFAS catalyst discovery, using chemically meaningful descriptors and ML models to capture adsorption/correlation behavior. Mole-BERT and related molecular GNN literature motivate using graph-based molecular representations because molecules are naturally graphs with atoms as nodes and bonds as edges.

### 2. Placeholder PFAS regression models

Two per-PFAS placeholder model families were implemented:

#### `per_pfas_rf_model.py`

This is the tabular baseline model. It uses PubChem/RDKit-style numeric descriptors and functional-group flags, filters by PFAS identity, and trains one model per PFAS type.

Current PFAS list:

- PFOA
- PFOS
- PFBA
- PFBS
- PFPrA
- HFPO-DA
- TFSI
- TFA

What it does:

- Loads `data/quantum_espress_placeholder.csv`
- Cleans numeric descriptor columns
- Converts `flag_*` boolean columns into 0/1
- Uses only molecular descriptors and flags as features
- Does **not** use PFAS one-hot features inside a per-PFAS model, because each model is already filtered to one PFAS
- Trains single Random Forest models
- Trains RF ensembles using bootstrapped training samples
- Reports per-PFAS MAE/R² and weighted overall MAE/R²
- Saves metrics into `models/`

This is the primary baseline because it is simple, interpretable, fast, and stable.

#### `per_pfas_gnn_model.py`

This is the graph neural network version of the per-PFAS model. It converts SMILES into PyTorch Geometric molecular graphs and trains GCN or GAT regressors.

What it does:

- Loads and validates SMILES
- Converts SMILES to graph objects with `torch_geometric.utils.from_smiles`
- Uses atoms as graph nodes and bonds as graph edges
- Trains one GCN/GAT model per PFAS type
- Supports target normalization
- Uses early stopping based on validation MAE
- Supports ensembles over multiple seeds
- Uses ensemble mean as prediction `mu`
- Uses ensemble standard deviation as uncertainty `sigma`
- Saves single-model and ensemble metrics into `models/`

This is a more molecule-structure-aware surrogate, but it is more expensive and more sensitive to data quantity/quality than RF.

### 3. Bayesian Optimization / UCB selector

#### `bo_ucb_selector.py`

This script implements the current placeholder active-learning loop.

What it does:

- Creates an initial labeled seed set per PFAS
- Keeps the rest of the data as a candidate pool
- Trains an RF ensemble on the labeled set
- Predicts `mu` and `sigma` on the pool
- Scores candidates using:
  - UCB: exploration + exploitation
  - greedy: predicted mean only
  - random: random baseline
- Selects a batch per PFAS
- Uses the hidden placeholder label as a stand-in for a future QE result
- Updates the labeled/pool state
- Saves ranked pools, selected batches, and BO history files

The current version is still an offline simulation. The function `evaluate_selected_offline()` is the key replacement point. In the future, this should be replaced with:

1. submit candidate to Quantum Espresso pipeline,
2. wait for the output,
3. parse adsorption/binding target,
4. append the true observed value back into the labeled set.

### 4. Benchmark regression scripts

Benchmarks were added to avoid only evaluating on synthetic placeholder PFAS labels. These scripts ask: “Do the model families behave reasonably on known molecular-property regression datasets?”

#### `benchmark_gnn_regression.py`

Supports:

- ESOL
- Lipo
- FreeSolv
- AQSOL

Models:

- GCN
- GAT

Outputs:

- MAE
- RMSE
- R²
- train/val/test metadata
- training history
- JSON metrics files

Purpose:

This checks whether the GNN implementation can train on public molecular graph datasets, not whether it solves PFAS adsorption yet.

#### `benchmark_ml_regression.py`

Supports the same datasets and converts each molecular graph into a tabular feature vector using graph summary statistics.

Models:

- Random Forest
- HistGradientBoosting
- XGBoost

Outputs:

- MAE
- RMSE
- R²
- chosen hyperparameters
- JSON metrics files

Purpose:

This provides non-GNN baselines against the GNN models. It also tests whether simpler models can compete with graph models when using basic graph-derived summaries.

### 5. Benchmark BO/UCB scripts

These scripts treat each benchmark dataset as a closed candidate pool where labels are hidden until selected.

#### `benchmark_ml_bo_ucb.py`

Surrogate models:

- RF
- HGB
- XGB

Acquisition modes:

- UCB
- greedy
- random

Purpose:

This tests whether the active-learning / BO logic finds good candidates more efficiently than random or greedy search on known datasets.

#### `benchmark_gnn_bo_ucb.py`

Surrogate models:

- GCN
- GAT

Acquisition modes:

- UCB
- greedy
- random

Purpose:

This is the GNN counterpart of the tabular BO benchmark. It uses GNN ensembles to produce `mu` and `sigma`, then ranks benchmark candidates using acquisition functions.

---

## Why GNNs?

GNNs are a natural modeling choice because molecules are graphs:

- atoms are nodes,
- bonds are edges,
- molecular properties depend on local and global structure,
- message passing lets each atom representation incorporate neighboring atom/bond context.

This is especially relevant for PFAS because functional groups, fluorination patterns, chain length, and local chemical environments can affect adsorption or degradation behavior. A GNN can potentially learn these structural relationships without requiring every relationship to be manually encoded.

GNNs are not guaranteed to outperform simpler baselines, especially with small or noisy datasets. Their role in this project is to test whether structure-aware learning becomes useful once enough real QE-labeled data exists.

---

## Baselines

Baselines help with interpreting whether GNNs and BO are actually helping.

The project currently uses several kinds of baselines:

1. **Random Forest per-PFAS baseline**
   - Fast, stable, interpretable.
   - Good first model when the dataset is small.
   - Gives a sanity-check performance floor.

2. **Gradient boosting / XGBoost benchmark baselines**
   - Strong tabular ML models.
   - Often hard to beat on descriptor-based chemistry tasks.

3. **Random acquisition**
   - Tests whether BO is better than just selecting arbitrary candidates.

4. **Greedy acquisition**
   - Tests whether uncertainty is helpful.
   - Greedy only selects by predicted mean, while UCB also considers uncertainty.

5. **Public molecular-property datasets**
   - Tests whether code works on real labels before official PFAS/QE labels exist.

A valid conclusion so far is not “GNNs are best.” The better conclusion is:

> The project has established a model-comparison and active-learning framework. RF/XGBoost-style models are important practical baselines, GCN/GAT models provide a graph-based alternative, and UCB/greedy/random comparisons test whether uncertainty-aware candidate selection improves sample efficiency.

---

## What the placeholder PFAS implementations represent

The placeholder PFAS scripts are scaffolding for the real workflow.

They represent:

- how data should be loaded,
- how one model per PFAS can be trained,
- how uncertainty can be estimated,
- how a candidate pool can be ranked,
- how selected batches can be saved,
- how the labeled set and pool should update after each simulated round.

They do **not** represent final scientific PFAS adsorption conclusions because `y_placeholder` is not an official Quantum Espresso adsorption target.

Expected replacement later:

```text
y_placeholder  -->  QE adsorption energy / binding score / degradation-relevant target
evaluate_selected_offline()  -->  real QE job submission + output parser
placeholder candidate CSV  -->  official candidate library with real metadata
```

Once real QE values exist, the same code structure should still apply.

---

## What was tried

1. **Basic molecular GNN prototype**
   - File: `basic_gnn_molecule.py`
   - Dataset: ESOL
   - Models: GCN and GAT
   - Purpose: learn PyTorch Geometric basics and verify that GNN training works.
   - Recommendation: remove from the final project folder because its functionality is superseded by `benchmark_gnn_regression.py` and `per_pfas_gnn_model.py`.

2. **Per-PFAS RF baseline**
   - More production-relevant baseline for placeholder PFAS tabular data.
   - Should stay.

3. **Per-PFAS GNN**
   - More advanced structure-aware model.
   - Should stay, but should be treated as experimental until real QE labels exist.

4. **Offline BO loop**
   - Implemented with RF ensembles first because RF is faster and stable.
   - Should stay.

5. **Public benchmark regression**
   - Added to check whether the GNN and ML model implementations work on known molecular datasets.
   - Should stay.

6. **Public benchmark BO**
   - Added to compare UCB, greedy, and random sample-selection behavior.
   - Should stay if project includes BO/active learning.

---

## How to run the current scripts

Assuming scripts are inside `gnn_ML/` and data is in `data/`.

### Per-PFAS Random Forest baseline

```bash
python gnn_ML/per_pfas_rf_model.py
```

Expected outputs:

```text
models/per_pfas_rf_metrics.json
models/per_pfas_rf_ensemble_metrics.json
```

### Per-PFAS GNN

```bash
python gnn_ML/per_pfas_gnn_model.py --model GAT
python gnn_ML/per_pfas_gnn_model.py --model GCN
```

Optional ensemble size:

```bash
python gnn_ML/per_pfas_gnn_model.py --model GAT --ensemble-M 3
```

Expected outputs:

```text
models/per_pfas_gat_metrics.json
models/per_pfas_gat_ensemble_metrics.json
```

### PFAS placeholder BO loop

Check the `acquisition` variable inside `bo_ucb_selector.py`. Set it to one of:

```python
acquisition = "ucb"
acquisition = "greedy"
acquisition = "random"
```

Then run:

```bash
python gnn_ML/bo_ucb_selector.py
```

Expected outputs:

```text
proposals/<acquisition>_ranked_pool_round0.csv
proposals/<acquisition>_selected_batch_round0.csv
proposals/bo_history_<acquisition>.json
proposals/bo_history_<acquisition>.csv
```

### GNN regression benchmarks

```bash
python gnn_ML/benchmark_gnn_regression.py --dataset ESOL --model GAT
python gnn_ML/benchmark_gnn_regression.py --dataset ESOL --model GCN
python gnn_ML/benchmark_gnn_regression.py --dataset Lipo --model GAT
python gnn_ML/benchmark_gnn_regression.py --dataset FreeSolv --model GAT
python gnn_ML/benchmark_gnn_regression.py --dataset AQSOL --model GAT
```

### Tabular ML regression benchmarks

```bash
python gnn_ML/benchmark_ml_regression.py --dataset ESOL --model rf
python gnn_ML/benchmark_ml_regression.py --dataset ESOL --model hgb
python gnn_ML/benchmark_ml_regression.py --dataset ESOL --model xgb
```

Repeat for:

```text
Lipo
FreeSolv
AQSOL
```

### Tabular BO benchmarks

```bash
python gnn_ML/benchmark_ml_bo_ucb.py --dataset ESOL --model rf --acquisition ucb
python gnn_ML/benchmark_ml_bo_ucb.py --dataset ESOL --model rf --acquisition greedy
python gnn_ML/benchmark_ml_bo_ucb.py --dataset ESOL --model rf --acquisition random
```

### GNN BO benchmarks

```bash
python gnn_ML/benchmark_gnn_bo_ucb.py --dataset ESOL --model gat --acquisition ucb
python gnn_ML/benchmark_gnn_bo_ucb.py --dataset ESOL --model gat --acquisition greedy
python gnn_ML/benchmark_gnn_bo_ucb.py --dataset ESOL --model gat --acquisition random
```

---

## GNN implementation direction so far

The current GNN implementation is intentionally simple:

- two message-passing layers,
- global mean pooling,
- MLP regression head,
- GCN or GAT,
- early stopping,
- target normalization,
- ensemble uncertainty by training multiple models with different seeds.

This is a good first version because it is understandable and directly comparable to baselines.

### Possible improvements next time

Priority improvements:

1. **Use real QE labels**
   - Most important improvement.
   - No model architecture change matters until the target is scientifically meaningful.

2. **Add stronger molecular featurization**
   - More RDKit descriptors.
   - Morgan fingerprints for tabular baselines.
   - PFAS-specific functional group flags.
   - Adsorbent/catalyst material descriptors if available.

3. **Use scaffold or chemistry-aware splits**
   - Random splits can overestimate performance if similar molecules appear in train and test.
   - Scaffold splits better test generalization to new chemical families.

4. **Compare GIN/GINE**
   - Mole-BERT-style literature often uses GIN-style backbones.
   - GINE can incorporate edge features more explicitly than the current basic GCN/GAT.

5. **Use edge attributes more directly**
   - Bond type, aromaticity, conjugation, ring membership.
   - Current GCN/GAT versions mostly depend on PyG’s basic graph representation.

6. **Add calibration checks**
   - Ensemble `sigma` is useful only if uncertainty correlates with error.
   - Plot error vs. sigma or bin predictions by uncertainty.

7. **Add BO diversity constraints**
   - UCB can still select near-duplicates.
   - Penalize high Tanimoto similarity to already selected candidates.
   - Select top candidates per cluster.

8. **Try multi-fidelity later**
   - Use cheaper proxy labels or simpler simulations as low-fidelity signals.
   - Only worth doing after single-fidelity pipeline is stable.

---

## Recommended environment strategy for the ML portion

Use a hybrid install:

- Put stable conda-friendly scientific packages in `environment.yml`.
- Put PyTorch, PyTorch Geometric, and optional GPU-sensitive packages in `requirements.txt`.

This avoids common Conda dependency conflicts with PyTorch Geometric and CUDA-specific wheels.

### Install steps

```bash
conda env create -f environment.yml
conda activate pfas-ml-bo
pip install -r requirements.txt
```

Then verify:

```bash
python - <<'PY'
import torch
import torch_geometric
import pandas
import sklearn
import xgboost
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("pyg:", torch_geometric.__version__)
print("ok")
PY
```
