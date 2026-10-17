# Mole-BERT: Fine-Tuning and Inference

This directory contains reusable training and inference pipelines for molecular property prediction using Mole-BERT, a graph neural network pretrained on molecular structures.

The implementation uses the original Mole-BERT architecture and PyTorch Geometric molecular graph representations.

## 1. Directory Structure

```text
molebert/
├── README.md
├── training/
│   ├── finetune_regression.py
│   ├── model.py
│   ├── loader.py
│   └── splitters.py
└── inference/
    ├── predict.py
    ├── model.py
    └── loader.py
```

The `training/` directory handles fine-tuning on labeled molecular datasets. The `inference/` directory loads fine-tuned checkpoints and predicts properties for new SMILES strings.

## 2. Environment Setup

The pipeline requires Python, PyTorch, PyTorch Geometric, RDKit, NumPy, Pandas, tqdm, and tensorboardX, together with the dependencies required by the original Mole-BERT implementation.

On Georgia Tech PACE ICE, activate the existing environment:

```bash
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate molebert

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Verify the environment:

```bash
python -c "import torch, rdkit, torch_geometric; print('Ready')"
```

For a new environment, install dependency versions compatible with the original Mole-BERT implementation. PyTorch Geometric extensions must match the installed PyTorch and CUDA versions.

## 3. Fine-Tuning

### Supported datasets

The current regression training script supports:

| Dataset | Prediction target |
|---|---|
| ESOL | Aqueous solubility |
| FreeSolv | Hydration free energy |
| Lipophilicity | Lipophilicity |

Datasets must be available in the original Mole-BERT dataset structure, including the expected raw data files.

### Pretrained checkpoint

Fine-tuning begins from a pretrained Mole-BERT encoder checkpoint, for example:

```text
checkpoints/pretrain_epoch10.pth
```

This checkpoint contains the pretrained GNN encoder. The regression prediction head is initialized and trained during fine-tuning.

### Run fine-tuning

From the original Mole-BERT repository root, run:

```bash
PYTHONPATH=molebert/training python molebert/training/finetune_regression.py \
  --dataset esol \
  --input_model_file checkpoints/pretrain_epoch10.pth \
  --epochs 100 \
  --batch_size 32 \
  --split scaffold \
  --save_model checkpoints/esol_pre10_best.pth
```

Replace `esol` with `freesolv` or `lipophilicity` to train on another supported dataset.

### Important training parameters

| Parameter | Description |
|---|---|
| `--dataset` | Regression dataset |
| `--input_model_file` | Pretrained GNN checkpoint |
| `--epochs` | Number of training epochs |
| `--batch_size` | Training batch size |
| `--lr` | Learning rate |
| `--split` | Dataset splitting strategy |
| `--seed` | Dataset split seed |
| `--runseed` | Training initialization seed |
| `--save_model` | Output path for best checkpoint |

The model is trained using mean squared error (MSE) loss.

Validation RMSE is used to select the best checkpoint. The current script also reports test RMSE after each epoch; test performance should not be used for model selection.

### Saved checkpoints

When `--save_model` is specified, the training script saves the full fine-tuned model state dictionary whenever validation RMSE improves.

For example:

```text
checkpoints/esol_pre10_best.pth
```

This includes both the Mole-BERT encoder and the trained regression prediction head.

## 4. Inference

The inference pipeline predicts molecular properties from SMILES strings using a previously fine-tuned checkpoint.

### Prepare input

Create a CSV file:

```csv
smiles
CCO
CCCO
CC(=O)O
```

Save it as `test_input.csv` in the `inference/` directory.

### Run inference

From the original Mole-BERT repository root:

```bash
cd molebert/inference

python predict.py \
  --checkpoint ../../checkpoints/esol_pre10_best.pth \
  --input test_input.csv \
  --output predictions.csv
```

The script loads the fine-tuned model, converts each valid SMILES string into a molecular graph, and performs batched predictions.

### Output

The generated CSV contains the original SMILES and a prediction column:

```csv
smiles,predicted_value
CCO,...
CCCO,...
CC(=O)O,...
```

Predicted values depend on the checkpoint used.

- An ESOL checkpoint predicts aqueous solubility.
- A FreeSolv checkpoint predicts hydration free energy.
- A Lipophilicity checkpoint predicts lipophilicity.

The script retains rows containing invalid or unsupported SMILES and assigns missing prediction values.

### Switch between models

For FreeSolv:

```bash
python predict.py \
  --checkpoint ../../checkpoints/freesolv_pre10_best.pth \
  --input test_input.csv \
  --output freesolv_predictions.csv
```

For Lipophilicity:

```bash
python predict.py \
  --checkpoint ../../checkpoints/lipophilicity_pre10_best.pth \
  --input test_input.csv \
  --output lipophilicity_predictions.csv
```

The current inference script assumes the default architecture: five GIN layers, 300-dimensional embeddings, last-layer representations, mean pooling, and one regression output. Fine-tuned checkpoints must use matching architecture settings.

## 5. Molecular Graph Representation

Molecules are represented as graphs using RDKit and PyTorch Geometric.

Each atom has two encoded features:

- Atomic number
- Chirality tag

Each bond has two encoded features:

- Bond type
- Bond direction

Molecular bonds are represented as bidirectional edges.

The graph is passed through the Mole-BERT GNN encoder, followed by graph pooling and a regression prediction head.

Training and inference use the same original graph feature encoding.

## 6. Using Mole-BERT for Adsorption Energy

The current implementation has been used for molecular property regression. It has not yet been fine-tuned on adsorption-energy data.

For future adsorption-energy prediction, a labeled dataset could use:

```csv
smiles,adsorption_energy_ev
```

Training on such a dataset requires extending the existing dataset loader and training script.

For a fixed-surface study, adsorption-energy labels should correspond to the same surface and a consistent energy definition and computational protocol.

The current molecular-only model does not explicitly represent surface geometry or adsorption configurations.

## 7. Notes

- Run GPU training on PACE compute nodes through Slurm, not on login nodes.
- Pretrained encoder checkpoints and fine-tuned prediction checkpoints are different.
- The inference script requires a full fine-tuned checkpoint.
- Do not commit large model checkpoints or processed datasets to Git unless the group explicitly wants to track them.
- Preserve the original Mole-BERT license and attribution when redistributing code.
- Model predictions should be evaluated on appropriate held-out data before scientific use.

## 8. Reference

Mole-BERT: Rethinking Pre-training Graph Neural Networks for Molecules, ICLR 2023.

Original implementation: https://github.com/junxia97/Mole-BERT
