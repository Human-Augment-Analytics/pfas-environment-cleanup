import argparse
import pandas as pd
import torch

from rdkit import Chem
from torch_geometric.data import Batch

from model import GNN_graphpred
from loader import mol_to_graph_data_obj_simple


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Same architecture as your existing regression training defaults.
    model = GNN_graphpred(
        num_layer=5,
        emb_dim=300,
        num_tasks=1,
        JK="last",
        drop_ratio=0.5,
        graph_pooling="mean",
        gnn_type="gin",
    )

    # Load the FULL fine-tuned checkpoint, not the pretrained encoder.
    state_dict = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    df = pd.read_csv(args.input)

    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    predictions = [float("nan")] * len(df)
    valid_graphs = []
    valid_indices = []

    for i, smiles in enumerate(df["smiles"]):
        if not isinstance(smiles, str) or not smiles.strip():
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        try:
            graph = mol_to_graph_data_obj_simple(mol)
        except (ValueError, IndexError):
            # Unsupported atom/bond features.
            continue

        if graph is None:
            continue

        # The original model has three chirality embeddings.
        if graph.x[:, 1].max().item() >= 3:
            continue

        valid_graphs.append(graph)
        valid_indices.append(i)

    with torch.no_grad():
        for start in range(0, len(valid_graphs), args.batch_size):
            graphs = valid_graphs[start:start + args.batch_size]
            indices = valid_indices[start:start + args.batch_size]

            batch = Batch.from_data_list(graphs).to(device)

            pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            for index, value in zip(indices, pred.view(-1).tolist()):
                predictions[index] = value

    df["predicted_value"] = predictions
    df.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")
    print(f"Predicted {len(valid_indices)} of {len(df)} molecules.")


if __name__ == "__main__":
    main()
