import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm

from loader import MoleculeDataset
from model import GNN_graphpred
from splitters import scaffold_split, random_split, random_scaffold_split


criterion = nn.MSELoss(reduction="none")


def train(args, epoch, model, device, loader, optimizer):
    model.train()

    epoch_iter = tqdm(loader, desc=f"Epoch {epoch}")
    total_squared_error = 0.0
    total_valid_targets = 0

    for batch in epoch_iter:
        batch = batch.to(device)

        pred, _ = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        y = batch.y.view(pred.shape).to(torch.float64)
        pred = pred.to(torch.float64)

        # Ignore missing labels represented by NaN or infinity.
        is_valid = torch.isfinite(y)

        if torch.sum(is_valid).item() == 0:
            continue

        loss_mat = criterion(pred, y)
        loss_mat = torch.where(
            is_valid,
            loss_mat,
            torch.zeros_like(loss_mat),
        )

        loss = torch.sum(loss_mat) / torch.sum(is_valid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid_count = torch.sum(is_valid).item()
        total_squared_error += torch.sum(loss_mat).item()
        total_valid_targets += valid_count

        running_rmse = np.sqrt(
            total_squared_error / total_valid_targets
        )

        epoch_iter.set_description(
            f"Epoch: {epoch} loss: {loss.item():.4f} "
            f"rmse: {running_rmse:.4f}"
        )

    if total_valid_targets == 0:
        raise RuntimeError(
            "No valid regression targets were found in the training set."
        )

    return np.sqrt(total_squared_error / total_valid_targets)


def evaluate(model, device, loader):
    model.eval()

    squared_error_sum = 0.0
    valid_target_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            batch = batch.to(device)

            pred, _ = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            y = batch.y.view(pred.shape).to(torch.float64)
            pred = pred.to(torch.float64)

            is_valid = torch.isfinite(y)

            if torch.sum(is_valid).item() == 0:
                continue

            squared_errors = (pred - y) ** 2

            squared_error_sum += torch.sum(
                squared_errors[is_valid]
            ).item()

            valid_target_count += torch.sum(is_valid).item()

    if valid_target_count == 0:
        raise RuntimeError(
            "No valid regression targets were found during evaluation."
        )

    rmse = np.sqrt(squared_error_sum / valid_target_count)
    return rmse


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Mole-BERT on molecular regression tasks."
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for prediction layers.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="Number of GNN message-passing layers.",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=300,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--dropout_ratio",
        type=float,
        default=0.5,
        help="Dropout ratio.",
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        choices=["sum", "mean", "max", "set2set", "attention"],
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        choices=["last", "sum", "max", "concat"],
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gin",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["esol", "freesolv", "lipophilicity"],
    )
    parser.add_argument(
        "--input_model_file",
        type=str,
        default="None",
        help="Pretrained GNN checkpoint.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="TensorBoard run name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset split seed.",
    )
    parser.add_argument(
        "--runseed",
        type=int,
        default=0,
        help="Training and initialization seed.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="scaffold",
        choices=["scaffold", "random", "random_scaffold"],
    )
    parser.add_argument(
        "--eval_train",
        type=int,
        default=1,
        help="Whether to calculate training RMSE each epoch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="",
        help="Optional path for saving the best model.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Dataset:", args.dataset)

    # ESOL, FreeSolv, and Lipophilicity each have one target.
    num_tasks = 1

    dataset_root = os.path.join("dataset", args.dataset)

    dataset = MoleculeDataset(
        dataset_root,
        dataset=args.dataset,
    )

    print(dataset)

    smiles_path = os.path.join(
        dataset_root,
        "processed",
        "smiles.csv",
    )

    smiles_list = pd.read_csv(
        smiles_path,
        header=None,
    )[0].tolist()

    if args.split == "scaffold":
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
        )
        print("Split: scaffold")

    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        print("Split: random")

    elif args.split == "random_scaffold":
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        print("Split: random scaffold")

    else:
        raise ValueError(f"Unsupported split: {args.split}")

    print("Training molecules:", len(train_dataset))
    print("Validation molecules:", len(valid_dataset))
    print("Test molecules:", len(test_dataset))
    print("Example:", train_dataset[0])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = GNN_graphpred(
        args.num_layer,
        args.emb_dim,
        num_tasks,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        graph_pooling=args.graph_pooling,
        gnn_type=args.gnn_type,
    )

    if args.input_model_file != "None":
        print(
            "Loading pretrained model:",
            args.input_model_file,
        )
        model.from_pretrained(args.input_model_file)
    else:
        print("Training from scratch.")

    model.to(device)

    model_param_group = [
        {
            "params": model.gnn.parameters(),
        }
    ]

    if args.graph_pooling == "attention":
        model_param_group.append(
            {
                "params": model.pool.parameters(),
                "lr": args.lr * args.lr_scale,
            }
        )

    model_param_group.append(
        {
            "params": model.graph_pred_linear.parameters(),
            "lr": args.lr * args.lr_scale,
        }
    )

    optimizer = optim.Adam(
        model_param_group,
        lr=args.lr,
        weight_decay=args.decay,
    )

    print(optimizer)

    train_rmse_list = []
    val_rmse_list = []
    test_rmse_list = []

    writer = None

    if args.filename:
        tensorboard_path = os.path.join(
            "runs",
            "finetune_reg_runseed" + str(args.runseed),
            args.filename,
        )

        if os.path.exists(tensorboard_path):
            shutil.rmtree(tensorboard_path)
            print("Removed existing TensorBoard directory.")

        writer = SummaryWriter(tensorboard_path)

    best_val_rmse = float("inf")
    best_test_rmse = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        print(f"==== Epoch {epoch}")

        training_step_rmse = train(
            args,
            epoch,
            model,
            device,
            train_loader,
            optimizer,
        )

        print("==== Evaluation")

        if args.eval_train:
            train_rmse = evaluate(
                model,
                device,
                train_loader,
            )
        else:
            train_rmse = training_step_rmse

        val_rmse = evaluate(
            model,
            device,
            val_loader,
        )

        test_rmse = evaluate(
            model,
            device,
            test_loader,
        )

        print(
            "train RMSE: %.6f val RMSE: %.6f test RMSE: %.6f"
            % (train_rmse, val_rmse, test_rmse)
        )

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)
        test_rmse_list.append(test_rmse)

        if writer is not None:
            writer.add_scalar(
                "data/train_rmse",
                train_rmse,
                epoch,
            )
            writer.add_scalar(
                "data/val_rmse",
                val_rmse,
                epoch,
            )
            writer.add_scalar(
                "data/test_rmse",
                test_rmse,
                epoch,
            )

        # Lower validation RMSE is better.
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse = test_rmse
            best_epoch = epoch

            if args.save_model:
                save_directory = os.path.dirname(args.save_model)

                if save_directory:
                    os.makedirs(
                        save_directory,
                        exist_ok=True,
                    )

                torch.save(
                    model.state_dict(),
                    args.save_model,
                )

                print(
                    "Saved best model to:",
                    args.save_model,
                )

    print("Best epoch:", best_epoch)
    print("Best validation RMSE:", best_val_rmse)
    print("Test RMSE at best validation epoch:", best_test_rmse)

    checkpoint_name = os.path.basename(
        args.input_model_file
    ).replace(".pth", "")

    if not checkpoint_name or checkpoint_name == "None":
        checkpoint_name = "scratch"

    experiment_path = os.path.join(
        os.getcwd(),
        f"{checkpoint_name}_regression_results",
        args.dataset,
    )

    os.makedirs(
        experiment_path,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        {
            "epoch": list(range(1, args.epochs + 1)),
            "train_rmse": train_rmse_list,
            "valid_rmse": val_rmse_list,
            "test_rmse": test_rmse_list,
        }
    )

    results_df.to_csv(
        os.path.join(
            experiment_path,
            f"seed{args.runseed}.csv",
        ),
        index=False,
    )

    log_line = (
        f"Dataset:{args.dataset}, "
        f"Seed:{args.runseed}, "
        f"Best Epoch:{best_epoch}, "
        f"Best Validation RMSE:{best_val_rmse:.6f}, "
        f"Test RMSE:{best_test_rmse:.6f}"
    )

    with open(
        os.path.join(
            experiment_path,
            f"{args.dataset}_log.csv",
        ),
        "a+",
    ) as log_file:
        log_file.write(log_line + "\n")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
