import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool



dataset = MoleculeNet(root ="data", name = "ESOL")
print(dataset)

print("Number of molecules: ", len(dataset))
print("Number of node features: ", dataset.num_node_features)

data = dataset[0]
print("item structure: ", data)

torch.manual_seed(42)

train_idx, temp_idx = train_test_split(
    list(range(len(dataset))),
    test_size = 0.2,
    random_state = 42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size = 0.5,
    random_state = 42
)


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        # Message passing
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x  = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        
        return x.squeeze()
    

from torch_geometric.nn import GATConv

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_dim, heads = 4)
        self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads = 1)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x.squeeze()  

# Model Orchestration
def get_model(model_name, in_channels, hidden_dim=64):
    if model_name == "GCN":
        return GCNModel(in_channels, hidden_dim)
    if model_name == "GAT":
        return GATModel(in_channels, hidden_dim)
    else:
        raise ValueError("Model Name NOT Found")

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.y.view(-1).float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

from sklearn.metrics import mean_absolute_error

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        preds.append(out.cpu())
        targets.append(data.y.view(-1).cpu())
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    mae = mean_absolute_error(targets, preds)

    return mae

import copy

def run_experiment(model_name, device):
    
    train_loader = DataLoader(dataset[train_idx], batch_size = 32, shuffle = True)
    val_loader = DataLoader(dataset[val_idx], batch_size = 32)
    test_loader = DataLoader(dataset[test_idx], batch_size = 32)

    model = get_model(model_name, dataset.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
    criterion = nn.MSELoss()


    best_val = float('inf')
    best_model_state = None
    patience = 50
    counter = 0

    print(f"Results for ")

    for epoch in range(300):
        loss = train(model, train_loader, optimizer, criterion)
        val_mae = evaluate(model, val_loader, device)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:0.4f}  |  Validation MAE: {val_mae:0.4f}")
        
        if val_mae < best_val:
            best_val = val_mae
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping triggered: {patience} iterations with no improvement")
            break

    model.load_state_dict(best_model_state)

    test_mae = evaluate(model, test_loader, device)
    return best_val, test_mae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = ["GCN", "GAT"]


import numpy as np

seeds = [0, 1, 2, 3, 4, 7, 42, 1234, 777]

for model_name in models:
    tst_scores = []
    val_scores = []
    print("=" * 45)
    print(f"Model: {model_name}")

    for seed in seeds:
        torch.manual_seed(seed)
        print(f"Seed: {seed}")

        val_mae, test_mae = run_experiment(model_name, device)

        print(f"Best Validation MAE: {val_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print("-" * 25)
        val_scores.append(val_mae)
        tst_scores.append(test_mae)

    print(f"Best Val MAE Results: {val_scores}")
    print(f"Best Test MAE results: {tst_scores} ")
    print(f"Mean Val MAE: {np.mean(val_scores)} | Std: {np.std(val_scores)}")
    print(f"Mean Test MAE: {np.mean(tst_scores)} | Std: {np.std(tst_scores)}")

# import matplotlib.pyplot as plt

# plt.scatter(test_targets, test_preds)
# plt.xlabel("True")
# plt.ylabel("Predicted")
# plt.plot([min(test_targets), max(test_targets)],
#          [min(test_targets), max(test_targets)],
#          color = "red")
# plt.show()
