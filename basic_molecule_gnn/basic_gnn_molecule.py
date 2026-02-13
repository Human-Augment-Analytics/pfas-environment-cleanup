import torch
from torch_geometric.datasets import MoleculeNet

dataset = MoleculeNet(root ="data", name = "ESOL")
print(dataset)

print("Number of molecules: ", len(dataset))
print("Number of node features: ", dataset.num_node_features)

data = dataset[0]
print("item structure: ", data)

from torch.utils.data import random_split

torch.manual_seed(42)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

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


train_loader = DataLoader(dataset[train_idx], batch_size = 32, shuffle = True)
val_loader = DataLoader(dataset[val_idx], batch_size = 32)
test_loader = DataLoader(dataset[test_idx], batch = 32)



import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # Message passing
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x  = self.conv2(x, edge_index)

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
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu()
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x.squeeze()


from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCNModel(dataset.num_node_features).to(device)
# model = GATModel(dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# def test(loader):
#     model.eval()
#     total_error = 0

#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             out = model(data.x, data.edge_index, data.batch)
#             total_error += torch.abs(out - data.y).sum().item()

#     return total_error / len(loader.dataset)

from sklearn.metrics import mean_absolute_error

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds = []
    targets = []

    for data in loader:
        out = model(data)
        preds.append(out.cpu())
        targets.append(data.y.cpu())
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    mae = mean_absolute_error(targets, preds)
    return mae, preds, targets

import copy

best_val_mae = float('inf')
best_model_state = None
patience = 30
counter = 0

for epoch in range(300):
    loss = train(model, train_loader, optimizer, criterion)
    val_mae, _, _ = evaluate(model, val_loader)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:0.4f}  |  Validation MAE: {val_mae:0.4f}")
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_model_state = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping triggers")
        break

model.load_state_dict(best_model_state)

test_mae, test_preds, test_targets = evaluate(model, test_loader)

print(f"Final Test Mae: {test_mae:0.4f}")

import matplotlib.pyplot as plt

plt.scatter(test_targets, test_preds)
plt.xlabel("True")
plt.ylabe("Predicted")
plt.plot([min(test_targets), max(test_targets)],
         [min(test_targets), max(test_targets)],
         color = "red")
plt.show()
