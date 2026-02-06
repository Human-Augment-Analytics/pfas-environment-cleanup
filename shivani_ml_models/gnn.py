import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    data
    data.num_graphs
    x = scatter(data.x, data.batch, dim=0, reduce='mean')
    x.size()

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(200):
    for data in loader: 
        data = data.to(device)
        optimizer.zero_grad()
        out_node_level = model(data) # Shape: [num_nodes_in_batch, num_classes]
        out_graph_level = global_mean_pool(out_node_level, data.batch) # Shape: [batch_size, num_classes]
        loss = F.nll_loss(out_graph_level, data.y)
        loss.backward()
        optimizer.step()

    model.eval()

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]


test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0

with torch.no_grad(): 
    for data in test_loader:
        data = data.to(device)
        out_node_level = model(data)
        out_graph_level = global_mean_pool(out_node_level, data.batch)
        pred = out_graph_level.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0) 

acc = correct / total
print(f'Accuracy: {acc:.4f}')