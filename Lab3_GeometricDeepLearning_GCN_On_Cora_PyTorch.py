# -*- coding: utf-8 -*-
"""
Aditya Intwala

This is a script to demonstrate the implementation of Graph Convolution Layer implementation
in pyTorch.

Refer Lab3_GeometricDeepLearning_GCN_On_Cora_PyTorch.ipynb for more information on this.

The script was provided to the students of ACM Summer School June 2021 for Geometric Deep Learning Session 
in Shape Modeling School.

"""
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='add')  # "Add" aggregation
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.linear(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class GCNNetwork(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNNetwork, self).__init__()
        self.conv1 = GCNLayer(dataset.num_node_features, 16) #use custom implemented GCN Layer
        self.conv2 = GCNLayer(16, dataset.num_classes) #use custom implemented GCN Layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCNNetworkPytorch(torch.nn.Module):
    def __init__(self,dataset):
        super(GCNNetworkPytorch, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True, normalize=True) #use pre implemented GCNConv Layer
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,  normalize=True) #use pre implemented GCNConv Layer
        

    def forward(self,data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = { 'node_size': 1, 'width': 0.2 }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, **options)
    plt.show()


def train(model, optimizer, data):
    train_accuracies, test_accuracies = list(), list()
    for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            train_acc = test(model, data)
            test_acc = test(model, data, train=False)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, loss, train_acc, test_acc))

    return train_accuracies, test_accuracies


def test(model, data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]

    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))


print('========================== Downloading / Loading Cora Dataset ==========================')
dataset = Planetoid(root='./Data/Dataset', name='Cora')

#plot the dataset
plot_dataset(dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('========================== GCNNetwork with our custom GCNLayer ==========================')
customGCN = GCNNetwork(dataset).to(device)
#assign data to device for training
data = dataset[0].to(device)
#create optimizer for training
optimizer = torch.optim.Adam(customGCN.parameters(), lr=0.01, weight_decay=5e-4)
print('============= Training & Evaluation =============')
customGCN_train_accuracies, customGCN_test_accuracies = train(customGCN, optimizer, data)

print('========================== GCNNetworkPytorch with pre implemented GraphConv Layer ==========================')
pytorchGCN = GCNNetworkPytorch(dataset).to(device)
#assign data to device for training
data = dataset[0].to(device)
#create optimizer for training
optimizer = torch.optim.Adam([ dict(params=pytorchGCN.conv1.parameters(), weight_decay=5e-4), dict(params=pytorchGCN.conv2.parameters(), weight_decay=0)], lr=0.01)  # Only perform weight-decay on first convolution.
print('============= Training & Evaluation =============')
pytorchGCN_train_accuracies, pytorchGCN_test_accuracies = train(pytorchGCN, optimizer, data)

plt.plot(customGCN_train_accuracies, label="customGCN Train accuracy")
plt.plot(customGCN_test_accuracies, label="customGCN Validation accuracy")
plt.plot(pytorchGCN_train_accuracies, label="pytorchGCN Train accuracy")
plt.plot(pytorchGCN_test_accuracies, label="pytorchGCN Validation accuracy")
plt.xlabel("# Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()