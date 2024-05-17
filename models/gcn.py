import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, Embedding, Linear, Module, ModuleList, ReLU,
                      Sequential)
from torch_geometric.nn import GCNConv, GINConv, GINEConv, SAGEConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree
from torch_scatter import scatter

from models.layers import MLP, AtomEncoder, MLPReadout


class GCN(Module):
    def __init__(self, hidden_channels, out_channels, in_channels, num_layers, dropout=0, residual=False, batch_norm=False, graph_level=False, pool_reduction="sum"):
        super().__init__()
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction
        self.dropout = dropout
        self.feature_encoder = AtomEncoder(hidden_channels)

        self.residual = residual

        self.layers = ModuleList()
        self.batch_norms = ModuleList()
        for layer_ind in range(num_layers):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            if batch_norm:
                self.batch_norms.append(BatchNorm(in_channels=hidden_channels))
        self.out = MLPReadout(hidden_channels, output_dim=out_channels)

    def forward(self, data):

        edge_index = data.edge_index
        x = self.feature_encoder(data)
        for layer_ind, layer in enumerate(self.layers):

            if self.residual:
                x_c = x

            x = layer(x, edge_index)

            if self.residual:
                x += x_c
            if self.batch_norms:
                x = self.batch_norms[layer_ind](x)

            if layer_ind != len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.relu()

        if self.graph_level:
            # Pooling
            x = scatter(x, data.batch, dim=0, reduce=self.pool_reduction)
        x = self.out(x)
        return x


class GCNSubstructure(Module):
    def __init__(self, hidden_channels, hidden_channels_substructure, out_channels, in_channels, in_channels_substructure, num_layers, dropout=0.5, residual=False, batch_norm=True, substructure_batch_norm=True, graph_level=False, pool_reduction="mean", substructure_messages=True):
        super().__init__()
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction
        self.dropout = dropout
        self.feature_encoder = Linear(in_channels, hidden_channels)
        self.feature_encoder_substructure = Linear(
            in_channels_substructure, hidden_channels_substructure)
        self.in_channels_substructure = in_channels_substructure
        self.in_channels = in_channels
        self.residual = residual
        self.substructure_messages = substructure_messages

        self.layers = ModuleList()
        self.to_substructure = ModuleList()
        self.from_substructure = ModuleList()
        self.batch_norms = ModuleList()
        self.batch_norms_substructure = ModuleList()

        for layer_ind in range(num_layers):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            self.to_substructure.append(SAGEConv(
                (hidden_channels, hidden_channels_substructure), hidden_channels_substructure))
            self.from_substructure.append(
                SAGEConv((hidden_channels_substructure, hidden_channels), hidden_channels))
            if batch_norm:
                self.batch_norms.append(BatchNorm(in_channels=hidden_channels))
            if substructure_batch_norm:
                self.batch_norms_substructure.append(
                    BatchNorm(in_channels=hidden_channels_substructure))
        self.out = MLPReadout(hidden_channels + hidden_channels_substructure,
                              output_dim=out_channels) if graph_level else MLPReadout(hidden_channels, output_dim=out_channels)

    def forward(self, data):
        x = data.x
        batch_size = max(data.x_batch) + 1
        x_substructure = data.fragments
        edge_index = data.edge_index
        substructure_edge_index = data.fragments_edge_index

        x = self.feature_encoder(x)
        x_substructure = self.feature_encoder_substructure(x_substructure)
        for layer_ind, (layer, to_sub, from_sub) in enumerate(zip(self.layers, self.to_substructure, self.from_substructure)):

            if self.residual:
                x_c = x
                x_substructure_c = x_substructure

            x = layer(x, edge_index)
            if self.batch_norms:
                x = self.batch_norms[layer_ind](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.substructure_messages:
                x_substructure = to_sub(
                    (x, x_substructure), substructure_edge_index)
                if self.batch_norms_substructure:
                    x_substructure = self.batch_norms_substructure[layer_ind](
                        x_substructure)
                x_substructure = F.relu(x_substructure)
                x_substructure = F.dropout(
                    x_substructure, self.dropout, training=self.training)
                x_substructure = x_substructure + x_substructure_c

                x = from_sub((x_substructure, x),
                             substructure_edge_index[[1, 0]])

            if self.residual:
                x += x_c

        if self.graph_level:
            # Pooling
            x = scatter(x, data.x_batch, dim=0, reduce=self.pool_reduction)
            x_substructure = scatter(x_substructure, data.fragments_batch,
                                     dim=0, reduce=self.pool_reduction, dim_size=batch_size)
            x = self.out(torch.concat([x, x_substructure], dim=1))
        else:
            x = self.out(x)
        return x


class VerySimpleGCN(Module):
    def __init__(self, hidden_channels, out_channels, in_channels, num_layers, dropout=0.5, graph_level=False, pool_reduction="sum"):
        super().__init__()
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction
        self.dropout = dropout

        self.layers = ModuleList()
        for layer_ind in range(num_layers):
            gcn_in = in_channels if layer_ind == 0 else hidden_channels
            gcn_out = out_channels if layer_ind == num_layers - 1 else hidden_channels
            self.layers.append(GCNConv(gcn_in, gcn_out))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for layer_ind, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if layer_ind != len(self.layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.graph_level:
            # Pooling
            x = scatter(x, data.batch, dim=0, reduce=self.pool_reduction)

        return x
