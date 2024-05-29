import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, Embedding, Linear, Module, ModuleList, ReLU,
                      Sequential)
from torch_geometric.nn import GCNConv, GINConv, GINEConv, SAGEConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree
from torch_scatter import scatter

from models.layers import (MLP, AtomEncoder, BondEncoder, FragEncoder,
                           InterMessage)


class FragGNN(torch.nn.Module):
    """Adapted from https://github.com/rusty1s/himp-gnn/blob/master/model.py, deprecated.
    Use FragGNNSmall instead"""

    def __init__(self, in_channels, in_channels_substructure, in_channels_edge,
                 hidden_channels, out_channels, num_layers, dropout=0.0,
                 linear_atom_encoder=False, encoding_size_scaling=False, rbf=0,
                 atom_feature_params={}, edge_feature_params={},
                 degree_scaling=False, additional_atom_features=[],
                 inter_message_passing=True, higher_message_passing=False,
                 no_frag_info=False,
                 low_high_edges=False, fragment_specific=False,
                 reduction="mean", frag_reduction=None, concat=False, graph_rep=False,
                 learned_edge_rep=False, higher_level_edge_features=False,
                 graph_rep_node=False, inter_message_params={}, hidden_channels_substructure=None,
                 num_layers_out=2):
        super(FragGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels_substructure = hidden_channels_substructure if hidden_channels_substructure else hidden_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing
        self.higher_message_passing = higher_message_passing
        self.low_high_edges = low_high_edges
        self.encoding_size_scaling = encoding_size_scaling
        self.rbf = rbf
        self.degree_scaling = degree_scaling
        self.fragment_specific = fragment_specific
        self.reduction = reduction
        self.frag_reduction = frag_reduction if frag_reduction else reduction
        self.concat = concat
        self.graph_rep = graph_rep
        self.graph_rep_node = graph_rep_node
        self.learned_edge_rep = learned_edge_rep
        self.higher_level_edge_features = higher_level_edge_features
        self.out_channels = out_channels
        self.no_frag_info = no_frag_info

        # self.atom_encoder = Linear(in_channels, hidden_channels)
        self.atom_encoder = Linear(in_channels, hidden_channels) if linear_atom_encoder else AtomEncoder(
            hidden_channels, degree_scaling, additional_atom_features, **atom_feature_params)

        self.clique_encoder = FragEncoder(
            in_channels_substructure, self.hidden_channels_substructure, encoding_size_scaling, rbf)

        if not self.learned_edge_rep:
            self.bond_encoders = ModuleList()
        else:
            self.bond_encoder = BondEncoder(
                hidden_channels, **edge_feature_params)
            self.atom2bond = ModuleList()
            self.bond_batch_norms = ModuleList()
            self.bond_convs = ModuleList()

        if self.graph_rep or self.graph_rep_node:
            hidden_channels_graph = hidden_channels
            self.graph_encoder = Embedding(1, hidden_channels_graph)

        if self.low_high_edges:
            self.bond_encoders_low_high = ModuleList()
        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()
        if self.graph_rep_node:
            self.atom2graph = ModuleList()
            self.graph2atom = ModuleList()
        if self.graph_rep or self.graph_rep_node:
            self.graph_batch_norms = ModuleList()
            self.graph_conv = ModuleList()

        for _ in range(num_layers):
            # self.bond_encoders.append(Linear(in_channels_edge, hidden_channels))
            if not self.learned_edge_rep:
                self.bond_encoders.append(BondEncoder(
                    hidden_channels, **edge_feature_params))
            if self.low_high_edges:
                self.bond_encoders_low_high.append(BondEncoder(
                    self.hidden_channels_substructure, **edge_feature_params))
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(
                GINEConv(nn, train_eps=True, edge_dim=hidden_channels))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))
            if self.graph_rep_node:
                self.atom2graph.append(InterMessage(
                    hidden_channels, hidden_channels_graph, **inter_message_params))
                self.graph2atom.append(
                    Linear(hidden_channels_graph, hidden_channels))
            if self.graph_rep or self.graph_rep_node:
                self.graph_batch_norms.append(
                    BatchNorm1d(hidden_channels_graph))
                self.graph_conv.append(
                    Linear(hidden_channels_graph, hidden_channels_graph))
            if self.learned_edge_rep:
                self.atom2bond.append(InterMessage(
                    hidden_channels, hidden_channels, **inter_message_params))
                self.bond_batch_norms.append(BatchNorm1d(hidden_channels))
                self.bond_convs.append(
                    Linear(hidden_channels, hidden_channels))

        if self.inter_message_passing:
            self.frag_convs = ModuleList()
            self.frag_batch_norms = ModuleList()
            if self.graph_rep:
                self.fragment2graph = ModuleList()
                self.graph2fragment = ModuleList()
            if self.concat:
                self.concat_lins = ModuleList()

            for _ in range(num_layers):
                nn = Sequential(
                    Linear(self.hidden_channels_substructure,
                           2 * self.hidden_channels_substructure),
                    BatchNorm1d(2 * self.hidden_channels_substructure),
                    ReLU(),
                    Linear(2 * self.hidden_channels_substructure,
                           self.hidden_channels_substructure),
                )
                if self.higher_level_edge_features:
                    self.frag_convs.append(
                        GINEConv(nn, train_eps=True, edge_dim=self.hidden_channels))
                else:
                    self.frag_convs.append(GINConv(nn, train_eps=True))
                self.frag_batch_norms.append(
                    BatchNorm1d(self.hidden_channels_substructure))
                if self.concat:
                    # TODO: probably wrong
                    self.concat_lins.append(
                        Linear(2*hidden_channels, hidden_channels))
                if self.graph_rep:
                    self.fragment2graph.append(InterMessage(
                        self.hidden_channels_substructure, hidden_channels_graph, **inter_message_params))
                    self.graph2fragment.append(
                        Linear(hidden_channels_graph, self.hidden_channels_substructure))

            self.atom2frag = ModuleList()
            self.frag2atom = ModuleList()

            for _ in range(num_layers):
                if not self.fragment_specific:
                    self.atom2frag.append(
                        InterMessage(hidden_channels, self.hidden_channels_substructure, **inter_message_params))
                    self.frag2atom.append(
                        InterMessage(self.hidden_channels_substructure, hidden_channels, **inter_message_params))
                else:
                    self.atom2frag.append(
                        ModuleList([InterMessage(hidden_channels, self.hidden_channels_substructure, **inter_message_params) for i in range(3)]))
                    self.frag2atom.append(
                        ModuleList([InterMessage(self.hidden_channels_substructure, hidden_channels, **inter_message_params) for i in range(3)]))

        self.frag_out = MLP(
            self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        self.atom_out = MLP(
            self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        if self.learned_edge_rep:
            self.edge_out = MLP(
                self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        # self.mol_out = MLP(self.hidden_channels_mol, self.hidden_channels, num_layers = 2)
        self.out = MLP(self.hidden_channels, self.out_channels,
                       num_layers=num_layers_out, batch_norm=False, last_relu=False)

    def forward(self, data):

        batch_size = torch.max(data.batch) + 1
        if self.degree_scaling:
            degrees = degree(
                data.edge_index[0], dtype=torch.float, num_nodes=data.x.size(0))
            x = self.atom_encoder(data, degrees)
        else:
            x = self.atom_encoder(data)

        if self.encoding_size_scaling:
            x_frag = self.clique_encoder(data.fragment_types)
        else:
            x_frag = self.clique_encoder(data.fragments)

        if not self.inter_message_passing and not self.no_frag_info:
            # append frag information as node features
            row, col = data.fragments_edge_index
            x = x + scatter(x_frag[col], row, dim=0,
                            dim_size=x.size(0), reduce=self.reduction)

        if self.graph_rep:
            x_graph = torch.zeros(batch_size, dtype=torch.int, device=x.device)
            x_graph = self.graph_encoder(x_graph)

        if self.learned_edge_rep:
            x_edge = self.bond_encoder(data.edge_attr)

        for i in range(self.num_layers):
            if not self.learned_edge_rep:
                x_edge = self.bond_encoders[i](data.edge_attr)
            x = self.atom_convs[i](x, data.edge_index, x_edge)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.graph_rep_node:
                x_graph = x_graph + \
                    self.atom2graph[i](x, data.batch, dim_size=batch_size)

            if self.learned_edge_rep:
                row_edge, col_edge = data.edge_index
                x_edge = x_edge + self.atom2bond[i](torch.concat([x[row_edge], x[col_edge]], dim=0), torch.concat([torch.arange(
                    row_edge.size(0), dtype=torch.int64, device=row_edge.device) for _ in range(2)], dim=0), dim_size=row_edge.size(0))
                x_edge = self.bond_convs[i](x_edge)
                x_edge = self.bond_batch_norms[i](x_edge)
                x_edge = F.relu(x_edge)
                x_edge = F.dropout(x_edge, self.dropout,
                                   training=self.training)

            if self.inter_message_passing:
                row, col = data.fragments_edge_index

                if self.graph_rep:
                    # frag to graph
                    x_graph = x_graph + \
                        self.fragment2graph[i](
                            x_frag, data.fragments_batch, dim_size=batch_size)

                if self.fragment_specific:
                    subgraph_message = torch.zeros_like(x_frag)
                    edge_masks = [
                        data.fragment_types[data.fragments_edge_index[1], 0] == i for i in range(3)]
                    for edge_mask, message in zip(edge_masks, self.atom2frag[i]):
                        subgraph_message += message(x[row[edge_mask]],
                                                    col[edge_mask], dim_size=x_frag.size(0))
                    # subgraph_message = F.relu(subgraph_message)
                else:
                    subgraph_message = self.atom2frag[i](
                        x[row], col, dim_size=x_frag.size(0))
                x_frag = x_frag + subgraph_message

                if self.low_high_edges:
                    edges, frags = data.low_high_edge_index
                    edge_attr_new = self.bond_encoders_low_high[i](
                        data.edge_attr)
                    x_frag = x_frag + \
                        scatter(edge_attr_new[edges], frags, dim=0, dim_size=x_frag.size(
                            0), reduce="mean")

                if self.higher_message_passing:
                    if self.higher_level_edge_features:
                        number_of_higher_edges = data.higher_edge_index.size(1)
                        higher_edge_id, lower_edge_id = data.join_edge_index
                        lower_edge_info = scatter(
                            x_edge[lower_edge_id], higher_edge_id, reduce=self.reduction, dim=0, dim_size=number_of_higher_edges)
                        higher_edge_id, lower_node_id = data.join_node_index
                        lower_node_info = scatter(
                            x[lower_node_id], higher_edge_id, reduce=self.reduction, dim=0, dim_size=number_of_higher_edges)
                        info = lower_edge_info + lower_node_info
                        x_frag = self.frag_convs[i](
                            x_frag, data.higher_edge_index, info)
                    else:
                        x_frag = self.frag_convs[i](
                            x_frag, data.higher_edge_index)

                # if self.graph_rep:
                #     #graph to frag
                #     x_frag = x_frag + F.relu(self.graph2fragment[i](x_graph[data.fragments_batch]))

                x_frag = self.frag_batch_norms[i](x_frag)
                x_frag = F.relu(x_frag)
                x_frag = F.dropout(x_frag, self.dropout,
                                   training=self.training)

                if self.fragment_specific:
                    subgraph_message = torch.zeros_like(x)
                    edge_masks = [
                        data.fragment_types[data.fragments_edge_index[1], 0] == i for i in range(3)]
                    for edge_mask, message in zip(edge_masks, self.frag2atom[i]):
                        subgraph_message += message(
                            x_frag[col[edge_mask]], row[edge_mask], dim_size=x.size(0))
                else:
                    subgraph_message = self.frag2atom[i](
                        x_frag[col], row, dim_size=x.size(0))
                if self.concat:
                    x = self.concat_lins[i](torch.concat(
                        [x, subgraph_message], dim=-1))
                else:
                    x = x + subgraph_message

            if self.graph_rep or self.graph_rep_node:
                x_graph = F.relu(self.graph_conv[i](x_graph))
                x_graph = self.graph_batch_norms[i](x_graph)
                if self.graph_rep:
                    # graph to frag
                    x_frag = x_frag + \
                        F.relu(self.graph2fragment[i](
                            x_graph[data.fragments_batch]))
                if self.graph_rep_node:
                    # graph to node
                    x = x + F.relu(self.graph2atom[i](x_graph[data.batch]))

        x = scatter(x, data.batch, dim=0, reduce=self.reduction)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_out(x)

        if self.inter_message_passing:
            x_frag = scatter(x_frag, data.fragments_batch, dim=0, dim_size=x.size(0),
                             reduce=self.frag_reduction)
            x_frag = F.dropout(x_frag, self.dropout,
                               training=self.training)
            x_frag = self.frag_out(x_frag)
            x = x + x_frag

        if self.learned_edge_rep:
            edge_batch = data.batch[data.edge_index[0]]
            x_edge = scatter(x_edge, edge_batch, dim=0,
                             dim_size=batch_size, reduce=self.reduction)
            x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            x_edge = self.edge_out(x_edge)
            x = x + x_edge

        if self.graph_rep:
            x = x + self.graph_out(x_graph)

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out(x)
        return x


class FragGNNSmall(torch.nn.Module):
    """Based on HIMP https://github.com/rusty1s/himp-gnn/blob/master/model.py"""

    def __init__(self, in_channels, in_channels_substructure, in_channels_edge,
                 hidden_channels, out_channels, num_layers, dropout=0.0, ordinal_encoding=False,
                 atom_feature_params={}, edge_feature_params={},
                 inter_message_passing=True, higher_message_passing=False, no_frag_info=False,
                 reduction="mean", frag_reduction=None,
                 learned_edge_rep=False, inter_message_params={}, hidden_channels_substructure=None,
                 num_layers_out=2):
        """
        Initialize the FragGNNSmall model.

        Args:
            in_channels (int): Number of input channels for atom features.
            in_channels_substructure (int): Number of input channels for substructure features.
            in_channels_edge (int): Number of input channels for edge features.
            hidden_channels (int): Number of hidden channels for atom features.
            out_channels (int): Number of output channels.
            num_layers (int): Number of GNN layers.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            ordinal_encoding (bool, optional): Whether to use ordinal encoding. Defaults to False.
            atom_feature_params (dict, optional): Additional parameters (num_atom_types, num_atom_features) for atom feature encoding. Defaults to {}.
            edge_feature_params (dict, optional): Additional parameters for edge feature encoding. Defaults to {}.
            inter_message_passing (bool, optional): Whether to use inter-message passing (messages between atoms and fragments). Defaults to True.
            higher_message_passing (bool, optional): Whether to use higher-level message passing (messages between neighboring fragments). Defaults to False.
            no_frag_info (bool, optional): Whether to exclude all fragment information. Defaults to False.
            reduction (str, optional): Reduction method for aggregating atom features. Defaults to "mean".
            frag_reduction (str, optional): Reduction method for aggregating fragment features. If None, reduction is used. Defaults to None.
            learned_edge_rep (bool, optional): Whether to use learned edge representations. Defaults to False.
            inter_message_params (dict, optional): Additional parameters for inter-message passing (e.g., used reduction method). Defaults to {}.
            hidden_channels_substructure (int, optional): Number of hidden channels for substructure features. 
                If not provided, it defaults to `hidden_channels`. Defaults to None.
            num_layers_out (int, optional): Number of output layers. Defaults to 2.
        """
        super(FragGNNSmall, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels_substructure = hidden_channels_substructure if hidden_channels_substructure else hidden_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing
        self.higher_message_passing = higher_message_passing
        self.no_frag_info = no_frag_info
        self.ordinal_encoding = ordinal_encoding

        self.reduction = reduction
        self.frag_reduction = frag_reduction if frag_reduction else reduction

        self.learned_edge_rep = learned_edge_rep
        self.out_channels = out_channels

        self.atom_encoder = AtomEncoder(
            hidden_channels, False, **atom_feature_params)

        self.clique_encoder = FragEncoder(
            in_channels_substructure, self.hidden_channels_substructure, self.ordinal_encoding)

        if not self.learned_edge_rep:
            self.bond_encoders = ModuleList()
        else:
            self.bond_encoder = BondEncoder(
                hidden_channels, **edge_feature_params)
            self.atom2bond = ModuleList()
            self.bond_batch_norms = ModuleList()
            self.bond_convs = ModuleList()

        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()

        for _ in range(num_layers):
            if not self.learned_edge_rep:
                self.bond_encoders.append(BondEncoder(
                    hidden_channels, **edge_feature_params))

            nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            self.atom_convs.append(
                GINEConv(nn, train_eps=True, edge_dim=hidden_channels))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))

            if self.learned_edge_rep:
                self.atom2bond.append(InterMessage(
                    hidden_channels, hidden_channels, **inter_message_params))
                self.bond_batch_norms.append(BatchNorm1d(hidden_channels))
                self.bond_convs.append(
                    Linear(hidden_channels, hidden_channels))

        if self.inter_message_passing:
            self.frag_convs = ModuleList()
            self.frag_batch_norms = ModuleList()

            for _ in range(num_layers):
                nn = Sequential(
                    Linear(self.hidden_channels_substructure,
                           self.hidden_channels_substructure),
                    BatchNorm1d(self.hidden_channels_substructure),
                    ReLU(),
                    Linear(self.hidden_channels_substructure,
                           self.hidden_channels_substructure),
                )

                self.frag_convs.append(GINConv(nn, train_eps=True))
                self.frag_batch_norms.append(
                    BatchNorm1d(self.hidden_channels_substructure))

            self.atom2frag = ModuleList()
            self.frag2atom = ModuleList()

            for _ in range(num_layers):
                self.atom2frag.append(
                    InterMessage(hidden_channels, self.hidden_channels_substructure, **inter_message_params))
                self.frag2atom.append(
                    InterMessage(self.hidden_channels_substructure, hidden_channels, **inter_message_params))

        self.frag_out = MLP(
            self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        self.atom_out = MLP(
            self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        if self.learned_edge_rep:
            self.edge_out = MLP(
                self.hidden_channels, self.hidden_channels, num_layers=2, batch_norm=False)
        self.out = MLP(self.hidden_channels, self.out_channels,
                       num_layers=num_layers_out, batch_norm=False, last_relu=False)

    def forward(self, data):

        batch_size = torch.max(data.batch) + 1
        x = self.atom_encoder(data)

        if self.ordinal_encoding:
            x_frag = self.clique_encoder(data.fragment_types)
        else:
            x_frag = self.clique_encoder(data.fragments)

        if not self.inter_message_passing and not self.no_frag_info:
            # append frag information as node features
            row, col = data.fragments_edge_index
            x = x + scatter(x_frag[col], row, dim=0,
                            dim_size=x.size(0), reduce=self.reduction)

        if self.learned_edge_rep:
            x_edge = self.bond_encoder(data.edge_attr)

        for i in range(self.num_layers):

            if not self.learned_edge_rep:
                x_edge = self.bond_encoders[i](data.edge_attr)

            x = self.atom_convs[i](x, data.edge_index, x_edge)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.learned_edge_rep:
                row_edge, col_edge = data.edge_index
                x_edge = x_edge + self.atom2bond[i](torch.concat([x[row_edge], x[col_edge]], dim=0), torch.concat([torch.arange(
                    row_edge.size(0), dtype=torch.int64, device=row_edge.device) for _ in range(2)], dim=0), dim_size=row_edge.size(0))
                x_edge = self.bond_convs[i](x_edge)
                x_edge = self.bond_batch_norms[i](x_edge)
                x_edge = F.relu(x_edge)
                x_edge = F.dropout(x_edge, self.dropout,
                                   training=self.training)

            if self.inter_message_passing:
                row, col = data.fragments_edge_index

                atom_message = self.atom2frag[i](
                    x[row], col, dim_size=x_frag.size(0))
                x_frag = x_frag + atom_message

                if self.higher_message_passing:
                    x_frag = self.frag_convs[i](
                        x_frag, data.higher_edge_index)

                x_frag = self.frag_batch_norms[i](x_frag)
                x_frag = F.relu(x_frag)
                x_frag = F.dropout(x_frag, self.dropout,
                                   training=self.training)

                frag_message = self.frag2atom[i](
                    x_frag[col], row, dim_size=x.size(0))
                x = x + frag_message

        x = scatter(x, data.batch, dim=0, reduce=self.reduction)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_out(x)

        if self.inter_message_passing:
            x_frag = scatter(x_frag, data.fragments_batch, dim=0, dim_size=x.size(0),
                             reduce=self.frag_reduction)
            x_frag = F.dropout(x_frag, self.dropout,
                               training=self.training)
            x_frag = self.frag_out(x_frag)
            x = x + x_frag

        if self.learned_edge_rep:
            edge_batch = data.batch[data.edge_index[0]]
            x_edge = scatter(x_edge, edge_batch, dim=0,
                             dim_size=batch_size, reduce=self.reduction)
            x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            x_edge = self.edge_out(x_edge)
            x = x + x_edge

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out(x)
        return x
