import torch
import torch.nn.functional as F
from torch.nn import Embedding, ModuleList
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_scatter import scatter
from torch_geometric.nn import GINConv, GINEConv


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.0,
                 inter_message_passing=True, motifs = [], motif_functions = []):
        
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing

        self.node_convs = ModuleList()
        self.node_batch_norms = ModuleList()

        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))

        self.motif_convs = [ModuleList() for _ in range(len(motifs))]

        for motif in range(len(motifs)):
            for _ in range(num_layers):
                self.motif_convs[motif].append(motif_functions[motif]())


    def reset_parameters(self):

        for conv, batch_norm in zip(self.node_convs,self.node_batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()

        for conv, batch_norm in zip(self.clique_convs,
                                    self.clique_batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()

        for lin1, lin2 in zip(self.atom2clique_lins, self.clique2atom_lins):
            lin1.reset_parameters()
            lin2.reset_parameters()

        self.atom_lin.reset_parameters()
        self.clique_lin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x = data.x

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr)
            x = self.atom_convs[i](x, data.edge_index, edge_attr)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            
            row, col = data.atom2clique_index

            x_clique = x_clique + F.relu(self.atom2clique_lins[i](scatter(
                x[row], col, dim=0, dim_size=x_clique.size(0),
                reduce='mean')))

            x_clique = self.clique_convs[i](x_clique, data.tree_edge_index)
            x_clique = self.clique_batch_norms[i](x_clique)
            x_clique = F.relu(x_clique)
            x_clique = F.dropout(x_clique, self.dropout,
                                    training=self.training)

            x = x + F.relu(self.clique2atom_lins[i](scatter(
                x_clique[col], row, dim=0, dim_size=x.size(0),
                reduce='mean')))

        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_lin(x)

        if self.inter_message_passing:
            tree_batch = torch.repeat_interleave(data.num_cliques)
            x_clique = scatter(x_clique, tree_batch, dim=0, dim_size=x.size(0),
                               reduce='mean')
            x_clique = F.dropout(x_clique, self.dropout,
                                 training=self.training)
            x_clique = self.clique_lin(x_clique)
            x = x + x_clique

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return x