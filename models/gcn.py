from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GINEConv
from torch_geometric.nn.norm import BatchNorm
from torch.nn import ModuleList, Module, Sequential, Linear, ReLU, BatchNorm1d, Embedding
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch

class GCN(Module):
    def __init__(self, hidden_channels, out_channels, in_channels, num_layers, dropout = 0.5, residual = False, batch_norm = False, graph_level = False, pool_reduction = "sum"):
        super().__init__()
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction
        self.dropout = dropout
        self.feature_encoder = Linear(in_channels, hidden_channels)
        
        self.residual = residual
        
        
        
        self.layers = ModuleList()
        self.batch_norms = ModuleList()
        for layer_ind in range(num_layers):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            if batch_norm:
                self.batch_norms.append(BatchNorm(in_channels = hidden_channels))
        self.out = MLPReadout(hidden_channels, output_dim= 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.feature_encoder(x)
        for layer_ind, layer in enumerate(self.layers):
            
            if self.residual:
                x_c = x

            x = layer(x, edge_index)

            if self.residual:
                x += x_c
            if self.batch_norms:
                x = self.batch_norms[layer_ind](x)

            if layer_ind != len(self.layers) -1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.relu()
        
        if self.graph_level:
            # Pooling 
            x = scatter(x, data.batch, dim=0, reduce = self.pool_reduction)
        x = self.out(x)
        return x
    
class GCNSubstructure(Module):
    def __init__(self, hidden_channels, hidden_channels_substructure, out_channels, in_channels, in_channels_substructure, num_layers, dropout = 0.5, residual = False, batch_norm = True, substructure_batch_norm = True, graph_level = False, pool_reduction = "mean", substructure_messages = True):
        super().__init__()
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction
        self.dropout = dropout
        self.feature_encoder = Linear(in_channels, hidden_channels)
        self.feature_encoder_substructure = Linear(in_channels_substructure, hidden_channels_substructure)
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
            self.to_substructure.append(SAGEConv((hidden_channels, hidden_channels_substructure), hidden_channels_substructure))
            self.from_substructure.append(SAGEConv((hidden_channels_substructure, hidden_channels), hidden_channels))
            if batch_norm:
                self.batch_norms.append(BatchNorm(in_channels = hidden_channels))
            if substructure_batch_norm:
                self.batch_norms_substructure.append(BatchNorm(in_channels=hidden_channels_substructure))
        self.out = MLPReadout(hidden_channels + hidden_channels_substructure, output_dim= out_channels) if graph_level else MLPReadout(hidden_channels, output_dim= out_channels)

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
                x_substructure = to_sub((x, x_substructure), substructure_edge_index)
                if self.batch_norms_substructure:
                    x_substructure = self.batch_norms_substructure[layer_ind](x_substructure)
                x_substructure = F.relu(x_substructure)
                x_substructure = F.dropout(x_substructure, self.dropout, training = self.training)
                x_substructure = x_substructure + x_substructure_c

                x = from_sub((x_substructure, x), substructure_edge_index[[1,0]])

            if self.residual:
                x += x_c

        
        if self.graph_level:
            # Pooling 
            x = scatter(x, data.x_batch, dim=0, reduce = self.pool_reduction)
            x_substructure = scatter(x_substructure, data.fragments_batch, dim=0, reduce = self.pool_reduction, dim_size = batch_size)
            x = self.out(torch.concat([x, x_substructure], dim = 1))
        else:
            x = self.out(x)
        return x

class HimpNet(torch.nn.Module):
    """Adapted from https://github.com/rusty1s/himp-gnn/blob/master/model.py"""

    def __init__(self, in_channels, in_channels_substructure, in_channels_edge, hidden_channels, out_channels, num_layers, dropout=0.0,
                 inter_message_passing=True):
        super(HimpNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing

        #self.atom_encoder = Linear(in_channels, hidden_channels)
        self.atom_encode = AtomEncoder(hidden_channels)
        self.clique_encoder = Linear(in_channels_substructure, hidden_channels)

        self.bond_encoders = ModuleList()
        self.atom_convs = ModuleList()
        self.atom_batch_norms = ModuleList()

        for _ in range(num_layers):
            #self.bond_encoders.append(Linear(in_channels_edge, hidden_channels))
            self.bond_encoders.append(BondEncoder(hidden_channels))
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))

        self.clique_convs = ModuleList()
        self.clique_batch_norms = ModuleList()

        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.clique_convs.append(GINConv(nn, train_eps=True))
            self.clique_batch_norms.append(BatchNorm1d(hidden_channels))

        self.atom2clique_lins = ModuleList()
        self.clique2atom_lins = ModuleList()

        for _ in range(num_layers):
            self.atom2clique_lins.append(
                Linear(hidden_channels, hidden_channels))
            self.clique2atom_lins.append(
                Linear(hidden_channels, hidden_channels))

        self.atom_lin = Linear(hidden_channels, hidden_channels)
        self.clique_lin = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.clique_encoder.reset_parameters()

        for emb, conv, batch_norm in zip(self.bond_encoders, self.atom_convs,
                                         self.atom_batch_norms):
            emb.reset_parameters()
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
        x = self.atom_encoder(data.x)

        if self.inter_message_passing:
            x_clique = self.clique_encoder(data.fragments)

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr)
            x = self.atom_convs[i](x, data.edge_index, edge_attr)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.inter_message_passing:
                row, col = data.fragments_edge_index

                x_clique = x_clique + F.relu(self.atom2clique_lins[i](scatter(
                    x[row], col, dim=0, dim_size=x_clique.size(0),
                    reduce='mean')))

                #x_clique = self.clique_convs[i](x_clique, data.tree_edge_index)
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
            #tree_batch = torch.repeat_interleave(data.num_cliques)
            x_clique = scatter(x_clique, data.fragments_batch, dim=0, dim_size=x.size(0),
                               reduce='mean')
            x_clique = F.dropout(x_clique, self.dropout,
                                 training=self.training)
            x_clique = self.clique_lin(x_clique)
            x = x + x_clique

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return x
    
class VerySimpleGCN(Module):
    def __init__(self, hidden_channels, out_channels, in_channels, num_layers, dropout = 0.5, graph_level = False, pool_reduction = "sum"):
        super().__init__()
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction
        self.dropout = dropout
        
        self.layers = ModuleList()
        for layer_ind in range(num_layers):
            gcn_in = in_channels if layer_ind == 0 else hidden_channels
            gcn_out = out_channels if layer_ind == num_layers -1 else hidden_channels
            self.layers.append(GCNConv(gcn_in, gcn_out))
 

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for layer_ind, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if layer_ind != len(self.layers) -1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        
        if self.graph_level:
            # Pooling 
            x = scatter(x, data.batch, dim=0, reduce= self.pool_reduction)

        return x
    
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class AtomEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AtomEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            self.embeddings.append(Embedding(100, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


class BondEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BondEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(3):
            self.embeddings.append(Embedding(6, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out
