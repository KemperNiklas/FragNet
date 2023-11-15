from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GINEConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree
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

    def __init__(self, in_channels, in_channels_substructure, in_channels_edge, 
                 hidden_channels, out_channels, num_layers, dropout=0.0,
                 linear_atom_encoder = False, encoding_size_scaling = False, 
                 degree_scaling = False, additional_atom_features = [], 
                 inter_message_passing=True, higher_message_passing = False, 
                 low_high_edges = False, fragment_specific = False, 
                 reduction = "mean", concat = False, graph_rep = False, graph_rep_node = False):
        super(HimpNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.inter_message_passing = inter_message_passing
        self.higher_message_passing = higher_message_passing
        self.low_high_edges = low_high_edges
        self.encoding_size_scaling = encoding_size_scaling
        self.degree_scaling = degree_scaling
        self.fragment_specific = fragment_specific
        self.reduction = reduction
        self.concat = concat
        self.graph_rep = graph_rep
        self.graph_rep_node = graph_rep_node


        #self.atom_encoder = Linear(in_channels, hidden_channels)
        self.atom_encoder = Linear(in_channels, hidden_channels) if linear_atom_encoder else AtomEncoder(hidden_channels, degree_scaling, additional_atom_features)
        if self.inter_message_passing:
            self.clique_encoder = CliqueEncoder(in_channels_substructure, hidden_channels, encoding_size_scaling)
        self.bond_encoders = ModuleList()
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
            #self.bond_encoders.append(Linear(in_channels_edge, hidden_channels))
            self.bond_encoders.append(BondEncoder(hidden_channels))
            if self.low_high_edges:
                self.bond_encoders_low_high.append(BondEncoder(hidden_channels))
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.atom_convs.append(GINEConv(nn, train_eps=True))
            self.atom_batch_norms.append(BatchNorm1d(hidden_channels))
            if self.graph_rep_node:
                self.atom2graph.append(Linear(hidden_channels, hidden_channels_graph))
                self.graph2atom.append(Linear(hidden_channels_graph, hidden_channels))
            if self.graph_rep or self.graph_rep_node:
                self.graph_batch_norms.append(BatchNorm1d(hidden_channels_graph))
                self.graph_conv.append(Linear(hidden_channels_graph, hidden_channels_graph))

        if self.inter_message_passing:
            self.clique_convs = ModuleList()
            self.clique_batch_norms = ModuleList()
            if self.graph_rep:
                self.fragment2graph = ModuleList()
                self.graph2fragment = ModuleList()
            if self.concat:
                self.concat_lins = ModuleList()

            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                )
                self.clique_convs.append(GINConv(nn, train_eps=True))
                self.clique_batch_norms.append(BatchNorm1d(hidden_channels))
                if self.concat:
                    self.concat_lins.append(Linear(2*hidden_channels, hidden_channels))
                if self.graph_rep:
                    self.fragment2graph.append(Linear(hidden_channels, hidden_channels_graph))
                    self.graph2fragment.append(Linear(hidden_channels_graph, hidden_channels))
                    
                    

            self.atom2clique_lins = ModuleList()
            self.clique2atom_lins = ModuleList()

            for _ in range(num_layers):
                if not self.fragment_specific:
                    self.atom2clique_lins.append(
                        Linear(hidden_channels, hidden_channels))
                    self.clique2atom_lins.append(
                        Linear(hidden_channels, hidden_channels))
                else:
                    self.atom2clique_lins.append(
                        ModuleList([Linear(hidden_channels, hidden_channels) for i in range(3)]))
                    self.clique2atom_lins.append(
                        ModuleList([Linear(hidden_channels, hidden_channels) for i in range(3)]))
                    
        self.atom_lin = Linear(hidden_channels, hidden_channels)
        self.clique_lin = Linear(hidden_channels, hidden_channels)     
        if self.graph_rep or self.graph_rep_node:
            self.graph_lin = Linear(hidden_channels_graph, hidden_channels)
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

        batch_size = torch.max(data.batch) + 1
        if self.degree_scaling:
            degrees = degree(data.edge_index[0], dtype = torch.float, num_nodes= data.x.size(0)) 
            x = self.atom_encoder(data, degrees)
        else:
            x = self.atom_encoder(data)
        
        if self.inter_message_passing:
            if self.encoding_size_scaling:
                x_clique = self.clique_encoder(data.fragment_types)
            else:
                x_clique = self.clique_encoder(data.fragments)

        if self.graph_rep:
            x_graph = torch.zeros(batch_size, dtype = torch.int, device = x.device)
            x_graph = self.graph_encoder(x_graph)

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr)
            x = self.atom_convs[i](x, data.edge_index, edge_attr)
            x = self.atom_batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.graph_rep_node:
                x_graph = x_graph + F.relu(self.atom2graph[i](scatter(
                        x, data.batch, dim=0, dim_size = batch_size, reduce='mean')))

            if self.inter_message_passing:
                row, col = data.fragments_edge_index

                if self.graph_rep:
                    #frag to graph
                    x_graph = x_graph + F.relu(self.fragment2graph[i](scatter(
                        x_clique, data.fragments_batch, dim=0, dim_size = batch_size, reduce='mean')))

                if self.fragment_specific:
                    subgraph_message = torch.zeros_like(x_clique)
                    edge_masks = [data.fragment_types[data.fragments_edge_index[1], 0] == i for i in range(3)]
                    for edge_mask, message in zip(edge_masks, self.clique2atom_lins[i]):
                        subgraph_message += message(scatter(
                            x[row[edge_mask]], col[edge_mask], dim=0, dim_size=x_clique.size(0),reduce='mean'))
                    subgraph_message = F.relu(subgraph_message)
                else:
                    subgraph_message = F.relu(self.atom2clique_lins[i](scatter(
                        x[row], col, dim=0, dim_size=x_clique.size(0),
                        reduce='mean')))
                x_clique = x_clique + subgraph_message
                
                if self.low_high_edges:
                    edges, frags = data.low_high_edge_index
                    edge_attr_new = self.bond_encoders_low_high[i](data.edge_attr)
                    x_clique = x_clique + scatter(edge_attr_new[edges], frags, dim = 0, dim_size = x_clique.size(0), reduce = "mean")

                if self.higher_message_passing:
                    x_clique = self.clique_convs[i](x_clique, data.higher_edge_index)

                # if self.graph_rep:
                #     #graph to frag
                #     x_clique = x_clique + F.relu(self.graph2fragment[i](x_graph[data.fragments_batch]))

                
                x_clique = self.clique_batch_norms[i](x_clique)
                x_clique = F.relu(x_clique)
                x_clique = F.dropout(x_clique, self.dropout,
                                     training=self.training)
                
                
                if self.fragment_specific:
                    subgraph_message = torch.zeros_like(x)
                    edge_masks = [data.fragment_types[data.fragments_edge_index[1], 0] == i for i in range(3)]
                    for edge_mask, message in zip(edge_masks, self.clique2atom_lins[i]):
                        subgraph_message += message(scatter(
                            x_clique[col[edge_mask]], row[edge_mask], dim=0, dim_size=x.size(0),reduce=self.reduction))
                    subgraph_message = F.relu(subgraph_message)
                else:
                    subgraph_message = F.relu(self.clique2atom_lins[i](scatter(
                        x_clique[col], row, dim=0, dim_size=x.size(0),
                        reduce = self.reduction)))
                if self.concat:
                    x = self.concat_lins[i](torch.concat([x, subgraph_message], dim = -1))
                else:
                    x = x + subgraph_message
            
            if self.graph_rep or self.graph_rep_node:
                x_graph = F.relu(self.graph_conv[i](x_graph))
                x_graph = self.graph_batch_norms[i](x_graph)
                if self.graph_rep:
                    #graph to frag
                    x_clique = x_clique + F.relu(self.graph2fragment[i](x_graph[data.fragments_batch]))
                if self.graph_rep_node:
                    #graph to node
                    x = x + F.relu(self.graph2atom[i](x_graph[data.batch]))



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
        
        if self.graph_rep:
            x = x + self.graph_lin(x_graph)

        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        return x

class HimpNetHigherGraph(torch.nn.Module):
    """Adapted from https://github.com/rusty1s/himp-gnn/blob/master/model.py"""

    def __init__(self, in_channels, in_channels_substructure, in_channels_edge, hidden_channels, out_channels, num_layers, dropout=0.0,
                 linear_atom_encoder = False, encode_atoms = True, encode_bonds = False, encoding_size_scaling = True):
        super(HimpNetHigherGraph, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_atoms = encode_atoms
        self.encode_bonds = encode_bonds
        self.encoding_size_scaling = encoding_size_scaling

        #self.atom_encoder = Linear(in_channels, hidden_channels)
        embedding_size1 = hidden_channels // 2 if encode_atoms else hidden_channels
        embedding_size2 = (hidden_channels +1) // 2 if encode_atoms else hidden_channels
        self.atom_encoder = Linear(in_channels, embedding_size1) if linear_atom_encoder else AtomEncoder(embedding_size1)
        self.clique_encoder = CliqueEncoder(in_channels_substructure, embedding_size2, encoding_size_scaling)

        self.bond_encoder = BondEncoder(hidden_channels)

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

        self.atom2clique_lin = Linear(embedding_size1, embedding_size1)
        self.bond2clique_lin = Linear(hidden_channels, hidden_channels)

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

        if self.encoding_size_scaling:
            x_clique = self.clique_encoder(data.fragment_types)
        else:
            x_clique = self.clique_encode(data.fragments)

        if self.encode_atoms:
            x = self.atom_encoder(data)
            row, col = data.fragments_edge_index
            x_clique = torch.concat([x_clique, F.relu(self.atom2clique_lin(scatter(
                    x[row], col, dim=0, dim_size=x_clique.size(0),
                    reduce='mean')))], dim = -1)
        
        if self.encode_bonds:
            bonds = self.bond_encoder(data.edge_attr)
            raise NotImplementedError()
    
        for i in range(self.num_layers):
            x_clique = self.clique_convs[i](x_clique, data.higher_edge_index)
            x_clique = F.relu(x_clique)
            x_clique = F.dropout(x_clique, self.dropout,
                                    training=self.training)
        
        batch_size = max(data.x_batch) + 1
        x_clique = scatter(x_clique, data.fragments_batch, dim=0, dim_size=batch_size,
                               reduce='mean')
        x_clique = F.dropout(x_clique, self.dropout,
                                    training=self.training)
        x_clique = self.lin(x_clique)
        return x_clique

        
    
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
    def __init__(self, hidden_channels, degree_scaling = False, additional_encoding = []):
        super(AtomEncoder, self).__init__()
        self.degree_scaling = degree_scaling
        self.hidden_channels = hidden_channels
        self.additional_encoding = additional_encoding

        self.embeddings = torch.nn.ModuleList()

        if additional_encoding:
            additional_sizes = sum([size for keyword, size in additional_encoding])
            self.lin = Linear(hidden_channels + additional_sizes, hidden_channels)

        for i in range(9):
            self.embeddings.append(Embedding(100, hidden_channels))
        

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, graph, degree_info = None):
        x = graph.x
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])

        if self.degree_scaling:
            out[:,:self.hidden_channels//2] = torch.unsqueeze(degree_info, dim = 1) * out[:,:self.hidden_channels//2]
        
        if self.additional_encoding:
            additional_features = [getattr(graph, name) for (name, size) in self.additional_encoding]
            additional_features.append(out)
            out  = self.lin(torch.concat(additional_features, dim = -1))

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
    
class BetweenMessage(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 1, transform_scatter = False):
        super(BetweenMessage, self).__init__()

        self.transform_scatter = transform_scatter

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(Linear(in_channels, out_channels))
            else:
                layers.append(Linear(out_channels, out_channels))
            
            if i != num_layers-1:
                layers.append(ReLU())

        self.transform = Sequential(*layers)
    
    def forward(self, from_tensor, to_index, dim_size, reduction = "mean"):
        if self.transform_scatter:
            # first transform then scatter
            message = self.transform(from_tensor)
            message = scatter(message, to_index, dim = 0, dim_size =dim_size, reduce = reduction)
        else:
            # first scatter then transform
            message = scatter(from_tensor, to_index, dim = 0, dim_size =dim_size, reduce = reduction)
            message = self.transform(message)
        return message
        
        
        

class CliqueEncoder(torch.nn.Module):

    def __init__(self, in_channels_substructure, hidden_channels, encoding_size_scaling):
        super(CliqueEncoder, self).__init__()
        self.encoding_size_scaling = encoding_size_scaling
        self.hidden_channels = hidden_channels
        if not encoding_size_scaling:
            self.embedding = Embedding(in_channels_substructure, hidden_channels)
        else:
            self.embedding = Embedding(4, hidden_channels) #embed paths, junction, ring
    
    def forward(self, clique_attr):
        
        if not self.encoding_size_scaling:
            # clique attr are currently one hot encoded 
            clique_attr = torch.argmax(clique_attr, dim = 1)
            return self.embedding(clique_attr)
        else:
            assert(clique_attr.size(1) == 2)
            embeddings = self.embedding(clique_attr[:,0])
            embeddings[:,:self.hidden_channels//2] = torch.unsqueeze(clique_attr[:,1], dim = 1) * embeddings[:,:self.hidden_channels//2]
            return embeddings

