from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GINEConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree
from torch.nn import ModuleList, Module, Sequential, Linear, ReLU, BatchNorm1d, Embedding
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch
from models.layers import FragEncoder, AtomEncoder, BondEncoder, InterMessage, MLP
import numpy as np


class HLG(torch.nn.Module):

    def __init__(self, in_channels, in_channels_edge, in_channels_frag,
          hidden_channels, hidden_channels_edge = None, hidden_channels_frag = None, hidden_channels_mol = None,
          out_channels = 1, num_layers = 3, num_layers_message_before = 0, 
          num_layers_message_after = 2, num_layers_out = 1,
          dropout = 0, ordinal_encoding = True, 
          reduction = "mean", reduction_out = "mean", message_types = {}, 
          higher_level_edge_features = False):
        super(HLG, self).__init__()
        self.in_channels = in_channels
        self.in_channels_edge = in_channels_edge
        self.in_channels_frag = in_channels_frag
        self.hidden_channels = hidden_channels
        self.hidden_channels_edge = hidden_channels_edge if hidden_channels_edge else hidden_channels
        self.hidden_channels_frag = hidden_channels_frag if hidden_channels_frag else hidden_channels
        self.hidden_channels_mol = hidden_channels_mol if hidden_channels_mol else hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.ordinal_encoding = ordinal_encoding
        self.reduction = reduction
        self.reduction_out = reduction_out
        self.message_types = message_types.copy() # copy needed to be able to change dict
        self.set_default_message_types()

        #Encoders
        self.atom_encoder = AtomEncoder(self.hidden_channels)
        self.frag_encoder = FragEncoder(self.in_channels_frag, self.hidden_channels_frag, encoding_size_scaling=ordinal_encoding)
        self.edge_encoder = BondEncoder(self.hidden_channels_edge)

        #Model
        if self.message_types["edge2edge"] or self.message_types["edge2atom"]:
            # edge2atom part of atom2atom
            raise NotImplementedError
        
        for message_type in self.message_types:
            if self.message_types[message_type]:
                self.__setattr__(message_type, ModuleList())

        self.atom_batch_norms = ModuleList()
        self.edge_batch_norms = ModuleList()
        self.frag_batch_norms = ModuleList()
        self.mol_batch_norms = ModuleList()

        self.combine_atom_messages = ModuleList()
        self.combine_edge_messages = ModuleList()
        self.combine_frag_messages = ModuleList()
        self.combine_mol_messages = ModuleList()

        for i in range(num_layers):
            if self.message_types["atom2atom"]:
                self.atom2atom.append(InterMessage(self.hidden_channels + self.hidden_channels_edge, self.hidden_channels, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction)) #TODO: +1 seems inelegant
            if self.message_types["atom2edge"]:
                self.atom2edge.append(InterMessage(self.hidden_channels, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["atom2frag"]:
                self.atom2frag.append(InterMessage(self.hidden_channels, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["atom2mol"]:
                self.atom2mol.append(InterMessage(self.hidden_channels, self.hidden_channels))
            
            
            if self.message_types["edge2frag"]:
                self.edge2frag.append(InterMessage(self.hidden_channels_edge, self.hidden_channels_edge, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["edge2mol"]:
                self.edge2mol.append(InterMessage(self.hidden_channels_edge, self.hidden_channels_edge, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            
            if self.message_types["frag2atom"]:
                self.frag2atom.append(InterMessage(self.hidden_channels_frag, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["frag2edge"]:
                self.frag2edge.append(InterMessage(self.hidden_channels_frag, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["frag2frag"]:
                self.frag2frag.append(InterMessage(self.hidden_channels_frag, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["frag2mol"]:
                self.frag2mol.append(InterMessage(self.hidden_channels_frag, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            
            if self.message_types["mol2atom"]:
                self.mol2atom.append(InterMessage(self.hidden_channels_mol, self.hidden_channels_mol, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["mol2edge"]:
                self.mol2edge.append(InterMessage(self.hidden_channels_mol, self.hidden_channels_mol, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["mol2frag"]:
                self.mol2frag.append(InterMessage(self.hidden_channels_mol, self.hidden_channels_mol, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))

            #self.edge2atom.append(InterMessage(self.hidden_channels, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            size_message_atom = self.hidden_channels * self.message_types["atom2atom"] + self.hidden_channels_frag * self.message_types["frag2atom"] + self.hidden_channels_mol * self.message_types["mol2atom"]
            size_message_edge = self.hidden_channels * self.message_types["atom2edge"] + self.hidden_channels_frag * self.message_types["frag2edge"] + self.hidden_channels_mol * self.message_types["mol2edge"]
            size_message_frag = self.hidden_channels * self.message_types["atom2frag"] + self.hidden_channels_edge * self.message_types["edge2frag"] + self.hidden_channels_frag * self.message_types["frag2frag"] + self.hidden_channels_mol * self.message_types["mol2frag"]
            size_message_mol = self.hidden_channels * self.message_types["atom2mol"] + self.hidden_channels_edge * self.message_types["edge2mol"] + self.hidden_channels_frag * self.message_types["frag2mol"]

            self.combine_atom_messages.append(MLP(in_channels = size_message_atom, out_channels = self.hidden_channels, num_layers = 1))
            self.combine_edge_messages.append(MLP(in_channels = size_message_edge, out_channels = self.hidden_channels_edge, num_layers = 1))
            self.combine_frag_messages.append(MLP(in_channels = size_message_frag, out_channels = self.hidden_channels_frag, num_layers = 1))
            self.combine_mol_messages.append(MLP(in_channels = size_message_mol, out_channels = self.hidden_channels_mol, num_layers = 1))

            self.atom_batch_norms.append(BatchNorm1d(self.hidden_channels))
            self.edge_batch_norms.append(BatchNorm1d(self.hidden_channels_edge))
            self.frag_batch_norms.append(BatchNorm1d(self.hidden_channels_frag))
            self.mol_batch_norms.append(BatchNorm1d(self.hidden_channels_mol))
        
        self.frag_out = MLP(self.hidden_channels_frag, self.hidden_channels, num_layers = 2)
        self.atom_out = MLP(self.hidden_channels, self.hidden_channels, num_layers = 2)
        self.edge_out = MLP(self.hidden_channels_edge, self.hidden_channels, num_layers = 2)
        self.mol_out = MLP(self.hidden_channels_mol, self.hidden_channels, num_layers = 2)
        self.out = MLP(self.hidden_channels, self.out_channels, num_layers = num_layers_out, batch_norm = False, last_relu = False)
    
    def set_default_message_types(self):
        defaults = {"atom2atom": True, "atom2edge": True, "atom2frag": True, "atom2mol": False, 
                    "edge2atom": False, "edge2edge": False, "edge2frag": True, "edge2mol" : False, 
                    "frag2atom": True, "frag2edge": False, "frag2frag": True, "frag2mol": False,
                    "mol2atom": False, "mol2edge": False, "mol2frag": False}
        for msg_type,val in defaults.items():
            if msg_type not in self.message_types:
                self.message_types[msg_type] = val


    def forward(self, data):
        row, col = data.fragments_edge_index
        row_edge, col_edge = data.edge_index
        edge_batch =  data.batch[data.edge_index[0]]
        batch_size = torch.max(data.batch) + 1
        x = self.atom_encoder(data)
        x_edge = self.edge_encoder(data.edge_attr)
        
        if self.ordinal_encoding:
            x_frag = self.frag_encoder(data.fragment_types)
        else:
            x_frag = self.frag_encoder(data.fragments)
        
        x_mol = torch.zeros((batch_size, self.hidden_channels_mol), device = x.device)
        
        for layer_ind in range(self.num_layers):

            # update atom representation
            atom_messages = []
            if self.message_types["atom2atom"]:
                atom_messages.append(self.atom2atom[layer_ind](torch.concat([x[row_edge], x_edge], dim = -1), col_edge, dim_size = x.size(0)))
            if self.message_types["frag2atom"]:
                atom_messages.append(self.frag2atom[layer_ind](x_frag[col], row, dim_size=x.size(0)))
            if self.message_types["mol2atom"]:
                atom_messages.append(self.mol2atom[layer_ind](x_mol[data.batch], torch.arange(x.size(0), device = x.device, dtype = torch.int64), dim_size = x.size(0)))
            
            if atom_messages:
                x = x + self.combine_atom_messages[layer_ind](torch.concat(atom_messages, dim = -1))
                x = self.atom_batch_norms[layer_ind](x)
                x = F.relu(x)
            
            # update edge representation
            edge_messages = []
            if self.message_types["atom2edge"]:
                edge_messages.append(self.atom2edge[layer_ind](torch.concat([x[row_edge], x[col_edge]], dim = 0), torch.concat([torch.arange(row_edge.size(0), dtype=torch.int64, device = row_edge.device) for _ in range(2)], dim = 0), dim_size = row_edge.size(0)))
            if self.message_types["frag2edge"]:
                edge_index, frag_index = data.low_high_edge_index
                edge_messages.append(self.frag2edge[layer_ind](x_frag[frag_index], edge_index, dim_size = x_edge.size(0)))
            if self.message_types["mol2edge"]:
                edge_messages.append(self.mol2edge[layer_ind](x_mol[edge_batch], torch.arange(x_edge.size(0), device = x_edge.device, dtype = torch.int64), dim_size = x_edge.size(0)))
            
            if edge_messages:
                x_edge = x_edge + self.combine_edge_messages[layer_ind](torch.concat(edge_messages, dim = -1))
                x_edge = self.edge_batch_norms[layer_ind](x_edge)
                x_edge = F.relu(x_edge)
            

            # updata frag representation
            frag_messages = []
            if self.message_types["atom2frag"]:
                frag_messages.append(self.atom2frag[layer_ind](x[row], col, dim_size = x_frag.size(0)))
            if self.message_types["edge2frag"]:
                edge_index, frag_index = data.low_high_edge_index
                frag_messages.append(self.edge2frag[layer_ind](x_edge[edge_index], frag_index, dim_size = x_frag.size(0)))
            if self.message_types["frag2frag"]:
                row_higher, col_higher = data.higher_edge_index
                frag_messages.append(self.frag2frag[layer_ind](x_frag[row_higher], col_higher, dim_size = x_frag.size(0)))
            if self.message_types["mol2frag"]:
                frag_messages.append(self.mol2frag[layer_ind](x_mol[data.fragments_batch], torch.arange(x_frag.size(0), device = x_frag.device, dtype = torch.int64), dim_size = x_frag.size(0)))
            
            if frag_messages:
                x_frag = x_frag + self.combine_frag_messages[layer_ind](torch.concat(frag_messages, dim = -1))
                x_frag = self.frag_batch_norms[layer_ind](x_frag)
                x_frag = F.relu(x_frag)
            
            #update mol representation
            mol_messages = []
            if self.message_types["atom2mol"]:
                mol_messages.append(self.atom2mol[layer_ind](x, data.batch, dim_size = batch_size))
            if self.message_types["edge2mol"]:
                mol_messages.append(self.edge2mol[layer_ind](x_edge, edge_batch, dim_size = batch_size))
            if self.message_types["frag2mol"]:
                mol_messages.append(self.frag2mol[layer_ind](x_frag, data.fragments_batch, dim_size = batch_size))
            
            if mol_messages:
                x_mol = x_mol + self.combine_mol_messages[layer_ind](torch.concat(mol_messages, dim = -1))
                x_mol = self.mol_batch_norms[layer_ind](x_mol)
                x_mol = F.relu(x_mol)
              
        
        x = scatter(x, data.batch, dim=0, reduce = self.reduction_out)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_out(x)

        edge_batch =  data.batch[data.edge_index[0]]
        x_edge = scatter(x_edge, edge_batch, dim = 0, dim_size = batch_size, reduce = self.reduction_out)
        x_edge = F.dropout(x_edge, self.dropout, training=self.training)
        x_edge = self.edge_out(x_edge)
        x = x + x_edge

        x_frag = scatter(x_frag, data.fragments_batch, dim=0, dim_size=batch_size,reduce=self.reduction_out)
        x_frag = F.dropout(x_frag, self.dropout, training=self.training)
        x_frag = self.frag_out(x_frag)
        x = x + x_frag

        x_mol = F.dropout(x_mol, self.dropout, training = self.training)
        x_mol = self.mol_out(x_mol)
        x = x + x_mol
        
        x = self.out(x)
        return x