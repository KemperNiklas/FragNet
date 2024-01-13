from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GINEConv, GATv2Conv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import degree
from torch.nn import ModuleList, Module, Sequential, Linear, ReLU, BatchNorm1d, Embedding
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch
from models.layers import FragEncoder, AtomEncoder, BondEncoder, InterMessage, MLP
import numpy as np
from typing import Literal, Optional


class HLG(torch.nn.Module):

    def __init__(self, in_channels, in_channels_edge, in_channels_frag,
          hidden_channels, hidden_channels_edge = None, hidden_channels_frag = None, hidden_channels_mol = None,
          out_channels = 1, num_layers = 3, num_layers_message_before = 0, 
          num_layers_message_after = 2, num_layers_out = 1,
          dropout = 0, ordinal_encoding = True, 
          reduction = "mean", reduction_out = "mean", reduction_nodes = None, message_types = {}, 
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
        self.reduction_nodes = reduction_nodes if reduction_nodes else reduction
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
                self.atom2atom.append(InterMessage(self.hidden_channels + self.hidden_channels_edge, self.hidden_channels, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction_nodes)) #TODO: +1 seems inelegant
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
            self.mol_rep_changes = size_message_mol > 0

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

        if self.mol_rep_changes:
            x_mol = F.dropout(x_mol, self.dropout, training = self.training)
            x_mol = self.mol_out(x_mol)
            x = x + x_mol
        
        x = self.out(x)
        return x

class HLGAlternative(torch.nn.Module):

    def __init__(self, in_channels, in_channels_edge, in_channels_frag,
          hidden_channels, hidden_channels_edge = None, hidden_channels_frag = None, hidden_channels_mol = None,
          out_channels = 1, num_layers = 3, num_layers_message_before = 0, 
          num_layers_message_after = 2, num_layers_out = 1,
          dropout = 0, ordinal_encoding = True, 
          reduction = "mean", reduction_out = "mean", message_types = {}, 
          higher_level_edge_features: Optional[Literal["aggregated", "basic"]] = None, attention = False, concat = 0, residuals = False, process_after = 0):
        super(HLGAlternative, self).__init__()
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
        self.attention = attention
        self.higher_level_edge_features = higher_level_edge_features
        self.concat = concat
        self.residuals = residuals
        self.process_after = process_after

        #Encoders
        self.atom_encoder = AtomEncoder(self.hidden_channels)
        self.frag_encoder = FragEncoder(self.in_channels_frag, self.hidden_channels_frag, encoding_size_scaling=ordinal_encoding)
        self.edge_encoder = BondEncoder(self.hidden_channels_edge)
        if self.higher_level_edge_features == "basic":
            # only encode if it is a node or edge connection
            self.higher_edge_encoder = Embedding(2, self.hidden_channels_edge)
        if self.concat != 0:
            self.frag_concat_encoder = FragEncoder(self.in_channels_frag, self.concat, encoding_size_scaling=ordinal_encoding)

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

        if self.process_after:
            self.process_after_atoms = ModuleList()
            self.process_after_edges = ModuleList()
            self.process_after_frags = ModuleList()

        for i in range(num_layers):
            if self.message_types["atom2atom"]:
                if self.attention:
                    self.atom2atom.append(GATv2Conv(self.hidden_channels, self.hidden_channels, heads = 1, edge_dim = self.hidden_channels_edge, add_self_loops = False))
                else: 
                    self.atom2atom.append(InterMessage(self.hidden_channels + self.hidden_channels_edge, self.hidden_channels, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction)) #TODO: +1 seems inelegant
            if self.message_types["atomfrag2atom"]:
                if self.attention:
                    self.atomfrag2atom.append(GATv2Conv(self.hidden_channels, self.hidden_channels, heads = 1, edge_dim = self.hidden_channels_edge + self.hidden_channels_frag, add_self_loops = False))
                else:
                    self.atomfrag2atom.append(InterMessage(self.hidden_channels + self.hidden_channels_edge + self.hidden_channels_frag, self.hidden_channels, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction)) #TODO: +1 seems inelegant
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
                if self.higher_level_edge_features == "aggregated":
                    assert(self.hidden_channels_edge == self.hidden_channels)
                    self.frag2frag.append(InterMessage(self.hidden_channels_frag + self.hidden_channels_edge, self.hidden_channels_frag, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction))
                elif self.higher_level_edge_features == "basic":
                    self.frag2frag.append(InterMessage(self.hidden_channels_frag + self.hidden_channels_edge, self.hidden_channels_frag, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction))
                else:
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
            size_message_atom = self.hidden_channels * self.message_types["atom2atom"] + self.hidden_channels * self.message_types["atomfrag2atom"] + self.hidden_channels_frag * self.message_types["frag2atom"] + self.hidden_channels_mol * self.message_types["mol2atom"]
            size_message_edge = self.hidden_channels * self.message_types["atom2edge"] + self.hidden_channels_frag * self.message_types["frag2edge"] + self.hidden_channels_mol * self.message_types["mol2edge"]
            size_message_frag = self.hidden_channels * self.message_types["atom2frag"] + self.hidden_channels_edge * self.message_types["edge2frag"] + self.hidden_channels_frag * self.message_types["frag2frag"] + self.hidden_channels_mol * self.message_types["mol2frag"] + self.concat
            size_message_mol = self.hidden_channels * self.message_types["atom2mol"] + self.hidden_channels_edge * self.message_types["edge2mol"] + self.hidden_channels_frag * self.message_types["frag2mol"]

            self.combine_atom_messages.append(MLP(in_channels = size_message_atom, out_channels = self.hidden_channels, num_layers = 1))
            self.combine_edge_messages.append(MLP(in_channels = size_message_edge, out_channels = self.hidden_channels_edge, num_layers = 1))
            self.combine_frag_messages.append(MLP(in_channels = size_message_frag, out_channels = self.hidden_channels_frag, num_layers = 1))
            self.combine_mol_messages.append(MLP(in_channels = size_message_mol, out_channels = self.hidden_channels_mol, num_layers = 1))

            self.atom_batch_norms.append(BatchNorm1d(self.hidden_channels))
            self.edge_batch_norms.append(BatchNorm1d(self.hidden_channels_edge))
            self.frag_batch_norms.append(BatchNorm1d(self.hidden_channels_frag))
            self.mol_batch_norms.append(BatchNorm1d(self.hidden_channels_mol))

            if self.process_after:
                self.process_after_atoms.append(MLP(in_channels = self.hidden_channels, out_channels = self.hidden_channels, num_layers = self.process_after))
                self.process_after_edges.append(MLP(in_channels = self.hidden_channels_edge, out_channels = self.hidden_channels_edge, num_layers = self.process_after))
                self.process_after_frags.append(MLP(in_channels=self.hidden_channels_frag, out_channels=self.hidden_channels_frag, num_layers=self.process_after))
        
        self.frag_out = MLP(self.hidden_channels_frag, self.hidden_channels, num_layers = 2)
        self.atom_out = MLP(self.hidden_channels, self.hidden_channels, num_layers = 2)
        self.edge_out = MLP(self.hidden_channels_edge, self.hidden_channels, num_layers = 2)
        self.mol_out = MLP(self.hidden_channels_mol, self.hidden_channels, num_layers = 2)
        self.out = MLP(self.hidden_channels, self.out_channels, num_layers = num_layers_out, batch_norm = False, last_relu = False)
    
    def set_default_message_types(self):
        defaults = {"atom2atom": True, "atom2edge": True, "atom2frag": True, "atom2mol": False, 
                    "edge2atom": False, "edge2edge": False, "edge2frag": True, "edge2mol" : False, 
                    "frag2atom": True, "frag2edge": False, "frag2frag": True, "frag2mol": False,
                    "mol2atom": False, "mol2edge": False, "mol2frag": False, "atomfrag2atom": False}
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
        
        if self.concat != 0:
            if self.ordinal_encoding:
                x_frag_const = self.frag_concat_encoder(data.fragment_types)
            else:
                x_frag_const = self.frag_concat_encoder(data.fragments)
        
        x_mol = torch.zeros((batch_size, self.hidden_channels_mol), device = x.device)
        
        for layer_ind in range(self.num_layers):

            # update atom representation
            atom_messages = []
            if self.message_types["atom2atom"]:
                if self.attention:
                    atom_messages.append(self.atom2atom[layer_ind](x, data.edge_index, x_edge))
                else:
                    atom_messages.append(self.atom2atom[layer_ind](torch.concat([x[row_edge], x_edge], dim = -1), col_edge, dim_size = x.size(0)))
                if self.residuals:
                    atom_messages[0] = atom_messages[0] + x
            if self.message_types["atomfrag2atom"]:
                edge_index, frag_index = data.low_high_edge_index
                frag_info = scatter(x_frag[frag_index], edge_index, reduce = self.reduction, dim = 0, dim_size = x_edge.size(0))
                if self.attention:
                    atom_messages.append(self.atomfrag2atom[layer_ind](x, data.edge_index, torch.concat([x_edge, frag_info], dim = -1)))
                else:
                    atom_messages.append(self.atomfrag2atom[layer_ind](torch.concat([x[row_edge], x_edge, frag_info], dim = -1), col_edge, dim_size = x.size(0)))
            if self.message_types["frag2atom"]:
                atom_messages.append(self.frag2atom[layer_ind](x_frag[col], row, dim_size=x.size(0)))
            if self.message_types["mol2atom"]:
                atom_messages.append(self.mol2atom[layer_ind](x_mol[data.batch], torch.arange(x.size(0), device = x.device, dtype = torch.int64), dim_size = x.size(0)))
            
            if atom_messages:
                x = x + self.combine_atom_messages[layer_ind](torch.concat(atom_messages, dim = -1))
                if self.process_after:
                    x = self.process_after_atoms[layer_ind](x)
                else:
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
                if self.process_after:
                    x_edge = self.process_after_edges[layer_ind](x_edge)
                else:
                    x_edge = self.edge_batch_norms[layer_ind](x_edge)
                    x_edge = F.relu(x_edge)
            

            # update frag representation
            frag_messages = []
            if self.message_types["atom2frag"]:
                frag_messages.append(self.atom2frag[layer_ind](x[row], col, dim_size = x_frag.size(0)))
            if self.message_types["edge2frag"]:
                edge_index, frag_index = data.low_high_edge_index
                frag_messages.append(self.edge2frag[layer_ind](x_edge[edge_index], frag_index, dim_size = x_frag.size(0)))
            if self.message_types["frag2frag"]:
                row_higher, col_higher = data.higher_edge_index
                number_of_higher_edges = row_higher.size(0)
                if self.higher_level_edge_features == "aggregated":
                    higher_edge_id, lower_edge_id = data.join_edge_index
                    lower_edge_info = scatter(x_edge[lower_edge_id], higher_edge_id, reduce = self.reduction, dim = 0, dim_size = number_of_higher_edges)
                    higher_edge_id, lower_node_id = data.join_node_index
                    lower_node_info = scatter(x[lower_node_id], higher_edge_id, reduce = self.reduction, dim = 0, dim_size = number_of_higher_edges)
                    frag_messages.append(self.frag2frag[layer_ind](torch.concat([x_frag[row_higher], lower_edge_info + lower_node_info], dim = -1), col_higher, dim_size = x_frag.size(0)))
                elif self.higher_level_edge_features == "basic":
                    higher_edge_features = self.higher_edge_encoder(data.higher_edge_types)
                    frag_messages.append(self.frag2frag[layer_ind](torch.concat([x_frag[row_higher], higher_edge_features], dim = -1), col_higher, dim_size = x_frag.size(0)))
                else:
                    frag_messages.append(self.frag2frag[layer_ind](x_frag[row_higher], col_higher, dim_size = x_frag.size(0)))
                
                if self.residuals:
                    frag_messages[-1] = frag_messages[-1] + x_frag
            if self.message_types["mol2frag"]:
                frag_messages.append(self.mol2frag[layer_ind](x_mol[data.fragments_batch], torch.arange(x_frag.size(0), device = x_frag.device, dtype = torch.int64), dim_size = x_frag.size(0)))
            if self.concat != 0:
                frag_messages.append(x_frag_const)
            
            if frag_messages:
                x_frag = x_frag + self.combine_frag_messages[layer_ind](torch.concat(frag_messages, dim = -1))
                if self.process_after:
                    x_frag = self.process_after_frags[layer_ind](x_frag)
                else:
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

        # x_mol = F.dropout(x_mol, self.dropout, training = self.training)
        # x_mol = self.mol_out(x_mol)
        # x = x + x_mol
        
        x = self.out(x)
        return x
    

class HLG_Old(torch.nn.Module):

    def __init__(self, in_channels, in_channels_edge, in_channels_frag,
          hidden_channels, hidden_channels_edge = None, hidden_channels_frag = None,
          out_channels = 1, num_layers = 3, num_layers_message_before = 0, 
          num_layers_message_after = 2, num_layers_out = 1,
          dropout = 0, ordinal_encoding = True, 
          reduction = "mean", message_types = {}):
        super(HLG_Old, self).__init__()
        self.in_channels = in_channels
        self.in_channels_edge = in_channels_edge
        self.in_channels_frag = in_channels_frag
        self.hidden_channels = hidden_channels
        self.hidden_channels_edge = hidden_channels_edge if hidden_channels_edge else hidden_channels
        self.hidden_channels_frag = hidden_channels_frag if hidden_channels_frag else hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.ordinal_encoding = ordinal_encoding
        self.reduction = reduction
        self.message_types = message_types.copy() # copy needed to be able to change dict
        self.set_default_message_types()

        #Encoders
        self.atom_encoder = AtomEncoder(self.hidden_channels)
        self.frag_encoder = FragEncoder(self.in_channels_frag, self.hidden_channels_frag, encoding_size_scaling=ordinal_encoding)
        self.edge_encoder = BondEncoder(self.hidden_channels_edge)

        #Model
        self.atom2atom = ModuleList() if self.message_types["atom2atom"] else None #includes edge2atom
        self.atom2edge = ModuleList() if self.message_types["atom2edge"] else None
        self.atom2frag = ModuleList() if self.message_types["atom2frag"] else None

        # edge2atom part or atom2atom
        if self.message_types["edge2edge"]:
            raise NotImplementedError
        self.edge2frag = ModuleList() if self.message_types["edge2frag"] else None

        self.frag2atom = ModuleList() if self.message_types["frag2atom"] else None
        if self.message_types["frag2edge"]:
            raise NotImplementedError
        self.frag2frag = ModuleList() if self.message_types["frag2frag"] else None
        
        self.combine_atom_messages = ModuleList()
        self.combine_edge_messages = ModuleList()
        self.combine_frag_messages = ModuleList()

        self.atom_batch_norms = ModuleList()
        self.edge_batch_norms = ModuleList()
        self.frag_batch_norms = ModuleList()

        for i in range(num_layers):
            if self.message_types["atom2atom"]:
                self.atom2atom.append(InterMessage(self.hidden_channels + self.hidden_channels_edge, self.hidden_channels, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction)) #TODO: +1 seems inelegant
            if self.message_types["atom2edge"]:
                self.atom2edge.append(InterMessage(self.hidden_channels, self.hidden_channels_edge, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["atom2frag"]:
                self.atom2frag.append(InterMessage(self.hidden_channels, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            
            if self.message_types["edge2frag"]:
                self.edge2frag.append(InterMessage(self.hidden_channels_edge, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            
            if self.message_types["frag2atom"]:
                self.frag2atom.append(InterMessage(self.hidden_channels_frag, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.message_types["frag2frag"]:
                self.frag2frag.append(InterMessage(self.hidden_channels_frag, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))

            #self.edge2atom.append(InterMessage(self.hidden_channels, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            size_message_node = self.hidden_channels * (self.message_types["atom2atom"] + self.message_types["frag2atom"])
            size_message_edge = self.hidden_channels_edge * (self.message_types["atom2edge"])
            size_message_frag = self.hidden_channels_frag * (self.message_types["atom2frag"] + self.message_types["edge2frag"] + self.message_types["frag2frag"])
            self.combine_atom_messages.append(MLP(in_channels = size_message_node, out_channels = self.hidden_channels, num_layers = 1))
            self.combine_edge_messages.append(MLP(in_channels = size_message_edge, out_channels = self.hidden_channels_edge, num_layers = 1))
            self.combine_frag_messages.append(MLP(in_channels = size_message_frag, out_channels = self.hidden_channels_frag, num_layers = 1))

            self.atom_batch_norms.append(BatchNorm1d(self.hidden_channels))
            self.edge_batch_norms.append(BatchNorm1d(self.hidden_channels_edge))
            self.frag_batch_norms.append(BatchNorm1d(self.hidden_channels_frag))
        
        self.frag_out = MLP(self.hidden_channels_frag, self.hidden_channels, num_layers = 2)
        self.atom_out = MLP(self.hidden_channels, self.hidden_channels, num_layers = 2)
        self.edge_out = MLP(self.hidden_channels_edge, self.hidden_channels, num_layers = 2)
        self.out = MLP(self.hidden_channels, self.out_channels, num_layers = num_layers_out, batch_norm = False, last_relu = False)
    
    def set_default_message_types(self):
        defaults = {"atom2atom": True, "atom2edge": True, "atom2frag": True, 
                    "edge2atom": True, "edge2edge": False, "edge2frag": True, 
                    "frag2atom": True, "frag2edge": False, "frag2frag": True}
        for msg_type,val in defaults.items():
            if msg_type not in self.message_types:
                self.message_types[msg_type] = val


    def forward(self, data):
        row, col = data.fragments_edge_index
        row_edge, col_edge = data.edge_index
        batch_size = torch.max(data.batch) + 1
        x = self.atom_encoder(data)
        x_edge = self.edge_encoder(data.edge_attr)
        
        if self.ordinal_encoding:
            x_frag = self.frag_encoder(data.fragment_types)
        else:
            x_frag = self.frag_encoder(data.fragments)
        
        for layer_ind in range(self.num_layers):

            # update atom representation
            atom_messages = []
            if self.message_types["atom2atom"]:
                atom_messages.append(self.atom2atom[layer_ind](torch.concat([x[row_edge], x_edge], dim = -1), col_edge, dim_size = x.size(0)))
            if self.message_types["frag2atom"]:
                atom_messages.append(self.frag2atom[layer_ind](x_frag[col], row, dim_size=x.size(0)))
            
            if atom_messages:
                x = x + self.combine_atom_messages[layer_ind](torch.concat(atom_messages, dim = -1))
                x = self.atom_batch_norms[layer_ind](x)
                x = F.relu(x)
            
            # update edge representation
            edge_messages = []
            if self.message_types["atom2edge"]:
                edge_messages.append(self.atom2edge[layer_ind](torch.concat([x[row_edge], x[col_edge]], dim = 0), torch.concat([torch.arange(row_edge.size(0), dtype=torch.int64, device = row_edge.device) for _ in range(2)], dim = 0), dim_size = row_edge.size(0)))
            
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
            
            if frag_messages:
                x_frag = x_frag + self.combine_frag_messages[layer_ind](torch.concat(frag_messages, dim = -1))
                x_frag = self.frag_batch_norms[layer_ind](x_frag)
                x_frag = F.relu(x_frag, inplace= False)

           
           
        
        x = scatter(x, data.batch, dim=0, reduce = self.reduction)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_out(x)

        edge_batch =  data.batch[data.edge_index[0]]
        x_edge = scatter(x_edge, edge_batch, dim = 0, dim_size = batch_size, reduce = self.reduction)
        x_edge = F.dropout(x_edge, self.dropout, training=self.training)
        x_edge = self.edge_out(x_edge)
        x = x + x_edge

        x_frag = scatter(x_frag, data.fragments_batch, dim=0, dim_size=batch_size,reduce=self.reduction)
        x_frag = F.dropout(x_frag, self.dropout, training=self.training)
        x_frag = self.frag_out(x_frag)
        x = x + x_frag
        
        x = self.out(x)
        return x

class HLG_HIMP(torch.nn.Module):

    def __init__(self, in_channels, in_channels_edge, in_channels_frag,
          hidden_channels, hidden_channels_edge = None, hidden_channels_frag = None,
          out_channels = 1, num_layers = 3, num_layers_message_before = 0, 
          num_layers_message_after = 2, num_layers_out = 1,
          dropout = 0, ordinal_encoding = True, 
          reduction = "mean", between_reduction = None,
          gin_conv = False,
          learned_edge_representation = False, just_sum = False,
          no_last_batch_norm = False,
          higher_level_edge_features: Optional[Literal["aggregated"]] = None, message_types = {}):
        super(HLG_HIMP, self).__init__()
        self.in_channels = in_channels
        self.in_channels_edge = in_channels_edge
        self.in_channels_frag = in_channels_frag
        self.hidden_channels = hidden_channels
        self.hidden_channels_edge = hidden_channels_edge if hidden_channels_edge else hidden_channels
        self.hidden_channels_frag = hidden_channels_frag if hidden_channels_frag else hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.ordinal_encoding = ordinal_encoding
        self.reduction = reduction
        self.between_reduction = between_reduction if between_reduction else reduction
        self.gin_conv = gin_conv
        self.learned_edge_representation = learned_edge_representation
        self.just_sum = just_sum
        self.no_last_batch_norm = no_last_batch_norm
        self.higher_level_edge_features = higher_level_edge_features
        self.message_types = message_types.copy() # copy needed to be able to change dict
        self.set_default_message_types()

        #Encoders
        self.atom_encoder = AtomEncoder(self.hidden_channels)
        self.frag_encoder = FragEncoder(self.in_channels_frag, self.hidden_channels_frag, encoding_size_scaling=ordinal_encoding)
        if self.learned_edge_representation:
            self.edge_encoder = BondEncoder(self.hidden_channels_edge)
        else:
            self.edge_encoder = ModuleList()
            for _ in range(num_layers):
                self.edge_encoder.append(BondEncoder(self.hidden_channels_edge))

        #Model
        self.atom2atom = ModuleList() if self.message_types["atom2atom"] else None #includes edge2atom
        self.atom2edge = ModuleList() if self.message_types["atom2edge"] else None
        self.atom2frag = ModuleList() if self.message_types["atom2frag"] else None

        # edge2atom part or atom2atom
        if self.message_types["edge2edge"]:
            raise NotImplementedError
        self.edge2frag = ModuleList() if self.message_types["edge2frag"] else None

        self.frag2atom = ModuleList() if self.message_types["frag2atom"] else None
        if self.message_types["frag2edge"]:
            raise NotImplementedError
        self.frag2frag = ModuleList() if self.message_types["frag2frag"] else None
        
        if not self.just_sum:
            self.combine_atom_messages = ModuleList()
            self.combine_edge_messages = ModuleList()
            self.combine_frag_messages = ModuleList()

        self.atom_batch_norms = ModuleList()
        self.edge_batch_norms = ModuleList()
        self.frag_batch_norms = ModuleList()

        for i in range(num_layers):
            if self.message_types["atom2atom"]:
                if gin_conv:
                    nn = Sequential(
                        Linear(hidden_channels, 2 * hidden_channels),
                        BatchNorm1d(2 * hidden_channels),
                        ReLU(),
                        Linear(2 * hidden_channels, hidden_channels),
                        )
                    self.atom2atom.append(GINEConv(nn, train_eps=True, edge_dim = self.hidden_channels_edge))
                else:
                    self.atom2atom.append(InterMessage(self.hidden_channels + self.hidden_channels_edge, self.hidden_channels, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction)) #TODO: +1 seems inelegant
            if self.message_types["atom2edge"]:
                self.atom2edge.append(InterMessage(self.hidden_channels, self.hidden_channels_edge, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.between_reduction))
            if self.message_types["atom2frag"]:
                self.atom2frag.append(InterMessage(self.hidden_channels, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.between_reduction))
            
            if self.message_types["edge2frag"]:
                self.edge2frag.append(InterMessage(self.hidden_channels_edge, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.between_reduction))
            
            if self.message_types["frag2atom"]:
                self.frag2atom.append(InterMessage(self.hidden_channels_frag, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.between_reduction))
            if self.message_types["frag2frag"]:
                if self.gin_conv:
                    nn = Sequential(
                        Linear(self.hidden_channels_frag, 2 * self.hidden_channels_frag),
                        BatchNorm1d(2 * self.hidden_channels_frag),
                        ReLU(),
                        Linear(2 * self.hidden_channels_frag, self.hidden_channels_frag),
                        )
                    if self.higher_level_edge_features:
                        raise NotImplementedError
                        self.frag2frag.append(GINEConv(nn, train_eps=True, edge_dim = self.hidden_channels_edge))
                    else:
                        self.frag2frag.append(GINConv(nn, train_eps=True))
                else:
                    self.frag2frag.append(InterMessage(self.hidden_channels_frag, self.hidden_channels_frag, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))

            #self.edge2atom.append(InterMessage(self.hidden_channels, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            size_message_node = self.hidden_channels * (self.message_types["atom2atom"] + self.message_types["frag2atom"])
            size_message_edge = self.hidden_channels_edge * (self.message_types["atom2edge"])
            size_message_frag = self.hidden_channels_frag * (self.message_types["atom2frag"] + self.message_types["edge2frag"] + self.message_types["frag2frag"])
            if not self.just_sum:
                self.combine_atom_messages.append(MLP(in_channels = size_message_node, out_channels = self.hidden_channels, num_layers = 1))
                self.combine_edge_messages.append(MLP(in_channels = size_message_edge, out_channels = self.hidden_channels_edge, num_layers = 1))
                self.combine_frag_messages.append(MLP(in_channels = size_message_frag, out_channels = self.hidden_channels_frag, num_layers = 1))

            self.atom_batch_norms.append(BatchNorm1d(self.hidden_channels))
            self.edge_batch_norms.append(BatchNorm1d(self.hidden_channels_edge))
            self.frag_batch_norms.append(BatchNorm1d(self.hidden_channels_frag))
        
        self.frag_out = MLP(self.hidden_channels_frag, self.hidden_channels, num_layers = 2)
        self.atom_out = MLP(self.hidden_channels, self.hidden_channels, num_layers = 2)
        if self.learned_edge_representation:
            self.edge_out = MLP(self.hidden_channels_edge, self.hidden_channels, num_layers = 2)
        self.out = MLP(self.hidden_channels, self.out_channels, num_layers = num_layers_out, batch_norm = False, last_relu = False)
    
    def set_default_message_types(self):
        defaults = {"atom2atom": True, "atom2edge": True, "atom2frag": True, 
                    "edge2atom": True, "edge2edge": False, "edge2frag": True, 
                    "frag2atom": True, "frag2edge": False, "frag2frag": True}
        for msg_type,val in defaults.items():
            if msg_type not in self.message_types:
                self.message_types[msg_type] = val


    def forward(self, data):
        row, col = data.fragments_edge_index
        row_edge, col_edge = data.edge_index
        batch_size = torch.max(data.batch) + 1
        x = self.atom_encoder(data)
        if self.learned_edge_representation:
            x_edge = self.edge_encoder(data.edge_attr)
        
        if self.ordinal_encoding:
            x_frag = self.frag_encoder(data.fragment_types)
        else:
            x_frag = self.frag_encoder(data.fragments)
        
        for layer_ind in range(self.num_layers):

            # update atom representation
            if not self.learned_edge_representation:
                x_edge = self.edge_encoder[layer_ind](data.edge_attr)

            atom_messages = []
            if self.message_types["atom2atom"]:
                if self.gin_conv:
                    atom_messages.append(self.atom2atom[layer_ind](x, data.edge_index, x_edge))
                else:
                    atom_messages.append(x + self.atom2atom[layer_ind](torch.concat([x[row_edge], x_edge], dim = -1), col_edge, dim_size = x.size(0)))
            
            if self.message_types["frag2atom"]:
                atom_messages.append(self.frag2atom[layer_ind](x_frag[col], row, dim_size=x.size(0)))
            
            if atom_messages:
                if self.just_sum:
                    x = sum(atom_messages)
                else:
                    x = x + self.combine_atom_messages[layer_ind](torch.concat(atom_messages, dim = -1)) #residual also in first message ?!
                if not self.no_last_batch_norm or layer_ind < self.num_layers - 1:
                    x = self.atom_batch_norms[layer_ind](x)
                    x = F.relu(x)
            
            # update edge representation
            edge_messages = []
            if self.message_types["atom2edge"]:
                edge_messages.append(self.atom2edge[layer_ind](torch.concat([x[row_edge], x[col_edge]], dim = 0), torch.concat([torch.arange(row_edge.size(0), dtype=torch.int64, device = row_edge.device) for _ in range(2)], dim = 0), dim_size = row_edge.size(0)))
            
            if edge_messages:
                if self.just_sum:
                    x_edge = x_edge + sum(edge_messages)
                else:
                    x_edge = x_edge + self.combine_edge_messages[layer_ind](torch.concat(edge_messages, dim = -1))
                if not self.no_last_batch_norm or layer_ind < self.num_layers - 1:
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
                if self.gin_conv:
                    if self.higher_level_edge_features:
                        raise NotImplementedError
                    else:
                        frag_messages.append(self.frag2frag[layer_ind](x_frag, data.higher_edge_index))
                else:
                    row_higher, col_higher = data.higher_edge_index
                    frag_messages.append(x_frag + self.frag2frag[layer_ind](x_frag[row_higher], col_higher, dim_size = x_frag.size(0)))
            
            if frag_messages:
                if self.just_sum:
                    x_frag = sum(frag_messages)
                else:
                    x_frag = x_frag + self.combine_frag_messages[layer_ind](torch.concat(frag_messages, dim = -1))
                if not self.no_last_batch_norm or layer_ind < self.num_layers - 1:
                    x_frag = self.frag_batch_norms[layer_ind](x_frag)
                    x_frag = F.relu(x_frag, inplace= False)

           
           
        
        x = scatter(x, data.batch, dim=0, reduce = self.reduction)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_out(x)

        if self.learned_edge_representation:
            edge_batch =  data.batch[data.edge_index[0]]
            x_edge = scatter(x_edge, edge_batch, dim = 0, dim_size = batch_size, reduce = self.reduction)
            x_edge = F.dropout(x_edge, self.dropout, training=self.training)
            x_edge = self.edge_out(x_edge)
            x = x + x_edge

        x_frag = scatter(x_frag, data.fragments_batch, dim=0, dim_size=batch_size,reduce=self.reduction)
        x_frag = F.dropout(x_frag, self.dropout, training=self.training)
        x_frag = self.frag_out(x_frag)
        x = x + x_frag
        
        x = self.out(x)
        return x