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
          hidden_channels, hidden_channels_edge = None, hidden_channels_frag = None,
          out_channels = 1, num_layers = 3, num_layers_message_before = 0, num_layers_message_after = 2, 
          dropout = 0, ordinal_encoding = True, 
          inter_message_passing = True, higher_message_passing = True,
          reduction = "mean"):
        super(HLG, self).__init__()
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
        self.inter_message_passing = inter_message_passing
        self.higher_message_passing = higher_message_passing
        self.reduction = reduction

        #Encoders
        self.atom_encoder = AtomEncoder(self.hidden_channels)
        self.frag_encoder = FragEncoder(self.in_channels_frag, self.hidden_channels_frag, encoding_size_scaling=ordinal_encoding)

        #Model
        self.atom2atom = ModuleList()
        self.edge_encoders = ModuleList()
        #self.edge2atom = ModuleList()

        if self.inter_message_passing:
            self.atom2frag = ModuleList()
            self.frag2atom = ModuleList()
            self.combine_atom_messages = ModuleList()

        if self.higher_message_passing:
            self.frag2frag = ModuleList()
        
        if self.higher_message_passing and self.inter_message_passing:
            self.combine_frag_messages = ModuleList()

        for i in range(num_layers):
            self.edge_encoders.append(BondEncoder(self.hidden_channels_edge))
            self.atom2atom.append(InterMessage(self.hidden_channels + self.hidden_channels_edge, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            #self.edge2atom.append(InterMessage(self.hidden_channels, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
            if self.inter_message_passing:
                self.frag2atom.append(InterMessage(self.hidden_channels_frag, self.hidden_channels, num_layers_before = num_layers_message_before, num_layers_after = num_layers_message_after, reduction = self.reduction))
                self.atom2frag.append(InterMessage(self.hidden_channels, self.hidden_channels_frag, num_layers_before = num_layers_message_before + 1, num_layers_after = num_layers_message_after, reduction = self.reduction))
                self.combine_atom_messages.append(MLP(in_channels= 2* self.hidden_channels, out_channels = self.hidden_channels, num_layers = 1))
            if self.higher_message_passing:
                self.frag2frag.append(InterMessage(self.hidden_channels_frag, self.hidden_channels_frag, num_layers_before=num_layers_message_before, num_layers_after=num_layers_message_after, reduction= self.reduction))
            
            if self.higher_message_passing and self.inter_message_passing:
                self.combine_frag_messages.append(MLP(in_channels= 2 * self.hidden_channels_frag, out_channels = self.hidden_channels_frag, num_layers = 1))
        
        self.frag_out = MLP(self.hidden_channels_frag, self.hidden_channels, num_layers = 2)
        self.atom_out = MLP(self.hidden_channels, self.hidden_channels, num_layers = 2)
        self.out = MLP(self.hidden_channels, self.out_channels, num_layers =1, batch_norm = False, last_relu = False)
    
    def forward(self, data):
        row, col = data.fragments_edge_index
        batch_size = torch.max(data.batch) + 1
        x = self.atom_encoder(data)
        
        if self.ordinal_encoding:
            x_frag = self.frag_encoder(data.fragment_types)
        else:
            x_frag = self.frag_encoder(data.fragments)
        
        for layer_ind in range(self.num_layers):

            # update atom representation
            row_edge, col_edge = data.edge_index

            edge_attr = self.edge_encoders[layer_ind](data.edge_attr)
            #edge_node_attr = scatter(edge_attr, row_edge, dim = 0, dim_size = x.size(0), reduce = self.reduction)

            atom2atom_msg = self.atom2atom[layer_ind](torch.concat([x[row_edge], edge_attr], dim = -1), col_edge, dim_size = x.size(0))
            
            if self.inter_message_passing:
                frag2atom_msg = self.frag2atom[layer_ind](x_frag[col], row, dim_size=x.size(0))
                x += self.combine_atom_messages[layer_ind](torch.concat([atom2atom_msg, frag2atom_msg], dim = -1))
            else:
                x += atom2atom_msg

            #update frag representation
            if self.inter_message_passing:
                atom2frag_msg = self.atom2frag[layer_ind](x[row], col, dim_size = x_frag.size(0))
            
            if self.higher_message_passing:
                row_higher, col_higher = data.higher_edge_index
                frag2frag_msg = self.frag2frag[layer_ind](x_frag[row_higher], col_higher, dim_size = x_frag.size(0))
            
            if self.higher_message_passing and self.inter_message_passing:
                x_frag += self.combine_frag_messages[layer_ind](torch.concat([frag2frag_msg, atom2frag_msg], dim = -1))
            elif self.higher_message_passing:
                x_frag += frag2frag_msg
            elif self.inter_message_passing:
                x_frag += atom2frag_msg
        
        x = scatter(x, data.batch, dim=0, reduce = self.reduction)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.atom_out(x)

        if self.inter_message_passing or self.higher_message_passing:
            x_frag = scatter(x_frag, data.fragments_batch, dim=0, dim_size=batch_size,reduce=self.reduction)
            x_frag = F.dropout(x_frag, self.dropout, training=self.training)
            x_frag = self.frag_out(x_frag)
            x = x + x_frag
        
        x = self.out(x)
        return x