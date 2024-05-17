import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, Embedding, Linear, Module, ModuleList, ReLU,
                      Sequential)
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter


class AtomEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, degree_scaling=False, num_atom_types=100, num_atom_features=9):
        """
        Initialize the AtomEncoder class.

        Args:
            hidden_channels (int): The number of hidden/output channels for the embeddings.
            degree_scaling (bool, optional): Whether to apply degree scaling, i.e., scale some features by degree of the node. Defaults to False.
            num_atom_types (int, optional): The number of atom types. Defaults to 100.
            num_atom_features (int, optional): The number of atom features. Defaults to 9.
        """
        super(AtomEncoder, self).__init__()
        self.degree_scaling = degree_scaling
        self.hidden_channels = hidden_channels

        self.embeddings = torch.nn.ModuleList()

        for i in range(num_atom_features):
            self.embeddings.append(Embedding(num_atom_types, hidden_channels))

    def forward(self, graph, degree_info=None):
        """
        Encode atom attributes.

        Args:
            graph: The input graph.
            degree_info (Tensor, optional): Degree information tensor. Defaults to None.

        Returns:
            Tensor: The encoded atom attributes.
        """

        x = graph.x
        if x.dim() == 1:
            x = x.unsqueeze(1)

            out = 0
            for i in range(x.size(1)):
                out += self.embeddings[i](x[:, i])

            if self.degree_scaling:
                out[:, :self.hidden_channels//2] = torch.unsqueeze(
                    degree_info, dim=1) * out[:, :self.hidden_channels//2]

            return out


class BondEncoder(torch.nn.Module):
    """
    BondEncoder module for encoding bond attributes.

    Args:
        hidden_channels (int): Number of hidden channels.
        num_bond_types (int, optional): Number of bond types . Defaults to 6.
        num_bond_features (int, optional): Number of bond features. Defaults to 3.
    """

    def __init__(self, hidden_channels: int, num_bond_types: int = 6, num_bond_features: int = 3):
        super(BondEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(num_bond_features):
            self.embeddings.append(Embedding(num_bond_types, hidden_channels))

    def forward(self, bond_attr: torch.Tensor):
        """
        Encode bond attributes.

        Args:
            bond_attr (torch.Tensor): The input bond attributes of shape (num_bonds, num_bond_features) where each entry < num_bond_types.

        Returns:
            torch.Tensor: Embeddings for the bonds.
        """

        if bond_attr.dim() == 1:
            bond_attr = bond_attr.unsqueeze(1)

        out = 0
        for i in range(bond_attr.size(1)):
            out += self.embeddings[i](bond_attr[:, i])
        return out


class FragEncoder(torch.nn.Module):
    def __init__(self, in_channels_substructure: int, hidden_channels: int, ordinal_encoding: bool):
        """
        Initialize the FragEncoder module.

        Args:
            in_channels_substructure (int): Number of different substructure types (only needed for non ordinal encoding).
            hidden_channels (int): Number of hidden/output channels for the embedding.
            ordinal_encoding (bool): Flag indicating whether to use ordinal encoding or not.
        """
        super(FragEncoder, self).__init__()
        self.ordinal_encoding = ordinal_encoding
        self.hidden_channels = hidden_channels
        if not self.ordinal_encoding:
            self.embedding = Embedding(
                in_channels_substructure, hidden_channels)
        else:
            # embed paths, junction, ring with ordinal encoding
            self.embedding = Embedding(4, hidden_channels)

    def forward(self, frag_attr: torch.Tensor):
        """
        Encode fragments.

        Args:
            frag_attr (torch.Tensor): Input fragment attributes.
                If not self.ordinal_encoding: frag_attr are expected to be one hot encoded, i.e., have shape (num_fragments, in_channels_substructure).
                If self.ordinal_encoding: frag_attr are expected to have shape (num_fragments, 2) where the first column contains the fragment class (e.g., path, ring, or junction) and the second column contains the size (e.g., path length, junction size, ring size)

        Returns:
            torch.Tensor: Embeddings of the input fragment attributes.
        """
        if not self.ordinal_encoding:
            # frag_attr are in this case one hot encoded
            frag_attr = torch.argmax(frag_attr, dim=1)
            return self.embedding(frag_attr)
        else:
            assert (frag_attr.size(1) == 2)
            # frag_attr are
            embeddings = self.embedding(frag_attr[:, 0])
            embeddings[:, :self.hidden_channels//2] = torch.unsqueeze(
                frag_attr[:, 1], dim=1) * embeddings[:, :self.hidden_channels//2]
            return embeddings


class InterMessage(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, transform_scatter=False, reduction="mean"):
        """
        Initialize the InterMessage (for messages between different levels of the hierarchy) class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_layers (int, optional): Number of layers. Defaults to 1.
            transform_scatter (bool, optional): Whether to apply MLP transformation before scatter (or after). Defaults to False.
            reduction (str, optional): Reduction method for scatter. Defaults to "mean".
        """

        super(InterMessage, self).__init__()

        self.transform_scatter = transform_scatter
        self.reduction = reduction

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(Linear(in_channels, out_channels))
            else:
                layers.append(Linear(out_channels, out_channels))

            layers.append(ReLU())

        self.transform = Sequential(*layers)

    def forward(self, from_tensor, to_index, dim_size):
        """
        Performs the forward pass of the InterMessage.

        Args:
            from_tensor (torch.Tensor): The input tensor.
            to_index (torch.Tensor): The indices to scatter the tensor to.
            dim_size (int): The size of the dimension to scatter.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        if self.transform_scatter:
            # first transform then scatter
            message = self.transform(from_tensor)
            message = scatter(message, to_index, dim=0,
                              dim_size=dim_size, reduce=self.reduction)
        else:
            # first scatter then transform
            message = scatter(from_tensor, to_index, dim=0,
                              dim_size=dim_size, reduce=self.reduction)
            message = self.transform(message)
        return message


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of hidden layers.
        batch_norm (bool, optional): Whether to apply batch normalization after each linear layer. Default is True.
        last_relu (bool, optional): Whether to apply ReLU activation after the last linear layer. Default is True.
    """

    def __init__(self, in_channels, out_channels, num_layers, batch_norm=True, last_relu=True):
        super(MLP, self).__init__()
        hidden_sizes = np.linspace(
            start=in_channels, stop=out_channels, num=num_layers + 1, endpoint=True, dtype=int)
        layers = []
        for i in range(num_layers):
            in_channel = hidden_sizes[i]
            out_channel = hidden_sizes[i + 1]
            layers.append(Linear(in_channel, out_channel))
            if batch_norm:
                layers.append(BatchNorm1d(out_channel))
            if i != num_layers - 1 or last_relu:
                layers.append(ReLU())
        self.nn = Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=number_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim//2**l, input_dim//2**(l+1), bias=True) for l in range(L)]
        list_FC_layers.append(
            nn.Linear(input_dim//2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
