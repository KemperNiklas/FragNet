from torch_geometric.nn.norm import BatchNorm
from torch.nn import ModuleList, Module, Sequential, Linear, ReLU, BatchNorm1d, Embedding
import torch.nn.functional as F
from torch_scatter import scatter
import torch
import numpy as np

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

class FragEncoder(torch.nn.Module):

    def __init__(self, in_channels_frag, hidden_channels, encoding_size_scaling, rbf = 0):
        super(FragEncoder, self).__init__()
        self.encoding_size_scaling = encoding_size_scaling
        self.hidden_channels = hidden_channels
        self.rbf = rbf
        if not encoding_size_scaling:
            self.embedding = Embedding(in_channels_frag, hidden_channels)
        elif rbf == 0:
            self.embedding = Embedding(4, hidden_channels) #embed paths, junction, ring
        else:
            #rbf
            self.embedding = Embedding(4, hidden_channels//2)
            self.linears = ModuleList([])
            self.max_distances = [20, 20, 20, 20] #ring, path, junction, else
            for i in range(4):
                self.linears.append(Linear(rbf, hidden_channels - hidden_channels//2))
    
    def forward(self, frag_attr):
        
        if not self.encoding_size_scaling:
            # frag attr are currently one hot encoded 
            frag_attr = torch.argmax(frag_attr, dim = 1)
            return self.embedding(frag_attr)
        elif self.rbf == 0:
            assert(frag_attr.size(1) == 2)
            embeddings = self.embedding(frag_attr[:,0])
            embeddings[:,:self.hidden_channels//2] = torch.unsqueeze(frag_attr[:,1], dim = 1) * embeddings[:,:self.hidden_channels//2]
            return embeddings
        else:
            assert(frag_attr.size(1) == 2)
            shape_embeddings = self.embedding(frag_attr[:,0])
            hidden_channels_size = self.hidden_channels - self.hidden_channels//2
            size_embeddings = torch.zeros((frag_attr.size(0), hidden_channels_size), device = frag_attr.device)
            for i in range(4):
                mask = frag_attr[:,0] == i
                if any(mask):
                    size_embeddings[mask] = self.linears[i](get_gaussian_basis(torch.squeeze(frag_attr[:,1][mask]), self.rbf, max_dist = self.max_distances[i]))
            return torch.concat([shape_embeddings, size_embeddings], dim = 1)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm = True, last_relu = True):
        super(MLP, self).__init__()
        hidden_sizes = np.linspace(start = in_channels, stop = out_channels, num = num_layers +1, endpoint=True, dtype=int)
        layers = []
        for i in range(num_layers):
            in_channel = hidden_sizes[i]
            out_channel = hidden_sizes[i+1]
            layers.append(Linear(in_channel, out_channel))
            if batch_norm:
                layers.append(BatchNorm1d(out_channel))
            if i != num_layers -1 or last_relu:
                layers.append(ReLU())
        self.nn = Sequential(*layers)
    
    def forward(self, x):
        return self.nn(x)
    
class InterMessage(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers_before = 0, num_layers_after = 2, reduction = "mean"):
        super(InterMessage, self).__init__()

        self.reduction = reduction

        self.before = MLP(in_channels, in_channels, num_layers_before)
        self.after = MLP(in_channels, out_channels, num_layers_after)
    
    def forward(self, from_tensor, to_index, dim_size):
        message = self.before(from_tensor)
        message = scatter(message, to_index, dim = 0, dim_size =dim_size, reduce = self.reduction)
        message = self.after(message)
        return message
    
def get_gaussian_basis(dist, num_basis, max_dist=None):
    """Taken from https://github.com/TUM-DAML/synthetic_coordinates/blob/master/deepergcn_smp/icgnn/models/basis.py"""
    if max_dist is None:
        # the largest distance
        max_dist = torch.max(dist)

    # n equally spaced bins between 0 and max
    centers = torch.linspace(0, max_dist, num_basis, dtype=torch.float, device = dist.device)
    # the size of each bin
    std = centers[1] - centers[0]
    # insert a size "1" dimension
    return torch.exp(-0.5 * (dist.unsqueeze(-1) - centers.unsqueeze(0)) ** 2 / std ** 2)