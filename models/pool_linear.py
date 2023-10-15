from torch.nn import ModuleList, Module, Sequential, Linear, ReLU, BatchNorm1d, LazyLinear
from torch_scatter import scatter
import torch.nn.functional as F

class PoolLinear(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout = 0) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = ModuleList([])
        self.num_layers = num_layers
        self.dropout = dropout
        for layer_ind in range(num_layers):
            if layer_ind == 0:
                self.layers.append(Linear(in_features= in_channels, out_features = hidden_channels))
            elif layer_ind == num_layers -1:
                self.layers.append(Linear(in_features= hidden_channels, out_features = out_channels))
            else:
                self.layers.append(Linear(in_features= hidden_channels, out_features = hidden_channels))
    
    def forward(self, data):
        x = data.x
        x = scatter(x, data.batch, dim=0, reduce= "sum") 
        for ind, layer in enumerate(self.layers):
            x = layer(x)
            if ind != self.num_layers -1:
                x = F.relu(x)
        return x
    
class GlobalLinear(Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout = 0) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = ModuleList([])
        self.num_layers = num_layers
        self.dropout = dropout
        for layer_ind in range(num_layers):
            if layer_ind == 0:
                self.layers.append(LazyLinear(out_features = hidden_channels))
            elif layer_ind == num_layers -1:
                self.layers.append(Linear(in_features= hidden_channels, out_features = out_channels))
            else:
                self.layers.append(Linear(in_features= hidden_channels, out_features = hidden_channels))
    
    def forward(self, data):
        x = data.motif_counts
        for ind, layer in enumerate(self.layers):
            x = layer(x)
            if ind != self.num_layers -1:
                x = F.relu(x)
        return x