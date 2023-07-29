from torch_geometric.nn import GCNConv
from torch.nn import ModuleList, Module, Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_scatter import scatter

class GCN(Module):
    def __init__(self, hidden_channels, out_channels, in_channels, num_layers, graph_level = False, pool_reduction = "sum"):
        super().__init__()
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction
        
        self.layers = ModuleList()
        for layer_ind in range(num_layers):
            if layer_ind == 0:
                self.layers.append(GCNConv(in_channels, hidden_channels))
            else:
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.out = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, out_channels),
                )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for layer_ind, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if layer_ind != len(self.layers) -1:
                x = F.dropout(x, p=0.5, training=self.training)
                x = x.relu()
        
        if self.graph_level:
            # Pooling 
            x = scatter(x, data.batch, dim=0, reduce= self.pool_reduction)
        x = self.out(x)
        return x