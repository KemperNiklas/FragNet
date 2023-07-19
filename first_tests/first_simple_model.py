import torch
import torch.nn.functional as F
from torch.nn import Embedding, ModuleList
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
#from torch_scatter import scatter
from torch_geometric.nn import GINConv, GINEConv

class SimpleGraphNeuralNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        
        super(Net, self).__init__()
        self.num_layers = num_layers

        self.node_convs = ModuleList()

        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            self.node_convs.append(GINConv(nn, train_eps=True))

        self.lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):

        for conv in self.node_convs:
            conv.reset_parameters()

        self.lin.reset_parameters()

    def forward(self, data):
        x = data.x

        for i in range(self.num_layers):

            x = self.node_convs[i](x=x, edge_index=data.edge_index)


        #x = scatter(x, data.batch, dim=0, reduce='mean')
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)


        #x = F.relu(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.lin(x)
        return x