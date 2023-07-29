import torch
from torch.nn import Sequential, Linear, ReLU, Module
from torch_geometric.nn import MessagePassing, GINConv
from torch.nn import Embedding, ModuleList, ParameterList
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_scatter import scatter
import torch.nn.functional as F

class SubstructureLayer(Module):
    def __init__(self, message_neighbor, node2substructures, substructures2node, reduction = "sum"):
        super().__init__()
        self.message_neighbor = message_neighbor 
        self.node2substructures = node2substructures
        self.substructures2node = substructures2node
        self.reduction = reduction
    
    def reset_parameters(self):
        self.message_neighbor.reset_parameters()
        self.node2substructures.reset_parameters()
        self.substructures2node.reset_parameters()


    def forward(self, x, neighbor_edge_index, substructures_edge_index):
        x = self.message_neighbor(x, neighbor_edge_index)
        
        #message from nodes to substructure
        for substructure_edge_index, node2substructure, substructure2node in zip(substructures_edge_index, self.node2substructures, self.substructures2node):
            row, col = substructure_edge_index
            substructure_x = scatter(x[row], col, reduce = self.reduction, dim = 0)
            #substructure transform
            substructure_x = node2substructure(substructure_x)

            #message from substructure to nodes
            substructure_message = substructure2node(scatter(
                substructure_x[col], row, dim = 0, dim_size = x.size(0), reduce = self.reduction
            ))
            x = x + substructure_message
        return x
    


class SubstructureNeuralNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, in_channels, num_layers, num_substructures, reduction = "sum", graph_level = False, pool_reduction = "sum"):
        
        super(SubstructureNeuralNet, self).__init__()
        self.num_layers = num_layers
        self.graph_level = graph_level
        self.pool_reduction = pool_reduction

        self.layers = ModuleList()

        for i in range(num_layers):
            if i==0:
                message_nn = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
                )
            else:
                message_nn = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                )

            node2substructures = ModuleList()
            substructures2node = ModuleList()
            

            for substructure in range(num_substructures):

                # node2substructure = Sequential(
                #     Linear(hidden_channels, 2 * hidden_channels),
                #     BatchNorm1d(2 * hidden_channels),
                #     ReLU(),
                #     Linear(2 * hidden_channels, hidden_channels),
                # )
                node2substructure = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                )
                node2substructures.append(node2substructure)

                # substructure2node = Sequential(
                #     Linear(hidden_channels, 2 * hidden_channels),
                #     BatchNorm1d(2 * hidden_channels),
                #     ReLU(),
                #     Linear(2 * hidden_channels, hidden_channels),
                # )
                substructure2node = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                )
                substructures2node.append(substructure2node)

            self.layers.append(SubstructureLayer(GINConv(message_nn, train_eps=True), node2substructures, substructures2node, reduction = reduction))         

        self.out = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, out_channels),
                )

    def reset_parameters(self):

        for layer in self.layers:
            layer.reset_parameters()

        self.out.reset_parameters()

    def forward(self, data):
        x = data.x

        for i in range(self.num_layers):

            x = self.layers[i](x=x, neighbor_edge_index=data.edge_index, substructures_edge_index = data.substructures_edge_index)

        if self.graph_level:
            # Pooling 
            x = scatter(x, data.batch, dim=0, reduce= self.pool_reduction)

        x = self.out(x)
        return x
    