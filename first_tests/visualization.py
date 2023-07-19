from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import networkx as nx

def visualize(graph, node_color = None, substructure = [], colorbar_label = None, **kwargs):
    """Visualize graph.

    Parameters
    ----------
    graph
        Pytorch-geometric graph
    node_color, optional
        Value indicating the color for each node 
    substructure, optional
        Substructure to be colored
    colorbar_label, optional
        Label for the colorbar
    """
    g = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(g)

    if node_color is not None:
        if "vmin" not in kwargs:
            kwargs["vmin"] = 0
        if "vmax" not in kwargs:
            kwargs["vmax"] = max(node_color)
        if "cmap" not in kwargs:
            kwargs["cmap"] = "viridis"
    if "node_size" not in kwargs:
        kwargs["node_size"] = 25
    
    nx.draw_networkx(g, pos = pos, with_labels=False, node_color=node_color, **kwargs)
    if node_color is not None:
        norm = colors.Normalize(0, kwargs["vmax"])
        cmap = cm.get_cmap(kwargs["cmap"])
        plt.gcf().colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label = colorbar_label)

    for sub in substructure:
        t = plt.Polygon([pos[node] for node in sub], facecolor = "red", alpha = 0.2)
        plt.gca().add_patch(t)

    plt.show()