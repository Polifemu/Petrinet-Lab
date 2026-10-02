import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from petrinet_lab.core.reachability import build_reachability_graph
from petrinet_lab.core.viz import draw_net, draw_reachability, net_to_networkx


def test_net_to_networkx(cycle):
    graph = net_to_networkx(cycle)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4


def test_draw_net_and_reachability(cycle):
    axis = draw_net(cycle)
    assert axis.get_title() == "cycle"
    plt.close("all")
    reachability = build_reachability_graph(cycle)
    axis = draw_reachability(reachability)
    assert axis.get_title() == "Reachability graph"
    plt.close("all")
