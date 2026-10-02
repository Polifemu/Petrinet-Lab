"""Structure, invariants and reachability of a production cell."""

import matplotlib.pyplot as plt

from petrinet_lab.agent.tools import build_cyclic_cell
from petrinet_lab.core.invariants import structural_properties
from petrinet_lab.core.reachability import build_reachability_graph
from petrinet_lab.core.viz import draw_net, draw_reachability

net = build_cyclic_cell()
print("summary:", net.summary())
print("structural properties:", structural_properties(net))

graph = build_reachability_graph(net)
print("reachable markings:", [graph.marking(i) for i in range(len(graph.states))])
print("bounded:", graph.is_bounded())
print("liveness levels:", {t: graph.liveness_level(t) for t in net.transition_ids})

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
draw_net(net, ax=axes[0])
draw_reachability(graph, ax=axes[1])
figure.tight_layout()
figure.savefig("output/01_structure_reachability.png", dpi=150)
print("figure written to output/01_structure_reachability.png")
