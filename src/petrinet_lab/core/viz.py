"""Matplotlib/NetworkX rendering of nets and reachability graphs (no Graphviz)."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import networkx as nx

from .model import PetriNet


def net_to_networkx(net: PetriNet):
    graph = nx.DiGraph()
    for place_id, place in net.places.items():
        graph.add_node(place_id, layer=0, kind="place", label=place.name or place_id)
    for transition_id, transition in net.transitions.items():
        graph.add_node(transition_id, layer=1, kind="transition", label=transition.label)
    for arc in net.arcs:
        graph.add_edge(arc.source, arc.target, weight=arc.weight)
    return graph


def draw_net(net: PetriNet, marking: Mapping[str, int] | None = None, ax=None, title: str | None = None):
    graph = net_to_networkx(net)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    if graph.number_of_nodes() == 0:
        ax.set_axis_off()
        return ax
    positions = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
    places = [n for n, data in graph.nodes(data=True) if data["kind"] == "place"]
    marking = marking if marking is not None else net.initial_marking()
    colors = ["#cfe8ff" if node in places else "#ffe0b3" for node in graph.nodes]
    shapes = ["o" if node in places else "s" for node in graph.nodes]
    sizes = [900 if node in places else 700 for node in graph.nodes]
    for node, shape, color, size in zip(graph.nodes, shapes, colors, sizes, strict=False):
        nx.draw_networkx_nodes(
            graph, positions, nodelist=[node], node_shape=shape, node_color=color,
            node_size=size, edgecolors="#333333", linewidths=1.2, ax=ax,
        )
    nx.draw_networkx_edges(
        graph, positions, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=14,
        edge_color="#555555", width=1.2, connectionstyle="arc3,rad=0.05",
    )
    labels = {node: graph.nodes[node]["label"] for node in graph.nodes}
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=ax)
    for place_id in places:
        tokens = marking.get(place_id, 0)
        if tokens:
            x, y = positions[place_id]
            ax.text(x, y - 0.08, "\u25cf" * min(tokens, 5), ha="center", va="top", fontsize=8)
    weights = {
        (source, target): data["weight"]
        for source, target, data in graph.edges(data=True)
        if data.get("weight", 1) != 1
    }
    if weights:
        nx.draw_networkx_edge_labels(
            graph, positions, edge_labels=weights, font_size=7, ax=ax, rotate=False
        )
    ax.set_title(title or net.name)
    ax.set_axis_off()
    return ax


def draw_reachability(graph, ax=None, title: str | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    exported = graph.to_networkx()
    if exported.number_of_nodes() == 0:
        ax.set_axis_off()
        return ax
    positions = nx.spring_layout(exported, seed=7)
    nx.draw_networkx_nodes(exported, positions, node_color="#d5f5e3", node_size=1200, ax=ax)
    nx.draw_networkx_edges(
        exported, positions, arrows=True, arrowstyle="-|>", arrowsize=12,
        edge_color="#555555", ax=ax,
    )
    labels = {
        node: "".join(str(value) for value in data["marking"].values())
        for node, data in exported.nodes(data=True)
    }
    nx.draw_networkx_labels(exported, positions, labels=labels, font_size=8, ax=ax)
    ax.set_title(title or "Reachability graph")
    ax.set_axis_off()
    return ax
