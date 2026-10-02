"""Structural and behavioural comparison between Petri nets."""

from __future__ import annotations

from ..core.model import PetriNet
from .conformance import log_fitness, token_replay


def structural_diff(first: PetriNet, second: PetriNet, by: str = "id") -> dict[str, object]:
    if by not in {"id", "label"}:
        raise ValueError("by must be 'id' or 'label'")
    first_nodes = _node_signature(first, by)
    second_nodes = _node_signature(second, by)
    first_arcs = {(arc.source, arc.target, arc.weight) for arc in first.arcs}
    second_arcs = {(arc.source, arc.target, arc.weight) for arc in second.arcs}
    return {
        "only_in_first": sorted(first_nodes - second_nodes),
        "only_in_second": sorted(second_nodes - first_nodes),
        "common_nodes": sorted(first_nodes & second_nodes),
        "arcs_only_in_first": sorted(first_arcs - second_arcs),
        "arcs_only_in_second": sorted(second_arcs - first_arcs),
        "arcs_common": len(first_arcs & second_arcs),
    }


def fitness_of_model(net: PetriNet, traces, final_marking=None) -> dict[str, float]:
    return log_fitness(token_replay(net, traces, final_marking=final_marking))


def _node_signature(net: PetriNet, by: str = "id") -> set[tuple[str, str]]:
    signature: set[tuple[str, str]] = set()
    for place_id, place in net.places.items():
        key = place_id if by == "id" else (place.name or place_id)
        signature.add((f"place:{key}", place.name))
    for transition_id, transition in net.transitions.items():
        key = transition_id if by == "id" else transition.label
        signature.add((f"transition:{key}", transition.label))
    return signature
