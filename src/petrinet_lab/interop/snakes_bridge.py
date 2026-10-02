"""SNAKES interoperability used as an independent semantic oracle.

SNAKES (https://snakes.tech) provides its own P/T net semantics. We convert
our nets to SNAKES, explore the state space directly with SNAKES enabling and
firing, and compare the result with the native reachability graph. This
cross-validates the hand-written core against a mature, independent library.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

from ..core.model import PetriNet

_MARKERS = "BlackToken"


def _snakes_objects():
    from snakes.nets import BlackToken, Marking, MultiArc, Place, Transition, Value
    from snakes.nets import PetriNet as SnakesPetriNet

    return {
        "PetriNet": SnakesPetriNet,
        "Place": Place,
        "Transition": Transition,
        "MultiArc": MultiArc,
        "Value": Value,
        "BlackToken": BlackToken,
        "Marking": Marking,
    }


def to_snakes(net: PetriNet) -> Any:
    objects = _snakes_objects()
    snakes_net = objects["PetriNet"](net.name)
    for place_id, place in net.places.items():
        snakes_net.add_place(objects["Place"](place_id, [objects["BlackToken"]()] * place.initial))
    for transition_id in net.transition_ids:
        transition = objects["Transition"](transition_id)
        for place_id, weight in net.preset(transition_id).items():
            transition.add_input(
                snakes_net.place(place_id),
                objects["MultiArc"]([objects["Value"](objects["BlackToken"]())] * weight),
            )
        for place_id, weight in net.postset(transition_id).items():
            transition.add_output(
                snakes_net.place(place_id),
                objects["MultiArc"]([objects["Value"](objects["BlackToken"]())] * weight),
            )
        snakes_net.add_transition(transition)
    return snakes_net


def _annotation_weight(annotation: Any) -> int:
    if isinstance(annotation, list):
        return sum(_annotation_weight(item) for item in annotation)
    if hasattr(annotation, "__len__"):
        return len(annotation)
    return 1


def _arc_items(raw: Any):
    entries = raw if isinstance(raw, list) else [raw]
    for entry in entries:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            yield entry[0], entry[1]


def from_snakes(snakes_net: Any) -> PetriNet:
    marking = snakes_net.get_marking()
    net = PetriNet(snakes_net.name)
    for place in snakes_net.place():
        net.add_place(place.name, initial=len(marking.get(place.name, [])))
    for transition in snakes_net.transition():
        net.add_transition(transition.name)
        for place, annotation in _arc_items(transition.input()):
            net.add_arc(place.name, transition.name, _annotation_weight(annotation))
        for place, annotation in _arc_items(transition.output()):
            net.add_arc(transition.name, place.name, _annotation_weight(annotation))
    return net


def _snapshot(snakes_net: Any, place_order: list[str]) -> tuple[int, ...]:
    marking = snakes_net.get_marking()
    return tuple(len(marking.get(place_id, [])) for place_id in place_order)


def snakes_reachable_markings(
    net: PetriNet, max_states: int = 200_000
) -> tuple[set[tuple[int, ...]], set[tuple[tuple[int, ...], tuple[int, ...], str]]]:
    """BFS using SNAKES enabling/firing semantics (SNAKES has no bundled state plugin)."""
    objects = _snakes_objects()
    snakes_net = to_snakes(net)
    place_order = net.place_ids
    initial = _snapshot(snakes_net, place_order)
    seen = {initial}
    edges: set[tuple[tuple[int, ...], tuple[int, ...], str]] = set()
    queue = deque([initial])
    while queue:
        key = queue.popleft()
        snapshot = objects["Marking"](
            {place_id: [objects["BlackToken"]()] * key[i] for i, place_id in enumerate(place_order)}
        )
        for transition in snakes_net.transition():
            snakes_net.set_marking(copy.deepcopy(snapshot))
            for mode in transition.modes():
                transition.fire(mode)
                successor = _snapshot(snakes_net, place_order)
                edges.add((key, successor, transition.name))
                if successor not in seen:
                    if len(seen) >= max_states:
                        raise RuntimeError("SNAKES exploration exceeded max_states")
                    seen.add(successor)
                    queue.append(successor)
    return seen, edges


def compare_state_spaces(net: PetriNet, max_states: int = 200_000) -> dict[str, Any]:
    from ..core.reachability import build_reachability_graph

    native = build_reachability_graph(net, max_states=max_states)
    snakes_states, snakes_edges = snakes_reachable_markings(net, max_states)
    native_states = {tuple(state) for state in native.states}
    native_edges = {
        (tuple(native.states[edge.source]), tuple(native.states[edge.target]), edge.transition)
        for edge in native.edges
    }
    return {
        "model": net.name,
        "native_states": len(native_states),
        "snakes_states": len(snakes_states),
        "native_edges": len(native_edges),
        "snakes_edges": len(snakes_edges),
        "states_equal": native_states == snakes_states,
        "edges_equal": native_edges == snakes_edges,
    }
