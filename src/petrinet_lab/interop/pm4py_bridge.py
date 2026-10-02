"""PM4Py interoperability: PNML conversion and independent reachability check.

PM4Py is already used for discovery and conformance; here it also provides an
independent transition-system construction to cross-validate the native
reachability graph.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ..core import pnml
from ..core.model import PetriNet


def to_pm4py(net: PetriNet, final_marking: dict[str, int] | None = None):
    import pm4py
    from pm4py.objects.petri_net.obj import Marking

    handle, path = tempfile.mkstemp(suffix=".pnml")
    os.close(handle)
    try:
        pnml.save_pnml(net, path)
        pm4py_net, initial_marking, _ = pm4py.read_pnml(path)
    finally:
        Path(path).unlink(missing_ok=True)
    if final_marking is None:
        sinks = [pid for pid in net.place_ids if not net.output_transitions(pid)]
        final_marking = {sinks[0]: 1} if len(sinks) == 1 else {}
    lookup = {str(place.name): place for place in pm4py_net.places}
    marking = Marking()
    for place_id, count in final_marking.items():
        if place_id in lookup and count:
            marking[lookup[place_id]] = count
    return pm4py_net, initial_marking, marking


def pm4py_state_space(net: PetriNet, final_marking: dict[str, int] | None = None):
    import pm4py

    pm4py_net, initial_marking, marking = to_pm4py(net, final_marking)
    transition_system = pm4py.convert_to_reachability_graph(pm4py_net, initial_marking, marking)
    return transition_system


def compare_state_spaces(net: PetriNet, final_marking: dict[str, int] | None = None) -> dict[str, Any]:
    from ..core.reachability import build_reachability_graph

    native = build_reachability_graph(net)
    transition_system = pm4py_state_space(net, final_marking)
    native_edges = len(native.edges)
    pm4py_edges = len(transition_system.transitions)
    pm4py_states = len(transition_system.states)
    return {
        "model": net.name,
        "native_states": len(native.states),
        "pm4py_states": pm4py_states,
        "native_edges": native_edges,
        "pm4py_edges": pm4py_edges,
        "states_equal": len(native.states) == pm4py_states,
        "edges_equal": native_edges == pm4py_edges,
    }
