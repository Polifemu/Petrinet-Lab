"""Tool layer exposing the Petri net toolkit to an LLM agent and the dashboard.

All functions return JSON-serialisable dictionaries and import heavy optional
dependencies lazily, so the module can be used without an LLM provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import inf
from typing import Any

from ..core.invariants import structural_properties
from ..core.model import PetriNet
from ..core.reachability import (
    StateSpaceTooLarge,
    UnboundedNetError,
    build_reachability_graph,
)
from ..core.simulation import simulate_event_log, traces_from_events
from ..core.stochastic import (
    build_ctmc,
    mean_sojourn_times,
    stationary_distribution,
    throughputs,
)
from ..core.temporal import TimePetriNet, build_state_class_graph, simulate_timed


def build_parallel_workflow() -> PetriNet:
    net = PetriNet("parallel_workflow")
    net.add_place("start", initial=1)
    net.add_place("pa")
    net.add_place("pb")
    net.add_place("p1")
    net.add_place("p2")
    net.add_place("end")
    for transition_id, label in [("split", "Split"), ("a", "Task A"), ("b", "Task B"), ("join", "Join")]:
        net.add_transition(transition_id, name=label)
    net.add_arc("start", "split")
    net.add_arc("split", "pa")
    net.add_arc("split", "pb")
    net.add_arc("pa", "a")
    net.add_arc("a", "p1")
    net.add_arc("pb", "b")
    net.add_arc("b", "p2")
    net.add_arc("p1", "join")
    net.add_arc("p2", "join")
    net.add_arc("join", "end")
    return net


def build_cyclic_cell() -> PetriNet:
    net = PetriNet("cyclic_cell")
    net.add_place("idle", initial=1)
    net.add_place("busy")
    net.add_transition("start", name="Start")
    net.add_transition("finish", name="Finish")
    net.add_arc("idle", "start")
    net.add_arc("start", "busy")
    net.add_arc("busy", "finish")
    net.add_arc("finish", "idle")
    return net


def build_machine() -> PetriNet:
    net = PetriNet("machine")
    net.add_place("up", initial=1)
    net.add_place("down")
    net.add_transition("failure", name="Failure")
    net.add_transition("repair", name="Repair")
    net.add_arc("up", "failure")
    net.add_arc("failure", "down")
    net.add_arc("down", "repair")
    net.add_arc("repair", "up")
    return net


def build_timelock() -> PetriNet:
    net = PetriNet("timelock")
    net.add_place("p", initial=1)
    net.add_place("fast_done")
    net.add_place("slow_done")
    net.add_transition("fast", name="Fast")
    net.add_transition("slow", name="Slow")
    net.add_arc("p", "fast")
    net.add_arc("fast", "fast_done")
    net.add_arc("p", "slow")
    net.add_arc("slow", "slow_done")
    return net


MODELS: dict[str, Any] = {
    "parallel_workflow": {
        "builder": build_parallel_workflow,
        "description": "Workflow net with an AND-split and an AND-join.",
    },
    "cyclic_cell": {
        "builder": build_cyclic_cell,
        "description": "Cyclic production cell, idle/busy with timed start and finish.",
        "intervals": {"start": (2, 4), "finish": (3, 5)},
    },
    "machine": {
        "builder": build_machine,
        "description": "Two-state failure/repair machine for stochastic analysis.",
        "rates": {"failure": 1.0, "repair": 3.0},
    },
    "timelock": {
        "builder": build_timelock,
        "description": "Conflicting fast and slow tasks showing time-lock under strong semantics.",
        "intervals": {"fast": (1, 2), "slow": (5, 10)},
    },
}


def _number(value: Any) -> Any:
    if value == inf:
        return "inf"
    if isinstance(value, Fraction):
        return int(value) if value.denominator == 1 else float(value)
    return value


def list_models() -> list[dict[str, str]]:
    return [{"name": name, "description": data["description"]} for name, data in MODELS.items()]


def get_model(name: str) -> PetriNet:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name]["builder"]()


def describe_model(name: str) -> dict[str, Any]:
    net = get_model(name)
    return net.summary() | {"initial_marking": net.initial_marking()}


def structural_analysis(name: str) -> dict[str, Any]:
    net = get_model(name)
    return {"model": name, **structural_properties(net)}


def reachability_analysis(name: str, max_states: int = 100_000) -> dict[str, Any]:
    net = get_model(name)
    try:
        graph = build_reachability_graph(net, max_states=max_states)
    except (UnboundedNetError, StateSpaceTooLarge) as error:
        return {"model": name, "bounded": False, "error": str(error)}
    liveness = {}
    for transition_id in net.transition_ids:
        try:
            liveness[transition_id] = graph.liveness_level(transition_id)
        except StateSpaceTooLarge:
            liveness[transition_id] = None
    return {
        "model": name,
        "states": len(graph.states),
        "bounded": graph.is_bounded(),
        "token_bounds": graph.token_bounds(),
        "deadlocks": [graph.marking(i) for i in range(len(graph.states)) if not graph.successor_edges(i)],
        "dead_transitions": graph.dead_transitions(),
        "liveness_levels": liveness,
        "truncated": graph.truncated,
    }


def temporal_analysis(name: str, max_classes: int = 20_000) -> dict[str, Any]:
    net = get_model(name)
    intervals = MODELS[name].get("intervals", {})
    tpn = TimePetriNet(net, intervals)
    graph = build_state_class_graph(tpn, max_classes=max_classes)
    bounds = {
        transition_id: [
            _number(value) for value in graph.firing_time_bounds(transition_id)
        ]
        if graph.firing_time_bounds(transition_id)
        else None
        for transition_id in net.transition_ids
    }
    return {
        "model": name,
        "intervals": {
            transition_id: [_number(alpha), _number(beta)]
            for transition_id, (alpha, beta) in tpn.intervals.items()
        },
        "classes": len(graph.classes),
        "truncated": graph.truncated,
        "firing_time_bounds": bounds,
        "deadlock_classes": [cls.id for cls in graph.deadlock_classes()],
        "time_lock_classes": [cls.id for cls in graph.time_lock_classes()],
    }


def stochastic_analysis(name: str) -> dict[str, Any]:
    net = get_model(name)
    rates = MODELS[name].get("rates", {tid: 1.0 for tid in net.transition_ids})
    ctmc = build_ctmc(net, rates)
    try:
        distribution = stationary_distribution(ctmc)
        stationary = sorted(
            (
                {"marking": ctmc.marking(i), "probability": float(distribution[i])}
                for i in range(len(ctmc.states))
            ),
            key=lambda item: -item["probability"],
        )
        throughput = {
            transition_id: float(value)
            for transition_id, value in throughputs(ctmc, distribution).items()
        }
    except ValueError:
        distribution = None
        stationary = []
        throughput = {}
    return {
        "model": name,
        "rates": rates,
        "states": [ctmc.marking(i) for i in range(len(ctmc.states))],
        "stationary": stationary,
        "throughput": throughput,
        "mean_sojourn_times": {
            str(state): float(value) for state, value in mean_sojourn_times(ctmc).items()
        },
    }


def simulate_model(name: str, n_cases: int = 20, seed: int = 0) -> dict[str, Any]:
    net = get_model(name)
    events = simulate_event_log(net, n_cases=n_cases, seed=seed)
    traces = traces_from_events(events)
    return {
        "model": name,
        "events": [
            {"case": event.case_id, "activity": event.activity, "time": round(event.timestamp, 3)}
            for event in events[:50]
        ],
        "traces": traces[:20],
        "variants": sorted({tuple(trace) for trace in traces}),
    }


def timed_simulation(name: str, steps: int = 10, seed: int = 0) -> dict[str, Any]:
    net = get_model(name)
    intervals = MODELS[name].get("intervals", {})
    tpn = TimePetriNet(net, intervals)
    events = simulate_timed(tpn, steps=steps, seed=seed)
    return {
        "model": name,
        "events": [
            {"time": _number(event.time), "transition": event.transition, "marking": event.marking}
            for event in events
        ],
    }


TOOL_SPECS = [
    {
        "name": "list_models",
        "description": "List the available Petri net models with a short description.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "describe_model",
        "description": "Show structure (places, transitions, arcs, workflow-net flag) of a model.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "structural_analysis",
        "description": "Compute P/T invariants and conservative/repetitive structural properties.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "reachability_analysis",
        "description": "Build the reachability graph: states, boundedness, deadlocks, dead transitions, liveness levels.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "temporal_analysis",
        "description": "Analyse a Time Petri Net: state classes, firing-time bounds, time-locks.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "stochastic_analysis",
        "description": "Build the CTMC of a Stochastic Petri Net: stationary distribution, throughput, sojourn times.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "simulate_model",
        "description": "Simulate token-tagged event traces from a model (log-level simulation).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "n_cases": {"type": "integer"},
                "seed": {"type": "integer"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "timed_simulation",
        "description": "Simulate a timed execution trace of a Time Petri Net.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "steps": {"type": "integer"},
                "seed": {"type": "integer"},
            },
            "required": ["name"],
        },
    },
]


def dispatch(tool_name: str, arguments: Mapping[str, Any]) -> Any:
    functions = {
        "list_models": lambda: list_models(),
        "describe_model": lambda: describe_model(arguments["name"]),
        "structural_analysis": lambda: structural_analysis(arguments["name"]),
        "reachability_analysis": lambda: reachability_analysis(arguments["name"]),
        "temporal_analysis": lambda: temporal_analysis(arguments["name"]),
        "stochastic_analysis": lambda: stochastic_analysis(arguments["name"]),
        "simulate_model": lambda: simulate_model(
            arguments["name"],
            int(arguments.get("n_cases", 20)),
            int(arguments.get("seed", 0)),
        ),
        "timed_simulation": lambda: timed_simulation(
            arguments["name"],
            int(arguments.get("steps", 10)),
            int(arguments.get("seed", 0)),
        ),
    }
    if tool_name not in functions:
        return {"error": f"unknown tool {tool_name!r}", "available_tools": sorted(functions)}
    try:
        return functions[tool_name]()
    except (KeyError, ValueError) as error:
        return {"error": str(error), "available_models": sorted(MODELS)}
