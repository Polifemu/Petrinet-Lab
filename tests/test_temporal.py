from fractions import Fraction
from math import inf

import pytest

from petrinet_lab.core.model import PetriNet
from petrinet_lab.core.temporal import (
    TimePetriNet,
    build_state_class_graph,
    simulate_timed,
)


def build_single() -> PetriNet:
    net = PetriNet("single")
    net.add_place("p1", initial=1)
    net.add_place("p2")
    net.add_transition("t1")
    net.add_arc("p1", "t1")
    net.add_arc("t1", "p2")
    return net


def build_concurrent() -> PetriNet:
    net = PetriNet("concurrent")
    net.add_place("p", initial=2)
    net.add_place("a")
    net.add_place("b")
    net.add_transition("t1")
    net.add_transition("t2")
    net.add_arc("p", "t1")
    net.add_arc("t1", "a")
    net.add_arc("p", "t2")
    net.add_arc("t2", "b")
    return net


def build_timelock() -> PetriNet:
    net = PetriNet("timelock")
    net.add_place("p", initial=1)
    net.add_place("a")
    net.add_place("b")
    net.add_transition("fast")
    net.add_transition("slow")
    net.add_arc("p", "fast")
    net.add_arc("fast", "a")
    net.add_arc("p", "slow")
    net.add_arc("slow", "b")
    return net


def build_cell() -> PetriNet:
    net = PetriNet("cell")
    net.add_place("idle", initial=1)
    net.add_place("busy")
    net.add_transition("start")
    net.add_transition("finish")
    net.add_arc("idle", "start")
    net.add_arc("start", "busy")
    net.add_arc("busy", "finish")
    net.add_arc("finish", "idle")
    return net


def test_single_transition_window():
    tpn = TimePetriNet(build_single(), {"t1": (1, 3)})
    graph = build_state_class_graph(tpn)
    assert len(graph.classes) == 2
    assert graph.firing_time_bounds("t1") == (Fraction(1), Fraction(3))
    edge = graph.edges[0]
    assert (edge.min_delay, edge.max_delay) == (Fraction(1), Fraction(3))


def test_interval_validation():
    with pytest.raises(ValueError):
        TimePetriNet(build_single(), {"t1": (-1, 3)})
    with pytest.raises(ValueError):
        TimePetriNet(build_single(), {"t1": (5, 2)})


def test_concurrent_transitions_state_classes():
    tpn = TimePetriNet(build_concurrent(), {"t1": (1, 2), "t2": (2, 3)})
    graph = build_state_class_graph(tpn)
    assert len(graph.classes) == 6
    first = next(edge for edge in graph.edges if edge.source == 0 and edge.transition == "t2")
    assert (first.min_delay, first.max_delay) == (Fraction(2), Fraction(2))
    after_t1 = graph.successor_edges(
        next(edge.target for edge in graph.edges if edge.source == 0 and edge.transition == "t1")
    )
    assert {edge.transition for edge in after_t1} == {"t1", "t2"}


def test_time_lock_under_strong_semantics():
    tpn = TimePetriNet(build_timelock(), {"fast": (1, 2), "slow": (5, 10)})
    graph = build_state_class_graph(tpn)
    assert graph.firable_transitions(0) == ["fast"]
    assert graph.firing_time_bounds("slow") is None


def test_cyclic_cell_and_simulation():
    tpn = TimePetriNet(build_cell(), {"start": (2, 4), "finish": (3, 5)})
    graph = build_state_class_graph(tpn)
    assert graph.firing_time_bounds("start") == (Fraction(2), Fraction(4))
    assert graph.firing_time_bounds("finish") == (Fraction(3), Fraction(5))
    events = simulate_timed(tpn, steps=4, policy="earliest")
    assert [event.transition for event in events] == ["start", "finish", "start", "finish"]
    assert [event.time for event in events] == [
        Fraction(2),
        Fraction(5),
        Fraction(7),
        Fraction(10),
    ]


def test_unbounded_latest_interval():
    tpn = TimePetriNet(build_single(), {"t1": (1, inf)})
    graph = build_state_class_graph(tpn)
    assert graph.edges[0].max_delay == inf


def test_newly_enabled_clock_reset():
    net = PetriNet("sequence")
    net.add_place("p1", initial=1)
    net.add_place("p2")
    net.add_place("p3")
    net.add_transition("a")
    net.add_transition("b")
    net.add_arc("p1", "a")
    net.add_arc("a", "p2")
    net.add_arc("p2", "b")
    net.add_arc("b", "p3")
    tpn = TimePetriNet(net, {"a": (1, 1), "b": (2, 2)})
    events = simulate_timed(tpn, steps=2, policy="earliest")
    assert [event.transition for event in events] == ["a", "b"]
    assert [event.time for event in events] == [Fraction(1), Fraction(3)]
