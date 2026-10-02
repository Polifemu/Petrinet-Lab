import pytest

from petrinet_lab.core.model import PetriNet
from petrinet_lab.core.reachability import (
    StateSpaceTooLarge,
    UnboundedNetError,
    build_reachability_graph,
)


def test_cycle_is_bounded_and_live(cycle):
    graph = build_reachability_graph(cycle)
    assert len(graph.states) == 2
    assert graph.is_bounded()
    assert graph.deadlock_markings() == []
    assert graph.dead_transitions() == []
    assert graph.liveness_level("start") == 4
    assert graph.occurrence_count("start") == 1


def test_deadlock_detection(deadlock):
    graph = build_reachability_graph(deadlock)
    deadlocks = graph.deadlock_markings()
    assert len(deadlocks) == 1
    assert deadlocks[0]["p3"] == 1
    assert graph.dead_transitions() == ["t2"]


def test_unbounded_net_is_detected():
    net = PetriNet("unbounded")
    net.add_place("p", initial=1)
    net.add_transition("grow")
    net.add_arc("p", "grow")
    net.add_arc("grow", "p", weight=2)
    with pytest.raises(UnboundedNetError):
        build_reachability_graph(net)


def test_state_limit():
    net = PetriNet("counter")
    net.add_place("p", initial=1)
    net.add_transition("inc")
    net.add_arc("p", "inc")
    net.add_arc("inc", "p")
    net.add_arc("inc", "p")
    with pytest.raises((UnboundedNetError, StateSpaceTooLarge)):
        build_reachability_graph(net, max_states=50)


def test_workflow_parallelism(workflow):
    graph = build_reachability_graph(workflow)
    assert graph.is_bounded()
    assert any(workflow.marking_from_key(state)["end"] == 1 for state in graph.states)
    assert len(graph.deadlock_markings()) == 1


def test_networkx_export(cycle):
    graph = build_reachability_graph(cycle)
    exported = graph.to_networkx()
    assert exported.number_of_nodes() == 2
    assert exported.number_of_edges() == 2
