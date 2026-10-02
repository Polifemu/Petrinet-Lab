from petrinet_lab.core.invariants import (
    incidence_matrix,
    is_conservative,
    is_repetitive,
    null_space,
    p_invariants,
    structural_properties,
    t_invariants,
)
from petrinet_lab.core.model import PetriNet


def test_cycle_invariants(cycle):
    assert p_invariants(cycle) == [{"ready": 1, "busy": 1}]
    assert t_invariants(cycle) == [{"start": 1, "end": 1}]
    assert is_conservative(cycle)
    assert is_repetitive(cycle)


def test_weighted_invariants():
    net = PetriNet("weighted")
    net.add_place("p1")
    net.add_place("p2")
    net.add_transition("t1")
    net.add_transition("t2")
    net.add_arc("p1", "t1", weight=2)
    net.add_arc("t1", "p2", weight=2)
    net.add_arc("p2", "t2", weight=2)
    net.add_arc("t2", "p1", weight=2)
    assert p_invariants(net) == [{"p1": 1, "p2": 1}]
    assert t_invariants(net) == [{"t1": 1, "t2": 1}]


def test_non_conservative_net():
    net = PetriNet("grower")
    net.add_place("p", initial=1)
    net.add_transition("t")
    net.add_arc("p", "t")
    net.add_arc("t", "p", weight=2)
    assert p_invariants(net) == []
    assert not is_conservative(net)
    assert t_invariants(net) == []


def test_null_space_of_matrix():
    matrix = [[1, 2], [2, 4]]
    basis = null_space(matrix)
    assert len(basis) == 1
    for row in matrix:
        assert sum(a * b for a, b in zip(row, basis[0], strict=False)) == 0


def test_structural_properties_keys(cycle):
    properties = structural_properties(cycle)
    assert properties["conservative"] is True
    assert properties["structurally_bounded"] is True
    assert properties["repetitive"] is True
    assert len(incidence_matrix(cycle)) == 2
