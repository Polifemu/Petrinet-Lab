import pytest

from petrinet_lab.core.model import PetriNet


def test_places_transitions_and_arcs():
    net = PetriNet("demo")
    net.add_place("p", initial=2)
    net.add_transition("t")
    net.add_arc("p", "t", weight=2)
    assert net.place_ids == ["p"]
    assert net.transition_ids == ["t"]
    assert net.preset("t") == {"p": 2}
    assert net.postset("t") == {}


def test_duplicate_and_invalid_definitions():
    net = PetriNet()
    net.add_place("p")
    net.add_transition("t")
    with pytest.raises(ValueError):
        net.add_place("p")
    with pytest.raises(ValueError):
        net.add_transition("t")
    with pytest.raises(KeyError):
        net.add_arc("nope", "t")
    with pytest.raises(ValueError):
        net.add_arc("p", "p")
    with pytest.raises(ValueError):
        net.add_arc("p", "t", weight=0)
    with pytest.raises(ValueError):
        PetriNet().add_place("bad", initial=-1)


def test_firing_semantics_with_weights():
    net = PetriNet()
    net.add_place("in", initial=3)
    net.add_place("out")
    net.add_transition("t")
    net.add_arc("in", "t", weight=2)
    net.add_arc("t", "out", weight=3)
    marked = net.initial_marking()
    assert net.enabled(marked, "t")
    after = net.fire(marked, "t")
    assert after == {"in": 1, "out": 3}
    assert not net.enabled(after, "t")
    assert net.try_fire(after, "t") is None


def test_marking_key_roundtrip(cycle):
    marking = {"ready": 0, "busy": 1}
    key = cycle.marking_key(marking)
    assert key == (0, 1)
    assert cycle.marking_from_key(key) == marking


def test_incidence_matrix(cycle):
    matrix = cycle.incidence_matrix()
    assert matrix == [[-1, 1], [1, -1]]


def test_workflow_net_detection(workflow, cycle):
    assert workflow.is_workflow_net()
    assert not cycle.is_workflow_net()
    assert workflow.summary()["workflow_net"] is True


def test_validate_isolated_nodes():
    net = PetriNet()
    net.add_place("p")
    net.add_transition("t")
    issues = net.validate()
    assert len(issues) == 2


def test_copy_and_dict_roundtrip(cycle):
    clone = cycle.copy()
    assert clone.to_dict() == cycle.to_dict()
    assert PetriNet.from_dict(cycle.to_dict()).to_dict() == cycle.to_dict()


def test_strongly_connected(cycle, workflow):
    assert cycle.strongly_connected()
    assert not workflow.strongly_connected()
