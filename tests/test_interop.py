import pytest

from petrinet_lab.core.model import PetriNet


def build_weighted() -> PetriNet:
    net = PetriNet("weighted")
    net.add_place("p", initial=2)
    net.add_place("q")
    net.add_transition("t1")
    net.add_transition("t2")
    net.add_arc("p", "t1")
    net.add_arc("t1", "q")
    net.add_arc("q", "t2", weight=2)
    net.add_arc("t2", "p")
    return net


def test_snakes_roundtrip():
    pytest.importorskip("snakes")
    from petrinet_lab.interop.snakes_bridge import from_snakes, to_snakes

    net = build_weighted()
    restored = from_snakes(to_snakes(net))
    assert restored.to_dict() == net.to_dict()


def test_snakes_cross_validation(cycle, workflow):
    pytest.importorskip("snakes")
    from petrinet_lab.interop.snakes_bridge import compare_state_spaces

    for net in (cycle, workflow, build_weighted()):
        report = compare_state_spaces(net)
        assert report["states_equal"], report
        assert report["edges_equal"], report
        assert report["snakes_states"] == report["native_states"]


def test_pm4py_cross_validation(cycle, workflow):
    pytest.importorskip("pm4py")
    from petrinet_lab.interop.pm4py_bridge import compare_state_spaces

    for net in (cycle, workflow, build_weighted()):
        report = compare_state_spaces(net)
        assert report["states_equal"], report
        assert report["edges_equal"], report


def test_cross_validation_on_workflow_with_deadlock(deadlock):
    pytest.importorskip("snakes")
    from petrinet_lab.interop.snakes_bridge import compare_state_spaces as snakes_compare

    pytest.importorskip("pm4py")
    from petrinet_lab.interop.pm4py_bridge import compare_state_spaces as pm4py_compare

    assert snakes_compare(deadlock)["states_equal"]
    assert pm4py_compare(deadlock)["states_equal"]
