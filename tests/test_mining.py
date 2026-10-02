import pytest

pytest.importorskip("pm4py")

from petrinet_lab.core import pnml
from petrinet_lab.core.model import PetriNet
from petrinet_lab.core.simulation import simulate_event_log, traces_from_events
from petrinet_lab.mining.compare import structural_diff
from petrinet_lab.mining.conformance import log_fitness, token_replay
from petrinet_lab.mining.discovery import discover, events_to_log, log_stats


def build_choice_net() -> PetriNet:
    net = PetriNet("choice")
    net.add_place("start", initial=1)
    net.add_place("p_mid")
    net.add_place("p_end")
    net.add_place("end")
    for transition_id, label in [("t_a", "a"), ("t_b", "b"), ("t_c", "c"), ("t_d", "d")]:
        net.add_transition(transition_id, name=label)
    net.add_arc("start", "t_a")
    net.add_arc("t_a", "p_mid")
    net.add_arc("p_mid", "t_b")
    net.add_arc("t_b", "p_end")
    net.add_arc("p_mid", "t_c")
    net.add_arc("t_c", "p_end")
    net.add_arc("p_end", "t_d")
    net.add_arc("t_d", "end")
    return net


def build_rework_net() -> PetriNet:
    net = PetriNet("rework")
    for place_id, initial in [("start", 1), ("checked", 0), ("rejected", 0), ("end", 0)]:
        net.add_place(place_id, initial=initial)
    for transition_id, label in [
        ("t_check", "Check"),
        ("t_reject", "Reject"),
        ("t_rework", "Rework"),
        ("t_close", "Close"),
    ]:
        net.add_transition(transition_id, name=label)
    net.add_arc("start", "t_check")
    net.add_arc("t_check", "checked")
    net.add_arc("checked", "t_reject")
    net.add_arc("t_reject", "rejected")
    net.add_arc("rejected", "t_rework")
    net.add_arc("t_rework", "checked")
    net.add_arc("rejected", "t_close")
    net.add_arc("t_close", "end")
    return net


@pytest.fixture(scope="module")
def simulated():
    net = build_choice_net()
    events = simulate_event_log(net, n_cases=200, seed=11)
    traces = traces_from_events(events)
    log, _ = events_to_log(events)
    return net, events, traces, log


def test_simulation_respects_language(simulated):
    _, events, traces, _ = simulated
    variants = {tuple(trace) for trace in traces}
    assert variants == {("a", "b", "d"), ("a", "c", "d")}
    assert len(events) == 3 * len(traces)


def test_log_stats(simulated):
    _, _, _, log = simulated
    stats = log_stats(log)
    assert stats["cases"] == 200
    assert stats["events"] == 600
    assert stats["activities"] == ["a", "b", "c", "d"]
    assert stats["variants"] == 2


def test_discovery_roundtrip(simulated):
    _net, _, traces, log = simulated
    result = discover(log, "inductive")
    assert len(result.net.places) > 0
    assert len(result.net.transitions) == 4
    assert {transition.label for transition in result.net.transitions.values()} == {
        "a",
        "b",
        "c",
        "d",
    }
    assert result.net.is_workflow_net()
    fitness = log_fitness(token_replay(result.net, traces, final_marking={"sink": 1}))
    assert fitness["log_fitness"] > 0.9


def test_token_replay_matches_pm4py(simulated):
    net, _, traces, log = simulated
    import tempfile
    from pathlib import Path

    import pm4py
    from pm4py.objects.petri_net.obj import Marking

    _, path = tempfile.mkstemp(suffix=".pnml")
    Path(path).write_text(pnml.to_pnml(net))
    pnet, pim, _ = pm4py.read_pnml(path)
    sink = next(place for place in pnet.places if str(place.name) == "end")
    pfm = Marking({sink: 1})
    reference = pm4py.fitness_token_based_replay(log, pnet, pim, pfm)
    ours = log_fitness(token_replay(net, traces, final_marking={"end": 1}))
    assert ours["log_fitness"] == pytest.approx(reference["log_fitness"], abs=0.01)
    Path(path).unlink(missing_ok=True)


def test_token_replay_with_silent_moves_matches_pm4py():
    import pm4py

    net = build_rework_net()
    events = simulate_event_log(net, n_cases=150, seed=3)
    traces = traces_from_events(events)
    log, _ = events_to_log(events)
    result = discover(log)
    ours = log_fitness(token_replay(result.net, traces, final_marking={"sink": 1}))
    reference = pm4py.fitness_token_based_replay(
        log, result.pm4py_net, result.pm4py_initial_marking, result.pm4py_final_marking
    )
    assert ours["log_fitness"] == pytest.approx(reference["log_fitness"], abs=0.05)


def test_structural_diff_by_label(simulated):
    net, _, _, log = simulated
    result = discover(log)
    diff = structural_diff(net, result.net, by="label")
    common = set(diff["common_nodes"])
    assert ("transition:a", "a") in common
    assert ("transition:d", "d") in common
