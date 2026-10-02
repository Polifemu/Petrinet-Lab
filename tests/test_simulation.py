from petrinet_lab.core.model import PetriNet
from petrinet_lab.core.simulation import (
    events_to_dataframe,
    simulate_event_log,
    traces_from_events,
)


def build_sequence() -> PetriNet:
    net = PetriNet("sequence")
    net.add_place("start", initial=1)
    net.add_place("p")
    net.add_place("end")
    net.add_transition("a", name="a")
    net.add_transition("b", name="b")
    net.add_arc("start", "a")
    net.add_arc("a", "p")
    net.add_arc("p", "b")
    net.add_arc("b", "end")
    return net


def test_simulate_event_log_stops_at_sink():
    events = simulate_event_log(build_sequence(), n_cases=3, seed=5)
    traces = traces_from_events(events)
    assert len(traces) == 3
    assert all(trace == ["a", "b"] for trace in traces)


def test_events_to_dataframe():
    events = simulate_event_log(build_sequence(), n_cases=2, seed=5)
    dataframe = events_to_dataframe(events)
    assert list(dataframe.columns) == ["case:concept:name", "concept:name", "time:timestamp"]
    assert len(dataframe) == 4
