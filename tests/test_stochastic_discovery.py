import pytest

from petrinet_lab.core.simulation import (
    simulate_stochastic_event_log,
    traces_with_timestamps,
)
from petrinet_lab.core.stochastic import build_ctmc, stationary_distribution
from petrinet_lab.core.templates import machine, mm1k
from petrinet_lab.mining.stochastic_discovery import (
    compare_rates,
    discover_stochastic,
    estimate_rates,
)


def simulated_machine(seed: int = 42):
    net = machine().net
    rates = {"break": 1.0, "repair": 3.0}
    events = simulate_stochastic_event_log(net, rates, n_cases=500, max_time=60.0, seed=seed)
    return net, rates, events


def test_machine_rates_recovered():
    net, rates, events = simulated_machine()
    result = estimate_rates(net, events)
    assert result.unreplayable == 0
    assert result.fitness["replay_fitness"] == 1.0
    assert result.cases == 500
    comparison = compare_rates(result.rates, rates)
    assert comparison["mean_relative_error"] < 0.1
    assert comparison["max_relative_error"] < 0.15


def test_mm1k_rates_recovered():
    net = mm1k(capacity=3).net
    rates = {"arrive": 1.0, "serve": 2.0}
    events = simulate_stochastic_event_log(net, rates, n_cases=500, max_time=60.0, seed=7)
    result = estimate_rates(net, events)
    assert result.unreplayable == 0
    assert compare_rates(result.rates, rates)["max_relative_error"] < 0.15


def test_learned_rates_reproduce_stationary_distribution():
    net, rates, events = simulated_machine(seed=5)
    result = estimate_rates(net, events)
    ctmc = build_ctmc(net, result.rates)
    assert abs(stationary_distribution(ctmc)[0] - 0.75) < 0.03


def test_traces_path_matches_event_path():
    net, rates, events = simulated_machine(seed=9)
    from_events = estimate_rates(net, events)
    from_traces = estimate_rates(net, traces_with_timestamps(events))
    assert from_traces.rates == from_events.rates


def test_choice_probabilities_are_empirical_distributions():
    net, _, events = simulated_machine(seed=3)
    result = estimate_rates(net, events)
    probabilities = result.choice_probabilities()
    assert len(probabilities) == 2
    assert all(abs(sum(values.values()) - 1.0) < 1e-12 for values in probabilities.values())
    assert all(len(values) == 1 for values in probabilities.values())


def test_compare_rates_edge_cases():
    assert compare_rates({}, {"a": 1.0}) == {
        "per_transition": {},
        "mean_relative_error": None,
        "max_relative_error": None,
    }
    comparison = compare_rates({"a": 1.1, "b": 0.5}, {"a": 1.0, "b": 0.5})
    assert comparison["per_transition"]["b"] == 0.0
    assert abs(comparison["mean_relative_error"] - 0.05) < 1e-12


def test_dataframe_input():
    pandas = pytest.importorskip("pandas")
    from petrinet_lab.core.simulation import events_to_dataframe

    net, rates, events = simulated_machine(seed=13)
    dataframe = events_to_dataframe(events)
    assert isinstance(dataframe, pandas.DataFrame)
    result = estimate_rates(net, dataframe)
    assert compare_rates(result.rates, rates)["max_relative_error"] < 0.15


def test_pm4py_discovery_integration():
    pytest.importorskip("pm4py")
    from petrinet_lab.mining.discovery import events_to_log

    _, rates, events = simulated_machine(seed=11)
    log, _ = events_to_log(events)
    result = discover_stochastic(log, algorithm="inductive")
    assert result.algorithm == "inductive"
    assert result.unreplayable == 0
    learned = {result.net.transitions[tid].label: rate for tid, rate in result.rates.items()}
    assert set(learned) >= {"break", "repair"}
    assert compare_rates(learned, rates)["max_relative_error"] < 0.35
