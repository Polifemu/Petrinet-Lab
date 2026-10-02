import numpy as np
import pytest

from petrinet_lab.core.model import PetriNet
from petrinet_lab.core.stochastic import (
    build_ctmc,
    long_run_distribution,
    mean_sojourn_times,
    simulate_ssa,
    stationary_distribution,
    throughputs,
    transient_distribution,
)


def build_machine() -> PetriNet:
    net = PetriNet("machine")
    net.add_place("up", initial=1)
    net.add_place("down")
    net.add_transition("break")
    net.add_transition("repair")
    net.add_arc("up", "break")
    net.add_arc("break", "down")
    net.add_arc("down", "repair")
    net.add_arc("repair", "up")
    return net


def test_two_state_machine_stationary():
    ctmc = build_ctmc(build_machine(), {"break": 1.0, "repair": 3.0})
    pi = stationary_distribution(ctmc)
    assert np.allclose(pi, [0.75, 0.25])
    assert np.allclose(ctmc.exit_rates, [1.0, 3.0])
    assert np.allclose(throughputs(ctmc, pi)["break"], 0.75)
    assert np.allclose(throughputs(ctmc, pi)["repair"], 0.75)


def test_transient_two_state_analytic():
    ctmc = build_ctmc(build_machine(), {"break": 1.0, "repair": 3.0})
    initial = transient_distribution(ctmc, 0.0)
    assert np.allclose(initial, [1.0, 0.0])
    time = 0.5
    expected_up = 0.75 + 0.25 * np.exp(-4.0 * time)
    assert np.allclose(transient_distribution(ctmc, time)[0], expected_up, atol=1e-9)


def test_mmk_queue_stationary():
    net = PetriNet("mm1k")
    net.add_place("slots", initial=3)
    net.add_place("jobs")
    net.add_transition("arrive")
    net.add_transition("serve")
    net.add_arc("slots", "arrive")
    net.add_arc("arrive", "jobs")
    net.add_arc("jobs", "serve")
    net.add_arc("serve", "slots")
    arrivals, service = 1.0, 2.0
    ctmc = build_ctmc(net, {"arrive": arrivals, "serve": service})
    pi = stationary_distribution(ctmc)
    rho = arrivals / service
    expected = np.array([(1 - rho) * rho**k / (1 - rho ** 4) for k in range(4)])
    assert np.allclose(pi, expected)


def test_ssa_matches_stationary_distribution():
    events = simulate_ssa(build_machine(), {"break": 1.0, "repair": 3.0}, time_end=20_000, seed=7)
    assert len(events) > 5_000
    occupied = np.zeros(2)
    previous = 0.0
    state = 0
    for event in events:
        occupied[state] += event.time - previous
        previous = event.time
        state = 1 - state
    fraction_up = occupied[0] / previous
    assert abs(fraction_up - 0.75) < 0.05


def test_self_loop_has_no_generator_effect():
    net = PetriNet("selfloop")
    net.add_place("p", initial=1)
    net.add_transition("tick")
    net.add_arc("p", "tick")
    net.add_arc("tick", "p")
    ctmc = build_ctmc(net, {"tick": 5.0})
    assert len(ctmc.states) == 1
    assert np.allclose(ctmc.generator, [[0.0]])
    assert mean_sojourn_times(ctmc)[0] == float("inf")


def test_marking_dependent_rate():
    ctmc = build_ctmc(
        build_machine(),
        {"break": 1.0, "repair": lambda marking: 3.0 * marking["down"]},
    )
    pi = stationary_distribution(ctmc)
    assert np.allclose(pi, [0.75, 0.25])


def test_negative_rate_rejected():
    with pytest.raises(ValueError):
        build_ctmc(build_machine(), {"break": -1.0, "repair": 1.0})


def test_long_run_distribution_fallback():
    ctmc = build_ctmc(build_machine(), {"break": 1.0, "repair": 3.0})
    pi = long_run_distribution(ctmc)
    assert np.allclose(pi, [0.75, 0.25], atol=1e-6)
