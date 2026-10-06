import numpy as np
import pytest

from petrinet_lab.core.gspn import (
    ImmediateLoopError,
    StochasticPetriNet,
    build_gspn_ctmc,
    immediate_throughputs,
    resolve_marking,
    simulate_gspn_ssa,
)
from petrinet_lab.core.stochastic import build_ctmc, stationary_distribution, throughputs
from petrinet_lab.core.templates import machine, mmc


def build_routing(weight_left: float = 3.0, weight_right: float = 1.0) -> StochasticPetriNet:
    gspn = StochasticPetriNet("routing")
    gspn.add_place("idle", initial=1)
    gspn.add_place("choice")
    gspn.add_place("left")
    gspn.add_place("right")
    gspn.add_timed_transition("tick", 1.0)
    gspn.add_immediate_transition("go_left", weight=weight_left)
    gspn.add_immediate_transition("go_right", weight=weight_right)
    gspn.add_timed_transition("back_left", 1.0)
    gspn.add_timed_transition("back_right", 1.0)
    gspn.add_arc("idle", "tick")
    gspn.add_arc("tick", "choice")
    gspn.add_arc("choice", "go_left")
    gspn.add_arc("go_left", "left")
    gspn.add_arc("choice", "go_right")
    gspn.add_arc("go_right", "right")
    gspn.add_arc("left", "back_left")
    gspn.add_arc("back_left", "idle")
    gspn.add_arc("right", "back_right")
    gspn.add_arc("back_right", "idle")
    return gspn


def test_timed_only_matches_plain_ctmc():
    gspn = machine(break_rate=1.0, repair_rate=3.0)
    gspn_ctmc = build_gspn_ctmc(gspn)
    plain = build_ctmc(gspn.net, {"break": 1.0, "repair": 3.0})
    assert gspn_ctmc.states == plain.states
    assert np.allclose(gspn_ctmc.generator, plain.generator)
    assert np.allclose(stationary_distribution(gspn_ctmc), [0.75, 0.25])


def test_immediate_weights_set_routing_probabilities():
    ctmc = build_gspn_ctmc(build_routing(weight_left=3.0, weight_right=1.0))
    pi = stationary_distribution(ctmc)
    assert len(ctmc.states) == 3
    assert np.allclose(pi, [0.5, 0.375, 0.125])
    flows = immediate_throughputs(ctmc, pi)
    assert np.isclose(flows["go_left"] / flows["go_right"], 3.0)
    assert np.isclose(flows["go_left"], 0.375)


def test_immediate_priority_preempts_lower_priority():
    gspn = StochasticPetriNet("priority")
    gspn.add_place("p", initial=1)
    gspn.add_place("fast_done")
    gspn.add_place("slow_done")
    gspn.add_immediate_transition("fast", weight=1.0, priority=0)
    gspn.add_immediate_transition("slow", weight=1000.0, priority=1)
    gspn.add_arc("p", "fast")
    gspn.add_arc("fast", "fast_done")
    gspn.add_arc("p", "slow")
    gspn.add_arc("slow", "slow_done")
    resolution = resolve_marking(gspn, {"p": 1, "fast_done": 0, "slow_done": 0})
    assert resolution.targets == {(0, 1, 0): 1.0}
    assert resolution.expected_firings == {"fast": 1.0}


def test_guard_restricts_enabling():
    gspn = StochasticPetriNet("guarded")
    gspn.add_place("p", initial=1)
    gspn.add_place("gate")
    gspn.add_place("q")
    gspn.add_immediate_transition("move")
    gspn.add_arc("p", "move")
    gspn.add_arc("move", "q")
    gspn.set_guard("move", lambda marking: marking["gate"] > 0)
    blocked = resolve_marking(gspn, {"p": 1, "gate": 0, "q": 0})
    assert blocked.targets == {(1, 0, 0): 1.0}
    allowed = resolve_marking(gspn, {"p": 1, "gate": 1, "q": 0})
    assert allowed.targets == {(0, 1, 1): 1.0}


def test_probabilistic_rework_resolution():
    gspn = StochasticPetriNet("rework")
    gspn.add_place("ready")
    gspn.add_place("busy", initial=1)
    gspn.add_place("done")
    gspn.add_immediate_transition("ok", weight=3.0)
    gspn.add_immediate_transition("rework", weight=1.0)
    gspn.add_arc("busy", "ok")
    gspn.add_arc("ok", "done")
    gspn.add_arc("busy", "rework")
    gspn.add_arc("rework", "ready")
    resolution = resolve_marking(gspn, {"ready": 0, "busy": 1, "done": 0})
    assert np.isclose(resolution.targets[(0, 0, 1)], 0.75)
    assert np.isclose(resolution.targets[(1, 0, 0)], 0.25)
    assert np.isclose(resolution.expected_firings["ok"], 0.75)
    assert np.isclose(resolution.expected_firings["rework"], 0.25)


def test_closed_immediate_loop_raises():
    gspn = StochasticPetriNet("loop")
    gspn.add_place("p", initial=1)
    gspn.add_immediate_transition("spin")
    gspn.add_arc("p", "spin")
    gspn.add_arc("spin", "p")
    with pytest.raises(ImmediateLoopError):
        resolve_marking(gspn, {"p": 1})
    with pytest.raises(ImmediateLoopError):
        simulate_gspn_ssa(gspn, max_immediate_steps=100)


def test_immediate_throughput_conservation():
    gspn = mmc(servers=2, buffer=1, arrival_rate=1.5, service_rate=1.0)
    ctmc = build_gspn_ctmc(gspn)
    distribution = stationary_distribution(ctmc)
    timeds = throughputs(ctmc, distribution)
    immediates = immediate_throughputs(ctmc, distribution)
    assert np.isclose(immediates["start"], timeds["arrive"])
    assert np.isclose(timeds["arrive"], timeds["finish"])


def test_marking_dependent_timed_rate():
    gspn = StochasticPetriNet("batch")
    gspn.add_place("pending", initial=3)
    gspn.add_place("done")
    gspn.add_timed_transition("process", lambda marking: 2.0 * marking["pending"])
    gspn.add_arc("pending", "process")
    gspn.add_arc("process", "done")
    ctmc = build_gspn_ctmc(gspn)
    assert [ctmc.marking(i)["pending"] for i in range(len(ctmc.states))] == [3, 2, 1, 0]
    assert np.allclose(ctmc.exit_rates, [6.0, 4.0, 2.0, 0.0])


def test_vanishing_initial_marking_rejected():
    gspn = StochasticPetriNet("vanishing-start")
    gspn.add_place("choice", initial=1)
    gspn.add_place("done")
    gspn.add_immediate_transition("go")
    gspn.add_arc("choice", "go")
    gspn.add_arc("go", "done")
    with pytest.raises(ValueError):
        build_gspn_ctmc(gspn)


def test_simulation_respects_weights():
    gspn = build_routing(weight_left=3.0, weight_right=1.0)
    events = simulate_gspn_ssa(gspn, time_end=20_000, seed=3)
    lefts = sum(1 for event in events if event.transition == "go_left")
    rights = sum(1 for event in events if event.transition == "go_right")
    assert lefts + rights > 5_000
    assert abs(lefts / (lefts + rights) - 0.75) < 0.02
