import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from petrinet_lab.core.model import PetriNet


def build_cycle() -> PetriNet:
    net = PetriNet("cycle")
    net.add_place("ready", initial=1)
    net.add_place("busy")
    net.add_transition("start")
    net.add_transition("end")
    net.add_arc("ready", "start")
    net.add_arc("start", "busy")
    net.add_arc("busy", "end")
    net.add_arc("end", "ready")
    return net


def build_deadlock() -> PetriNet:
    net = PetriNet("deadlock")
    net.add_place("p1", initial=1)
    net.add_place("p2", initial=1)
    net.add_place("p3")
    net.add_place("p4")
    net.add_transition("t1")
    net.add_transition("t2")
    net.add_arc("p1", "t1")
    net.add_arc("p2", "t1")
    net.add_arc("t1", "p3")
    net.add_arc("p3", "t2")
    net.add_arc("p4", "t2")
    net.add_arc("t2", "p1")
    return net


def build_workflow() -> PetriNet:
    net = PetriNet("workflow")
    net.add_place("start", initial=1)
    net.add_place("pa")
    net.add_place("pb")
    net.add_place("p1")
    net.add_place("p2")
    net.add_place("end")
    net.add_transition("split")
    net.add_transition("a")
    net.add_transition("b")
    net.add_transition("join")
    net.add_arc("start", "split")
    net.add_arc("split", "pa")
    net.add_arc("split", "pb")
    net.add_arc("pa", "a")
    net.add_arc("a", "p1")
    net.add_arc("pb", "b")
    net.add_arc("b", "p2")
    net.add_arc("p1", "join")
    net.add_arc("p2", "join")
    net.add_arc("join", "end")
    return net


@pytest.fixture
def cycle():
    return build_cycle()


@pytest.fixture
def deadlock():
    return build_deadlock()


@pytest.fixture
def workflow():
    return build_workflow()
