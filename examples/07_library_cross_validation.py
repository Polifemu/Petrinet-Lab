"""Cross-validate the hand-written core against SNAKES and PM4Py.

For each model we compare the state space computed by three independent
engines: the native reachability graph, SNAKES enabling/firing exploration and
PM4Py's transition-system construction.
"""

from petrinet_lab.agent.tools import build_cyclic_cell, build_parallel_workflow
from petrinet_lab.core.model import PetriNet
from petrinet_lab.interop.pm4py_bridge import compare_state_spaces as pm4py_compare
from petrinet_lab.interop.snakes_bridge import compare_state_spaces as snakes_compare


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


MODELS = [build_cyclic_cell(), build_parallel_workflow(), build_weighted(), build_deadlock()]

header = f"{'model':<20}{'native':>8}{'SNAKES':>8}{'PM4Py':>8}{'states ok':>11}{'edges ok':>10}"
print(header)
print("-" * len(header))
for net in MODELS:
    snakes = snakes_compare(net)
    pm4py = pm4py_compare(net)
    print(
        f"{net.name:<20}{snakes['native_states']:>8}{snakes['snakes_states']:>8}"
        f"{pm4py['pm4py_states']:>8}{str(snakes['states_equal']):>11}{str(snakes['edges_equal']):>10}"
    )
    assert snakes["states_equal"] and snakes["edges_equal"]
    assert pm4py["states_equal"] and pm4py["edges_equal"]

print("\nAll engines agree on states and edges for every model.")
