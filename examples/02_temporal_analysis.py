"""Time Petri Nets: state classes, firing windows and a time-lock."""

from fractions import Fraction

from petrinet_lab.agent.tools import build_cyclic_cell, build_timelock
from petrinet_lab.core.temporal import (
    TimePetriNet,
    build_state_class_graph,
    simulate_timed,
)

cell = TimePetriNet(build_cyclic_cell(), {"start": (2, 4), "finish": (3, 5)})
graph = build_state_class_graph(cell)
print("cyclic cell: state classes =", len(graph.classes))
for transition in cell.net.transition_ids:
    print(f"  {transition}: firing time bounds {graph.firing_time_bounds(transition)}")

events = simulate_timed(cell, steps=6, seed=3)
print("timed trace (random policy):")
for event in events:
    print(f"  t={float(event.time):6.2f}  fire {event.transition}")

lock = TimePetriNet(build_timelock(), {"fast": (1, 2), "slow": (5, 10)})
lock_graph = build_state_class_graph(lock)
print("time-lock model: firable from initial =", lock_graph.firable_transitions(0))
print("  slow firing bounds =", lock_graph.firing_time_bounds("slow"))

deterministic = simulate_timed(cell, steps=4, policy="earliest")
print("earliest-policy trace:", [(e.transition, e.time) for e in deterministic])
assert deterministic[0].time == Fraction(2)
