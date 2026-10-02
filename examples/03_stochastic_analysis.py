"""Stochastic Petri Nets: CTMC, stationary distribution, throughput and SSA."""

from collections import Counter

from petrinet_lab.agent.tools import build_machine
from petrinet_lab.core.stochastic import (
    build_ctmc,
    mean_sojourn_times,
    simulate_ssa,
    stationary_distribution,
    throughputs,
    transient_distribution,
)

net = build_machine()
rates = {"failure": 1.0, "repair": 3.0}
ctmc = build_ctmc(net, rates)
print("states:", [ctmc.marking(i) for i in range(len(ctmc.states))])
distribution = stationary_distribution(ctmc)
for state in range(len(ctmc.states)):
    print(f"  pi{ctmc.marking(state)} = {distribution[state]:.4f}")
print("throughput:", throughputs(ctmc, distribution))
print("mean sojourn times:", mean_sojourn_times(ctmc))
print("transient at t=0.5:", transient_distribution(ctmc, 0.5))

events = simulate_ssa(net, rates, time_end=5_000, seed=42)
counts = Counter(event.transition for event in events)
print("SSA events:", len(events), dict(counts))
occupied = {0: 0.0, 1: 0.0}
previous = 0.0
state = 0
for event in events:
    occupied[state] += event.time - previous
    previous = event.time
    state = 1 - state
print("empirical time-weighted fraction up:", occupied[0] / previous)
