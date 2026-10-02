"""Stochastic Petri Nets with exponential firing times.

Rates may be constant or marking-dependent callables. The marking process is
a continuous-time Markov chain (CTMC) that this module builds explicitly,
computes stationary and transient distributions for, and simulates exactly
with the Gillespie stochastic simulation algorithm.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from .model import Marking, PetriNet

Rate = float | Callable[[Marking], float]


class StateSpaceTooLarge(RuntimeError):
    pass


class CTMC:
    def __init__(
        self,
        net: PetriNet,
        states: list[tuple[int, ...]],
        transitions: list[list[tuple[str, float, int]]],
        self_loop_rates: np.ndarray,
        generator: np.ndarray,
    ) -> None:
        self.net = net
        self.states = states
        self.transitions = transitions
        self.self_loop_rates = self_loop_rates
        self.generator = generator
        self._index = {key: i for i, key in enumerate(states)}
        self.exit_rates = -np.diag(generator)

    @property
    def index(self) -> dict[tuple[int, ...], int]:
        return dict(self._index)

    def marking(self, state: int) -> Marking:
        return self.net.marking_from_key(self.states[state])

    def state_of(self, marking: Mapping[str, int]) -> int:
        return self._index[self.net.marking_key(marking)]

    def state_names(self) -> list[str]:
        return [str(self.marking(i)) for i in range(len(self.states))]


def _rate_of(rate: Rate | Mapping[str, Rate], transition_id: str, marking: Marking) -> float:
    value = rate[transition_id] if isinstance(rate, Mapping) else rate
    if callable(value):
        resolved = float(value(marking))
    else:
        resolved = float(value)
    if resolved < 0:
        raise ValueError(f"negative firing rate for transition {transition_id!r}")
    return resolved


def build_ctmc(
    net: PetriNet,
    rates: Mapping[str, Rate],
    max_states: int = 200_000,
) -> CTMC:
    initial = net.marking_key(net.initial_marking())
    states = [initial]
    index = {initial: 0}
    transitions: list[list[tuple[str, float, int]]] = [[]]
    self_loops: list[float] = [0.0]
    queue = deque([initial])
    while queue:
        key = queue.popleft()
        source = index[key]
        marking = net.marking_from_key(key)
        for transition_id in net.enabled_transitions(marking):
            rate = _rate_of(rates, transition_id, marking)
            if rate == 0:
                continue
            successor_key = net.marking_key(net.fire(marking, transition_id))
            if successor_key == key:
                self_loops[source] += rate
                continue
            target = index.get(successor_key)
            if target is None:
                if len(states) >= max_states:
                    raise StateSpaceTooLarge(
                        f"CTMC exceeded max_states={max_states}"
                    )
                target = len(states)
                index[successor_key] = target
                states.append(successor_key)
                transitions.append([])
                self_loops.append(0.0)
                queue.append(successor_key)
            transitions[source].append((transition_id, rate, target))
    size = len(states)
    generator = np.zeros((size, size), dtype=float)
    for i, outgoing in enumerate(transitions):
        for _, rate, target in outgoing:
            generator[i, target] += rate
        generator[i, i] = -float(sum(rate for _, rate, _ in outgoing))
    return CTMC(net, states, transitions, np.array(self_loops, dtype=float), generator)


def stationary_distribution(ctmc: CTMC, tolerance: float = 1e-10) -> np.ndarray:
    size = len(ctmc.states)
    if size == 1:
        return np.ones(1)
    system = ctmc.generator.T
    rhs = np.zeros(size)
    system = np.vstack([system, np.ones(size)])
    rhs = np.append(rhs, 1.0)
    solution, _, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
    if rank < size:
        raise ValueError(
            "stationary distribution is not unique (reducible or absorbing chain); "
            "use long_run_distribution from an initial state"
        )
    if np.any(solution < -tolerance):
        raise ValueError("stationary distribution has negative entries; check the model")
    solution = np.clip(solution, 0.0, None)
    return solution / solution.sum()


def transient_distribution(
    ctmc: CTMC,
    time: float,
    initial: int | None = None,
    tolerance: float = 1e-12,
    max_terms: int = 100_000,
) -> np.ndarray:
    size = len(ctmc.states)
    state = np.zeros(size)
    state[0 if initial is None else initial] = 1.0
    rates = ctmc.exit_rates
    uniformisation = float(rates.max()) if size else 0.0
    if uniformisation == 0.0:
        return state
    kernel = np.eye(size) + ctmc.generator / uniformisation
    weight = float(np.exp(-uniformisation * time))
    result = weight * state
    cumulative = weight
    for k in range(1, max_terms + 1):
        state = state @ kernel
        weight *= uniformisation * time / k
        result = result + weight * state
        cumulative += weight
        if weight < tolerance and 1.0 - cumulative < tolerance:
            break
    total = result.sum()
    return result / total if total > 0 else result


def long_run_distribution(
    ctmc: CTMC,
    initial: int | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 1_000_000,
) -> np.ndarray:
    size = len(ctmc.states)
    state = np.zeros(size)
    state[0 if initial is None else initial] = 1.0
    uniformisation = float(ctmc.exit_rates.max()) if size else 0.0
    if uniformisation == 0.0:
        return state
    kernel = np.eye(size) + ctmc.generator / uniformisation
    for _ in range(max_iterations):
        successor = state @ kernel
        if np.max(np.abs(successor - state)) < tolerance:
            total = successor.sum()
            return successor / total if total > 0 else successor
        state = successor
    total = state.sum()
    return state / total if total > 0 else state


def throughputs(ctmc: CTMC, distribution: np.ndarray | None = None) -> dict[str, float]:
    if distribution is None:
        distribution = stationary_distribution(ctmc)
    totals: dict[str, float] = {}
    for state, outgoing in enumerate(ctmc.transitions):
        for transition_id, rate, _ in outgoing:
            totals[transition_id] = totals.get(transition_id, 0.0) + distribution[state] * rate
    for state, rate in enumerate(ctmc.self_loop_rates):
        if rate:
            key = "__self_loop__"
            totals[key] = totals.get(key, 0.0) + distribution[state] * rate
    return totals


def mean_sojourn_times(ctmc: CTMC) -> dict[int, float]:
    return {
        state: (float("inf") if ctmc.exit_rates[state] == 0 else 1.0 / ctmc.exit_rates[state])
        for state in range(len(ctmc.states))
    }


@dataclass(frozen=True)
class StochasticEvent:
    time: float
    transition: str
    marking: Marking


def simulate_ssa(
    net: PetriNet,
    rates: Mapping[str, Rate],
    time_end: float,
    seed: int | None = None,
    max_events: int = 1_000_000,
) -> list[StochasticEvent]:
    rng = random.Random(seed)
    marking = net.initial_marking()
    now = 0.0
    events: list[StochasticEvent] = []
    for _ in range(max_events):
        candidates: list[tuple[str, float]] = []
        for transition_id in net.enabled_transitions(marking):
            rate = _rate_of(rates, transition_id, marking)
            if rate > 0:
                candidates.append((transition_id, rate))
        total = sum(rate for _, rate in candidates)
        if total <= 0:
            break
        now += rng.expovariate(total)
        if now > time_end:
            break
        threshold = rng.random() * total
        cumulative = 0.0
        chosen = candidates[-1][0]
        for transition_id, rate in candidates:
            cumulative += rate
            if cumulative >= threshold:
                chosen = transition_id
                break
        marking = net.fire(marking, chosen)
        events.append(StochasticEvent(now, chosen, dict(marking)))
    return events
