"""Generalised Stochastic Petri Nets (GSPN) with immediate and timed transitions.

Timed transitions fire after an exponential delay whose rate may be constant
or marking-dependent. Immediate transitions fire in zero time and are selected
by priority first (lower value wins) and weight (proportional choice among the
enabled transitions of the highest priority class). Markings with at least one
enabled immediate transition are *vanishing*: they are eliminated by solving
the absorption probabilities of the immediate sub-chain, so the analysis
reduces to a CTMC over *tangible* markings. Optional guards restrict the
enabling of any transition as a function of the marking.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from .model import Marking, PetriNet
from .stochastic import CTMC, Rate, StochasticEvent, rate_of

Guard = Callable[[Marking], bool]


class ImmediateLoopError(RuntimeError):
    """Raised when immediate transitions trap probability mass with no tangible exit."""


@dataclass(frozen=True)
class TimedTransition:
    id: str
    rate: Rate


@dataclass(frozen=True)
class ImmediateTransition:
    id: str
    weight: float = 1.0
    priority: int = 0


@dataclass(frozen=True)
class ResolutionResult:
    """Elimination of the vanishing markings reachable from a start marking."""

    targets: dict[tuple[int, ...], float]
    expected_firings: dict[str, float]


class StochasticPetriNet:
    """Builder for GSPNs: places and arcs of a P/T net plus transition timing."""

    def __init__(self, name: str = "StochasticPetriNet") -> None:
        self._net = PetriNet(name)
        self._timed: dict[str, TimedTransition] = {}
        self._immediate: dict[str, ImmediateTransition] = {}
        self._guards: dict[str, Guard] = {}

    @property
    def net(self) -> PetriNet:
        return self._net

    @property
    def name(self) -> str:
        return self._net.name

    @property
    def timed_ids(self) -> list[str]:
        return list(self._timed)

    @property
    def immediate_ids(self) -> list[str]:
        return list(self._immediate)

    def add_place(self, place_id: str, name: str = "", initial: int = 0):
        return self._net.add_place(place_id, name, initial)

    def add_timed_transition(self, transition_id: str, rate: Rate, name: str = "") -> TimedTransition:
        if not callable(rate) and float(rate) < 0:
            raise ValueError(f"negative firing rate for transition {transition_id!r}")
        transition = TimedTransition(transition_id, rate)
        self._net.add_transition(transition_id, name)
        self._timed[transition_id] = transition
        return transition

    def add_immediate_transition(
        self,
        transition_id: str,
        weight: float = 1.0,
        priority: int = 0,
        name: str = "",
    ) -> ImmediateTransition:
        if weight <= 0:
            raise ValueError(f"immediate transition {transition_id!r} needs a positive weight")
        if priority < 0:
            raise ValueError(f"immediate transition {transition_id!r} needs a non-negative priority")
        transition = ImmediateTransition(transition_id, float(weight), int(priority))
        self._net.add_transition(transition_id, name)
        self._immediate[transition_id] = transition
        return transition

    def add_arc(self, source: str, target: str, weight: int = 1):
        return self._net.add_arc(source, target, weight)

    def set_guard(self, transition_id: str, guard: Guard) -> None:
        self._require_transition(transition_id)
        self._guards[transition_id] = guard

    def clear_guard(self, transition_id: str) -> None:
        self._guards.pop(transition_id, None)

    def transition_kind(self, transition_id: str) -> str:
        self._require_transition(transition_id)
        return "timed" if transition_id in self._timed else "immediate"

    def enabled(self, marking: Mapping[str, int], transition_id: str) -> bool:
        if not self._net.enabled(marking, transition_id):
            return False
        guard = self._guards.get(transition_id)
        return True if guard is None else bool(guard(marking))

    def enabled_timed(self, marking: Mapping[str, int]) -> list[TimedTransition]:
        return [t for t in self._timed.values() if self.enabled(marking, t.id)]

    def enabled_immediates(self, marking: Mapping[str, int]) -> list[ImmediateTransition]:
        enabled = [t for t in self._immediate.values() if self.enabled(marking, t.id)]
        if not enabled:
            return []
        top = min(t.priority for t in enabled)
        return [t for t in enabled if t.priority == top]

    def is_vanishing(self, marking: Mapping[str, int]) -> bool:
        return bool(self.enabled_immediates(marking))

    def build_ctmc(self, max_states: int = 200_000) -> GSPNCTMC:
        return build_gspn_ctmc(self, max_states=max_states)

    def summary(self) -> dict[str, object]:
        return {
            "name": self._net.name,
            "places": len(self._net.places),
            "timed_transitions": len(self._timed),
            "immediate_transitions": len(self._immediate),
            "arcs": len(self._net.arcs),
            "guards": len(self._guards),
        }

    def _require_transition(self, transition_id: str) -> None:
        if transition_id not in self._timed and transition_id not in self._immediate:
            raise KeyError(f"unknown transition {transition_id!r}")


class GSPNCTMC(CTMC):
    """CTMC over tangible markings with immediate-transition flow accounting."""

    def __init__(
        self,
        net: PetriNet,
        states: list[tuple[int, ...]],
        transitions: list[list[tuple[str, float, int]]],
        self_loop_rates: np.ndarray,
        generator: np.ndarray,
        immediate_ids: list[str],
        immediate_flows: np.ndarray,
    ) -> None:
        super().__init__(net, states, transitions, self_loop_rates, generator)
        self.immediate_ids = immediate_ids
        self.immediate_flows = immediate_flows

    def immediate_throughput(self, transition_id: str, distribution: np.ndarray | None = None) -> float:
        if transition_id not in self.immediate_ids:
            raise KeyError(f"unknown immediate transition {transition_id!r}")
        if distribution is None:
            from .stochastic import stationary_distribution

            distribution = stationary_distribution(self)
        column = self.immediate_ids.index(transition_id)
        return float(np.dot(distribution, self.immediate_flows[:, column]))


def resolve_marking(gspn: StochasticPetriNet, marking: Mapping[str, int]) -> ResolutionResult:
    """Eliminate the immediate sub-chain from ``marking`` down to tangible markings.

    Returns the absorption probabilities over tangible markings and the
    expected number of firings of each immediate transition along the way.
    """
    net = gspn.net
    start = net.marking_key(marking)
    enabled = gspn.enabled_immediates(marking)
    if not enabled:
        return ResolutionResult({start: 1.0}, {})

    order: list[tuple[int, ...]] = []
    index: dict[tuple[int, ...], int] = {}
    outgoing: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
    probabilities: dict[tuple[int, ...], dict[str, float]] = {}
    queue = deque([start])
    index[start] = 0
    order.append(start)
    while queue:
        key = queue.popleft()
        current = net.marking_from_key(key)
        choices = gspn.enabled_immediates(current)
        total_weight = sum(choice.weight for choice in choices)
        targets: dict[tuple[int, ...], float] = {}
        firing_probabilities: dict[str, float] = {}
        for choice in choices:
            probability = choice.weight / total_weight
            firing_probabilities[choice.id] = probability
            successor_key = net.marking_key(net.fire(current, choice.id))
            targets[successor_key] = targets.get(successor_key, 0.0) + probability
        outgoing[key] = targets
        probabilities[key] = firing_probabilities
        for successor_key in targets:
            if successor_key not in index and gspn.is_vanishing(net.marking_from_key(successor_key)):
                index[successor_key] = len(order)
                order.append(successor_key)
                queue.append(successor_key)

    vanishing_order = [
        key for key in order if key in outgoing and gspn.is_vanishing(net.marking_from_key(key))
    ]
    vanishing_indices = {key: i for i, key in enumerate(vanishing_order)}
    rows = len(vanishing_indices)
    tangible_targets: list[tuple[int, ...]] = []
    seen_targets: set[tuple[int, ...]] = set()
    for key in vanishing_order:
        for target in outgoing[key]:
            if target not in vanishing_indices and target not in seen_targets:
                seen_targets.add(target)
                tangible_targets.append(target)
    target_indices = {key: i for i, key in enumerate(tangible_targets)}
    matrix_q = np.zeros((rows, rows))
    matrix_r = np.zeros((rows, len(tangible_targets)))
    for key in vanishing_order:
        row = vanishing_indices[key]
        for target, probability in outgoing[key].items():
            if target in vanishing_indices:
                matrix_q[row, vanishing_indices[target]] += probability
            else:
                matrix_r[row, target_indices[target]] += probability
    if rows:
        try:
            fundamental = np.linalg.solve(np.eye(rows) - matrix_q, np.eye(rows))
        except np.linalg.LinAlgError as error:
            raise ImmediateLoopError(
                "immediate transitions form a closed class with no tangible exit"
            ) from error
    else:
        fundamental = np.zeros((0, 0))
    absorption = fundamental @ matrix_r
    start_row = vanishing_indices[start]
    targets_result = {
        key: float(absorption[start_row, i])
        for key, i in target_indices.items()
        if absorption[start_row, i] > 1e-12
    }
    mass = sum(targets_result.values())
    if abs(mass - 1.0) > 1e-9:
        raise ImmediateLoopError(
            "immediate transitions trap probability mass in a vanishing class"
        )
    expected: dict[str, float] = {}
    for key in vanishing_order:
        row = vanishing_indices[key]
        for transition_id, probability in probabilities[key].items():
            expected[transition_id] = expected.get(transition_id, 0.0) + (
                float(fundamental[start_row, row]) * probability
            )
    expected = {key: value for key, value in expected.items() if abs(value) > 1e-12}
    return ResolutionResult(targets_result, expected)


class _ResolutionCache:
    def __init__(self, gspn: StochasticPetriNet) -> None:
        self.gspn = gspn
        self._cache: dict[tuple[int, ...], ResolutionResult] = {}

    def resolve(self, marking: Mapping[str, int]) -> ResolutionResult:
        key = self.gspn.net.marking_key(marking)
        cached = self._cache.get(key)
        if cached is None:
            cached = resolve_marking(self.gspn, marking)
            self._cache[key] = cached
        return cached


def build_gspn_ctmc(gspn: StochasticPetriNet, max_states: int = 200_000) -> GSPNCTMC:
    net = gspn.net
    resolver = _ResolutionCache(gspn)
    initial = net.initial_marking()
    if gspn.is_vanishing(initial):
        raise ValueError("the initial marking must be tangible for CTMC construction")
    start_key = net.marking_key(initial)

    immediate_ids = gspn.immediate_ids
    immediate_index = {transition_id: i for i, transition_id in enumerate(immediate_ids)}
    states = [start_key]
    index = {start_key: 0}
    transitions: list[list[tuple[str, float, int]]] = [[]]
    self_loops = [0.0]
    flows = [np.zeros(len(immediate_ids)) for _ in states]
    queue = deque([start_key])
    while queue:
        key = queue.popleft()
        source = index[key]
        marking = net.marking_from_key(key)
        departures: dict[tuple[str, int], float] = {}
        for timed in gspn.enabled_timed(marking):
            rate = rate_of(timed.rate, timed.id, marking)
            if rate == 0:
                continue
            resolution = resolver.resolve(net.fire(marking, timed.id))
            for transition_id, count in resolution.expected_firings.items():
                flows[source][immediate_index[transition_id]] += rate * count
            for target_key, probability in resolution.targets.items():
                contribution = rate * probability
                if target_key == key:
                    self_loops[source] += contribution
                    continue
                target = index.get(target_key)
                if target is None:
                    if len(states) >= max_states:
                        from .stochastic import StateSpaceTooLarge

                        raise StateSpaceTooLarge(
                            f"CTMC exceeded max_states={max_states}"
                        )
                    target = len(states)
                    index[target_key] = target
                    states.append(target_key)
                    transitions.append([])
                    self_loops.append(0.0)
                    flows.append(np.zeros(len(immediate_ids)))
                    queue.append(target_key)
                departures[(timed.id, target)] = departures.get((timed.id, target), 0.0) + contribution
        transitions[source] = [
            (transition_id, rate, target) for (transition_id, target), rate in departures.items()
        ]
    size = len(states)
    generator = np.zeros((size, size), dtype=float)
    for i, outgoing in enumerate(transitions):
        for _, rate, target in outgoing:
            generator[i, target] += rate
        generator[i, i] = -float(sum(rate for _, rate, _ in outgoing))
    flows_array = np.array(flows, dtype=float).reshape(size, len(immediate_ids))
    return GSPNCTMC(
        net,
        states,
        transitions,
        np.array(self_loops, dtype=float),
        generator,
        immediate_ids,
        flows_array,
    )


def immediate_throughputs(
    ctmc: GSPNCTMC, distribution: np.ndarray | None = None
) -> dict[str, float]:
    if distribution is None:
        from .stochastic import stationary_distribution

        distribution = stationary_distribution(ctmc)
    return {
        transition_id: float(np.dot(distribution, ctmc.immediate_flows[:, column]))
        for column, transition_id in enumerate(ctmc.immediate_ids)
    }


def simulate_gspn_ssa(
    gspn: StochasticPetriNet,
    time_end: float | None = None,
    seed: int | None = None,
    max_events: int = 1_000_000,
    max_immediate_steps: int = 10_000,
) -> list[StochasticEvent]:
    """Exact simulation: immediates fire at the current epoch, timeds after an exponential delay."""
    rng = random.Random(seed)
    marking = gspn.net.initial_marking()
    now = 0.0
    events: list[StochasticEvent] = []
    for _ in range(max_events):
        steps = 0
        while True:
            immediates = gspn.enabled_immediates(marking)
            if not immediates:
                break
            steps += 1
            if steps > max_immediate_steps:
                raise ImmediateLoopError(
                    "immediate transitions did not reach a tangible marking; likely an infinite loop"
                )
            total_weight = sum(choice.weight for choice in immediates)
            threshold = rng.random() * total_weight
            cumulative = 0.0
            chosen = immediates[-1].id
            for choice in immediates:
                cumulative += choice.weight
                if cumulative >= threshold:
                    chosen = choice.id
                    break
            marking = gspn.net.fire(marking, chosen)
            events.append(StochasticEvent(now, chosen, dict(marking)))
        candidates = [
            (timed.id, rate_of(timed.rate, timed.id, marking))
            for timed in gspn.enabled_timed(marking)
        ]
        candidates = [(transition_id, rate) for transition_id, rate in candidates if rate > 0]
        total = sum(rate for _, rate in candidates)
        if total <= 0:
            break
        now += rng.expovariate(total)
        if time_end is not None and now > time_end:
            break
        threshold = rng.random() * total
        cumulative = 0.0
        chosen = candidates[-1][0]
        for transition_id, rate in candidates:
            cumulative += rate
            if cumulative >= threshold:
                chosen = transition_id
                break
        marking = gspn.net.fire(marking, chosen)
        events.append(StochasticEvent(now, chosen, dict(marking)))
    return events
