"""Time Petri Nets (Merlin & Farber) with exact state-class graph analysis.

Semantics: strong time semantics. Each transition has a static firing
interval [earliest, latest]; an enabled transition must fire within its
interval, otherwise the system cannot let time pass further (time-lock).
State classes group markings with a domain of possible clock values,
represented as a Difference Bound Matrix (DBM) over enabling dates.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import inf

from .model import Marking, PetriNet

Number = Fraction | float


def _exact(value: Number) -> Number:
    return inf if value == inf else Fraction(value)


class DBM:
    """Difference Bound Matrix over enabling dates.

    Node 0 is the constant 0; variable ``v`` at index ``i`` satisfies
    ``v_i - v_j <= bounds[i][j]``. Variables are the enabled transitions and
    hold the enabling date relative to the class time (always <= 0).
    """

    __slots__ = ("bounds", "variables")

    def __init__(self, variables: Iterable[str], bounds: list[list[Number]]) -> None:
        self.variables = tuple(variables)
        self.bounds = bounds

    @classmethod
    def single_point(cls, variables: Iterable[str], values: Iterable[Number]) -> DBM:
        variables = tuple(variables)
        values = list(values)
        size = len(variables) + 1
        bounds: list[list[Number]] = [[inf] * size for _ in range(size)]
        for i in range(size):
            bounds[i][i] = Fraction(0)
        for i, value in enumerate(values, start=1):
            bounds[i][0] = value
            bounds[0][i] = -value
        return cls(variables, bounds)

    def copy(self) -> DBM:
        return DBM(self.variables, [row[:] for row in self.bounds])

    def index(self, variable: str) -> int:
        return self.variables.index(variable) + 1

    def size(self) -> int:
        return len(self.variables) + 1

    def set_bound(self, i: int, j: int, value: Number) -> None:
        self.bounds[i][j] = min(self.bounds[i][j], value)

    def close(self) -> DBM:
        size = self.size()
        for k in range(size):
            row_k = self.bounds[k]
            for i in range(size):
                value_ik = self.bounds[i][k]
                if value_ik == inf:
                    continue
                row_i = self.bounds[i]
                for j in range(size):
                    if row_k[j] == inf:
                        continue
                    candidate = value_ik + row_k[j]
                    row_i[j] = min(row_i[j], candidate)
        return self

    def feasible(self) -> bool:
        return all(self.bounds[i][i] >= 0 for i in range(self.size()))

    def lower_bound(self, variable: str) -> Number:
        return -self.bounds[0][self.index(variable)]

    def upper_bound(self, variable: str) -> Number:
        return self.bounds[self.index(variable)][0]

    def variable_bound(self, first: str, second: str) -> Number:
        return self.bounds[self.index(first)][self.index(second)]

    def included_in(self, other: DBM) -> bool:
        if self.variables != other.variables:
            return False
        return all(
            self.bounds[i][j] <= other.bounds[i][j]
            for i in range(self.size())
            for j in range(self.size())
        )

    def with_free_variable(self, name: str) -> DBM:
        size = self.size()
        bounds: list[list[Number]] = [[inf] * (size + 1) for _ in range(size + 1)]
        for i in range(size + 1):
            bounds[i][i] = Fraction(0)
        for i in range(size):
            for j in range(size):
                bounds[i][j] = self.bounds[i][j]
        return DBM((*self.variables, name), bounds)


@dataclass(frozen=True)
class TimedEdge:
    source: int
    target: int
    transition: str
    min_delay: Number
    max_delay: Number


@dataclass
class StateClass:
    id: int
    marking: tuple[int, ...]
    variables: tuple[str, ...]
    dbm: DBM

    @property
    def enabled(self) -> tuple[str, ...]:
        return self.variables


class StateClassGraph:
    def __init__(
        self,
        net: PetriNet,
        classes: list[StateClass],
        edges: list[TimedEdge],
        truncated: bool = False,
    ) -> None:
        self.net = net
        self.classes = classes
        self.edges = edges
        self.truncated = truncated
        self._out: dict[int, list[TimedEdge]] = {}
        for edge in edges:
            self._out.setdefault(edge.source, []).append(edge)

    def successor_edges(self, class_id: int) -> list[TimedEdge]:
        return list(self._out.get(class_id, []))

    def firable_transitions(self, class_id: int) -> list[str]:
        return [edge.transition for edge in self._out.get(class_id, [])]

    def reachable_markings(self) -> set[tuple[int, ...]]:
        return {cls.marking for cls in self.classes}

    def deadlock_classes(self) -> list[StateClass]:
        return [cls for cls in self.classes if not cls.variables]

    def time_lock_classes(self) -> list[StateClass]:
        return [cls for cls in self.classes if cls.variables and not self._out.get(cls.id)]

    def firing_time_bounds(self, transition_id: str) -> tuple[Number, Number] | None:
        delays = [
            (edge.min_delay, edge.max_delay)
            for edge in self.edges
            if edge.transition == transition_id
        ]
        if not delays:
            return None
        return (
            min(low for low, _ in delays),
            max(high for _, high in delays),
        )


class TimePetriNet:
    def __init__(
        self,
        net: PetriNet,
        intervals: Mapping[str, tuple[Number, Number]],
        default: tuple[Number, Number] = (0, inf),
    ) -> None:
        self.net = net
        self.default = (_exact(default[0]), _exact(default[1]))
        self.intervals = {
            tid: tuple(map(_exact, intervals.get(tid, self.default)))  # type: ignore[misc]
            for tid in net.transition_ids
        }
        for tid, (earliest, latest) in self.intervals.items():
            if earliest < 0:
                raise ValueError(f"transition {tid!r} has a negative earliest time")
            if latest < earliest:
                raise ValueError(f"transition {tid!r} has latest < earliest")

    def interval(self, transition_id: str) -> tuple[Number, Number]:
        return self.intervals[transition_id]


_DELTA = "__delta__"


def _firing_system(tpn: TimePetriNet, dbm: DBM, transition_id: str,
                   alpha: Number, beta: Number) -> DBM | None:
    """Exact DBM of the firing system: class domain plus waiting time delta."""
    work = dbm.with_free_variable(_DELTA)
    delta_index = work.index(_DELTA)
    transition_index = work.index(transition_id)
    work.set_bound(0, delta_index, Fraction(0))
    work.set_bound(transition_index, delta_index, -alpha)
    for variable in dbm.variables:
        upper = tpn.interval(variable)[1]
        work.set_bound(delta_index, work.index(variable), upper)
    work.close()
    return work if work.feasible() else None


def _successor(net: PetriNet, marking: Mapping[str, int], cls: StateClass,
               transition_id: str, system: DBM) -> StateClass:
    successor_marking = net.fire(marking, transition_id)
    after_consume = dict(marking)
    for pid, weight in net.preset(transition_id).items():
        after_consume[pid] = after_consume.get(pid, 0) - weight
    enabled_before = set(net.enabled_transitions(after_consume))
    enabled_after = net.enabled_transitions(successor_marking)
    newly = [u for u in enabled_after if u not in enabled_before]
    persistent = [u for u in enabled_after if u in enabled_before]
    variables = tuple(sorted(persistent + newly))
    size = len(variables) + 1
    bounds: list[list[Number]] = [[inf] * size for _ in range(size)]
    for i in range(size):
        bounds[i][i] = Fraction(0)
    index = {name: i + 1 for i, name in enumerate(variables)}
    delta_index = system.index(_DELTA)
    for name in persistent:
        i = index[name]
        source = system.index(name)
        bounds[i][0] = system.bounds[source][delta_index]
        bounds[0][i] = system.bounds[delta_index][source]
        for other in persistent:
            bounds[i][index[other]] = system.variable_bound(name, other)
    for name in newly:
        i = index[name]
        bounds[i][0] = Fraction(0)
        bounds[0][i] = Fraction(0)
        for other in persistent:
            j = index[other]
            bounds[i][j] = bounds[0][j]
            bounds[j][i] = bounds[j][0]
    dbm = DBM(variables, bounds)
    dbm.close()
    return StateClass(-1, net.marking_key(successor_marking), variables, dbm)


def build_state_class_graph(tpn: TimePetriNet, max_classes: int = 20_000) -> StateClassGraph:
    net = tpn.net
    initial_marking = net.initial_marking()
    initial_key = net.marking_key(initial_marking)
    enabled = tuple(sorted(net.enabled_transitions(initial_marking)))
    initial_dbm = DBM.single_point(enabled, [Fraction(0)] * len(enabled))
    classes = [StateClass(0, initial_key, enabled, initial_dbm)]
    edges: list[TimedEdge] = []
    per_marking: dict[tuple[int, ...], list[int]] = {initial_key: [0]}
    queue = deque([0])
    truncated = False
    while queue:
        class_id = queue.popleft()
        cls = classes[class_id]
        marking = net.marking_from_key(cls.marking)
        for transition_id in cls.variables:
            alpha, beta = tpn.interval(transition_id)
            system = _firing_system(tpn, cls.dbm, transition_id, alpha, beta)
            if system is None:
                continue
            min_delay = system.lower_bound(_DELTA)
            max_delay = system.upper_bound(_DELTA)
            successor = _successor(net, marking, cls, transition_id, system)
            target = None
            for candidate_id in per_marking.get(successor.marking, []):
                if successor.dbm.included_in(classes[candidate_id].dbm):
                    target = candidate_id
                    break
            if target is None:
                if len(classes) >= max_classes:
                    truncated = True
                    continue
                target = len(classes)
                successor.id = target
                classes.append(successor)
                per_marking.setdefault(successor.marking, []).append(target)
                queue.append(target)
            edges.append(TimedEdge(class_id, target, transition_id, min_delay, max_delay))
    return StateClassGraph(net, classes, edges, truncated)


@dataclass(frozen=True)
class TimedEvent:
    time: Fraction
    transition: str
    marking: Marking


def simulate_timed(
    tpn: TimePetriNet,
    steps: int = 100,
    seed: int | None = None,
    policy: str = "random",
) -> list[TimedEvent]:
    net = tpn.net
    rng = random.Random(seed)
    marking = net.initial_marking()
    ages: dict[str, Number] = {t: Fraction(0) for t in net.enabled_transitions(marking)}
    now: Number = Fraction(0)
    events: list[TimedEvent] = []
    for _ in range(steps):
        candidates = []
        for transition_id, age in ages.items():
            alpha, beta = tpn.interval(transition_id)
            wait_min = max(Fraction(0), alpha - age)
            wait_max = min(
                beta - other_age for other_age in ages.values()
            )
            if wait_min <= wait_max:
                candidates.append((transition_id, wait_min, wait_max))
        if not candidates:
            break
        if policy == "earliest":
            transition_id, wait_min, wait_max = min(candidates, key=lambda item: item[1])
        else:
            transition_id, wait_min, wait_max = rng.choice(candidates)
        if policy == "earliest":
            delta = wait_min
        elif wait_max == inf:
            delta = wait_min + Fraction(rng.expovariate(1.0)).limit_denominator(10**9)
        else:
            lower, upper = Fraction(wait_min), Fraction(wait_max)
            delta = lower + (upper - lower) * Fraction(rng.random()).limit_denominator(10**9)
        after_consume = dict(marking)
        for pid, weight in net.preset(transition_id).items():
            after_consume[pid] = after_consume.get(pid, 0) - weight
        enabled_before = set(net.enabled_transitions(after_consume))
        marking = net.fire(marking, transition_id)
        now = now + delta
        aged = {t: age + delta for t, age in ages.items()}
        enabled_after = net.enabled_transitions(marking)
        ages = {t: aged[t] for t in enabled_after if t in enabled_before and t in aged}
        for t in enabled_after:
            if t not in ages:
                ages[t] = Fraction(0)
        events.append(TimedEvent(now, transition_id, dict(marking)))
    return events
