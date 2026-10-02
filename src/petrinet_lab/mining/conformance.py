"""Conformance checking: PM4Py metrics plus a transparent token-replay implementation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..core.model import Marking, PetriNet


@dataclass(frozen=True)
class TraceFitness:
    trace: tuple[str, ...]
    consumed: int
    produced: int
    missing: int
    remaining: int
    fitness: float


def token_replay(
    net: PetriNet,
    traces: Iterable[Iterable[str]],
    final_marking: Mapping[str, int] | None = None,
    silent_depth: int = 8,
    silent_visits: int = 500,
) -> list[TraceFitness]:
    """Token-based replay with greedy silent moves.

    When the next activity is not enabled, a bounded breadth-first search over
    silent transitions tries to enable it. Fit traces have ``missing == 0``;
    ``remaining`` counts leftover tokens beyond the expected final marking.
    """
    label_map: dict[str, list[str]] = defaultdict(list)
    for transition_id, transition in net.transitions.items():
        if transition.name:
            label_map[transition.label].append(transition_id)
    silent = [tid for tid, transition in net.transitions.items() if not transition.name]
    expected_final = dict(final_marking or {})
    results: list[TraceFitness] = []
    for trace in traces:
        trace = tuple(trace)
        marking: Marking = net.initial_marking()
        consumed = 0
        produced = 1
        missing = 0
        for activity in trace:
            action, silent_path = _next_action(net, marking, activity, label_map, silent, silent_depth, silent_visits)
            for silent_transition in silent_path:
                marking = net.fire(marking, silent_transition)
            if action is not None:
                for pid, weight in net.preset(action).items():
                    marking[pid] -= weight
                    consumed += weight
                for pid, weight in net.postset(action).items():
                    marking[pid] = marking.get(pid, 0) + weight
                    produced += weight
                continue
            candidates = label_map.get(activity, [])
            if not candidates:
                missing += 1
                consumed += 1
                continue
            transition_id = candidates[0]
            for pid, weight in net.preset(transition_id).items():
                available = marking.get(pid, 0)
                defect = max(0, weight - available)
                missing += defect
                consumed += weight
                marking[pid] = available - weight
            for pid, weight in net.postset(transition_id).items():
                marking[pid] = marking.get(pid, 0) + weight
                produced += weight
        remaining = sum(
            max(0, count - expected_final.get(pid, 0)) for pid, count in marking.items()
        )
        fitness = 0.5 * (1 - missing / consumed if consumed else 1.0) + 0.5 * (
            1 - remaining / produced if produced else 1.0
        )
        results.append(
            TraceFitness(trace, consumed, produced, missing, remaining, max(0.0, fitness))
        )
    return results


def _next_action(net, marking, activity, label_map, silent, max_depth, max_visits):
    candidates = [tid for tid in label_map.get(activity, []) if net.enabled(marking, tid)]
    if candidates:
        return candidates[0], []
    if not silent:
        return None, []
    from collections import deque

    queue = deque([(marking, [])])
    visited = {net.marking_key(marking)}
    while queue and len(visited) < max_visits:
        current, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for silent_transition in silent:
            if not net.enabled(current, silent_transition):
                continue
            successor = net.fire(current, silent_transition)
            candidates = [
                tid for tid in label_map.get(activity, []) if net.enabled(successor, tid)
            ]
            if candidates:
                return candidates[0], path + [silent_transition]
            key = net.marking_key(successor)
            if key not in visited:
                visited.add(key)
                queue.append((successor, path + [silent_transition]))
    return None, []


def log_fitness(results: list[TraceFitness]) -> dict[str, float]:
    if not results:
        return {"log_fitness": 0.0, "average_trace_fitness": 0.0, "perc_fit_traces": 0.0}
    fitting = sum(1 for result in results if result.missing == 0)
    return {
        "log_fitness": sum(result.fitness for result in results) / len(results),
        "average_trace_fitness": sum(result.fitness for result in results) / len(results),
        "perc_fit_traces": 100.0 * fitting / len(results),
    }


def pm4py_token_replay(log, result) -> dict:
    import pm4py

    return pm4py.fitness_token_based_replay(
        log,
        result.pm4py_net,
        result.pm4py_initial_marking,
        result.pm4py_final_marking,
    )


def pm4py_alignments(log, result, max_traces: int | None = None) -> dict:
    import pm4py

    subset = log
    if max_traces is not None:
        from pm4py.objects.log.obj import EventLog

        subset = EventLog(list(log)[:max_traces])
    return pm4py.fitness_alignments(
        subset,
        result.pm4py_net,
        result.pm4py_initial_marking,
        result.pm4py_final_marking,
    )
