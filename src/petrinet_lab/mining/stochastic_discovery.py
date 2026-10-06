"""Stochastic process discovery: estimate exponential firing rates from logs.

The estimator replays each trace with token-based replay, attributes the time
elapsed between consecutive events to the marking in which the producing
transition is enabled (after any silent moves), and computes the
maximum-likelihood rate of every visible transition as

    rate(t) = firings(t) / time_during_which_t_was_enabled

Under the exponential race semantics implemented in ``core.stochastic`` this
recovers the true rates from complete event logs (see
``tests/test_stochastic_discovery.py``). The first interval of each case starts
at the first observed event, so transitions that always fire first in a case
cannot be estimated; cyclic processes (e.g. failure/repair) are unaffected.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..core.model import PetriNet
from ..core.simulation import SimulatedEvent, traces_with_timestamps
from .conformance import next_action


@dataclass
class TransitionEstimate:
    transition: str
    label: str
    firings: int = 0
    enabled_time: float = 0.0
    rate: float = 0.0


@dataclass
class StochasticDiscoveryResult:
    net: PetriNet
    rates: dict[str, float]
    estimates: dict[str, TransitionEstimate]
    choices: dict[tuple[int, ...], dict[str, int]]
    cases: int
    events: int
    unreplayable: int = 0
    algorithm: str | None = None
    fitness: dict[str, float] = field(default_factory=dict)

    def choice_probabilities(self) -> dict[tuple[int, ...], dict[str, float]]:
        """Empirical distribution of the transition that fires in each marking."""
        probabilities: dict[tuple[int, ...], dict[str, float]] = {}
        for key, counts in self.choices.items():
            total = sum(counts.values())
            if total:
                probabilities[key] = {tid: count / total for tid, count in counts.items()}
        return probabilities


def _timed_traces(log) -> list[list[tuple[str, float]]]:
    if isinstance(log, list):
        if not log:
            return []
        if isinstance(log[0], SimulatedEvent):
            return traces_with_timestamps(log)
        return [[(str(activity), float(timestamp)) for activity, timestamp in trace] for trace in log]
    import pandas as pd
    import pm4py

    dataframe = log if hasattr(log, "columns") else pm4py.convert_to_dataframe(log)
    case_key = "case:concept:name"
    activity_key = "concept:name"
    time_key = "time:timestamp"
    dataframe = dataframe.copy()
    dataframe[time_key] = pd.to_datetime(dataframe[time_key])
    traces: list[list[tuple[str, float]]] = []
    ordered = dataframe.sort_values([case_key, time_key])
    for _, group in ordered.groupby(case_key, sort=False):
        traces.append(
            [
                (str(activity), timestamp.timestamp())
                for activity, timestamp in zip(group[activity_key], group[time_key], strict=False)
            ]
        )
    return traces


def _replay_maps(
    net: PetriNet, observed: set[str]
) -> tuple[dict[str, list[str]], list[str]]:
    """Split transitions into visible (label seen in the log) and silent."""
    label_map: dict[str, list[str]] = defaultdict(list)
    silent: list[str] = []
    for transition_id, transition in net.transitions.items():
        if transition.label in observed:
            label_map[transition.label].append(transition_id)
        else:
            silent.append(transition_id)
    return label_map, silent


def estimate_rates(
    net: PetriNet,
    log,
    silent_depth: int = 8,
    silent_visits: int = 500,
) -> StochasticDiscoveryResult:
    """Estimate exponential firing rates of ``net`` from an event log.

    ``log`` may be a list of ``(activity, timestamp)`` traces, a list of
    :class:`SimulatedEvent`, a PM4Py event log or a case/activity/timestamp
    DataFrame. A transition is considered visible when its label occurs in the
    log; silent transitions never fire visible events and get no rate.
    """
    traces = _timed_traces(log)
    observed = {activity for trace in traces for activity, _ in trace}
    label_map, silent = _replay_maps(net, observed)
    estimates: dict[str, TransitionEstimate] = {}
    for transition_ids in label_map.values():
        for transition_id in transition_ids:
            transition = net.transitions[transition_id]
            estimates[transition_id] = TransitionEstimate(transition_id, transition.label)
    choices: dict[tuple[int, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unreplayable = 0
    total_events = 0
    for trace in traces:
        total_events += len(trace)
        if not trace:
            continue
        marking = net.initial_marking()
        previous = trace[0][1]
        for activity, timestamp in trace:
            elapsed = max(0.0, timestamp - previous)
            transition_id, silent_path = next_action(
                net, marking, activity, label_map, silent, silent_depth, silent_visits
            )
            for silent_transition in silent_path:
                marking = net.fire(marking, silent_transition)
            if elapsed > 0:
                for enabled_id in net.enabled_transitions(marking):
                    if enabled_id in estimates:
                        estimates[enabled_id].enabled_time += elapsed
            if transition_id is None:
                unreplayable += 1
                previous = timestamp
                continue
            if transition_id in estimates:
                estimates[transition_id].firings += 1
                marking_key = net.marking_key(marking)
                choices[marking_key][transition_id] += 1
            marking = net.fire(marking, transition_id)
            previous = timestamp
    rates: dict[str, float] = {}
    for transition_id, estimate in estimates.items():
        if estimate.firings > 0 and estimate.enabled_time > 0:
            estimate.rate = estimate.firings / estimate.enabled_time
        rates[transition_id] = estimate.rate
    fitness = {
        "replay_fitness": (1.0 - unreplayable / total_events) if total_events else 0.0
    }
    return StochasticDiscoveryResult(
        net=net,
        rates=rates,
        estimates=estimates,
        choices={key: dict(counts) for key, counts in choices.items()},
        cases=len(traces),
        events=total_events,
        unreplayable=unreplayable,
        fitness=fitness,
    )


def discover_stochastic(log, algorithm: str = "inductive", **kwargs) -> StochasticDiscoveryResult:
    """Discover a Petri net with PM4Py and estimate its exponential rates."""
    from .discovery import discover

    discovery = discover(log, algorithm=algorithm, **kwargs)
    result = estimate_rates(discovery.net, log)
    result.algorithm = algorithm
    return result


def compare_rates(
    learned: Mapping[str, float],
    reference: Mapping[str, float],
) -> dict[str, object]:
    """Relative error between estimated and reference rates on shared transitions."""
    shared = [transition_id for transition_id in reference if transition_id in learned]
    errors = {
        transition_id: abs(learned[transition_id] - reference[transition_id]) / reference[transition_id]
        for transition_id in shared
        if reference[transition_id] > 0
    }
    if not errors:
        return {"per_transition": {}, "mean_relative_error": None, "max_relative_error": None}
    values = list(errors.values())
    return {
        "per_transition": errors,
        "mean_relative_error": sum(values) / len(values),
        "max_relative_error": max(values),
    }
